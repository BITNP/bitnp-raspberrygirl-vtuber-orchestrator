from __future__ import annotations

import asyncio
import logging
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from time import monotonic_ns
from typing import TYPE_CHECKING, Literal, Protocol, final, override

from websockets.asyncio.server import serve

from orchestrator.caption_timeline import CaptionTimelineCancel, CaptionTimelineCommand
from orchestrator.comment_ingress import (
    AuthenticatedCommentIngress,
    CommentAccessToken,
    CommentIngressConfig,
    CommentTokenValue,
)
from orchestrator.control_ingress import SessionEndControl, parse_session_control
from orchestrator.frontend_effects import (
    FrontendEffectDispatcher,
    send_caption_timeline,
    send_frontend_operation,
)
from orchestrator.ids import (
    SegmentId as SchedulerSegmentId,
)
from orchestrator.ids import (
    SessionId,
    TraceId,
)
from orchestrator.ids import (
    TurnId as SchedulerTurnId,
)
from orchestrator.interaction_ingress import parse_comment_proposal
from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.sessions import EventCorrelation, EventSequence
from orchestrator.streaming_contracts import EnvelopeIdentity, StreamKey
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_control import (
    AsrFinal,
    ControlEnvelopeError,
    bearer_token_matches,
    parse_control_event,
)
from orchestrator.transport_dispatch import TransportControlDispatch
from orchestrator.transport_hub import (
    DatagramSender,
    OnsiteBridge,
    RtpHub,
)

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from websockets.http11 import Request, Response

    from orchestrator.agent_pipeline import FrontendOperation
    from orchestrator.ids import TurnId
    from orchestrator.observability import OnsiteObservability
    from orchestrator.runtime_contracts import RuntimeOutcome
    from orchestrator.scheduler_reflex import OutputLease, SchedulerOutputFence
    from orchestrator.scheduler_runtime import SessionRuntime
    from orchestrator.streaming_contracts import (
        CancellationEpoch,
        FlushClock,
        FlushFailure,
        SegmentId,
        StreamFlush,
    )
    from orchestrator.task_registry import TaskId


type DatagramListener = Callable[[str, int, RtpHub], Awaitable[DatagramSender]]


class ControlServer(Protocol):
    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...


class ControlConnection(Protocol):
    @property
    def remote_address(self) -> tuple[str, int] | None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    def respond(self, status: HTTPStatus, text: str) -> Response: ...

    async def send(self, message: str) -> None: ...


class FrontendConnection(Protocol):
    async def send(self, message: str) -> None: ...


type ControlHandler = Callable[[ControlConnection], Awaitable[None]]

type ControlListener = Callable[
    [TransportConfig, ControlHandler], Awaitable[ControlServer]
]

type SessionRuntimeFactory = Callable[[SessionId], SessionRuntime]


@dataclass(frozen=True, slots=True)
class TransportReadiness:
    listener_ready: bool

    route_ready: bool

    @property
    def ready(self) -> bool:
        return self.listener_ready


@final
class TransportRuntime:
    def __init__(
        self,
        config: TransportConfig,
        datagram_listener: DatagramListener | None = None,
        control_listener: ControlListener | None = None,
        onsite_bridge: OnsiteBridge | None = None,
        clock: FlushClock | None = None,
    ) -> None:
        self._config: TransportConfig = config

        self._datagram_listener: DatagramListener = (
            _listen_udp if datagram_listener is None else datagram_listener
        )

        self._control_listener: ControlListener = (
            _listen_control if control_listener is None else control_listener
        )

        self._hub: RtpHub = RtpHub(onsite_bridge=onsite_bridge)

        self._onsite_bridge: OnsiteBridge | None = onsite_bridge

        self._control_dispatch: TransportControlDispatch = TransportControlDispatch(
            self._hub, clock=clock
        )

        self._hub.set_output_finished_callback(
            self._control_dispatch.finish_generated_stream
        )
        self._hub.set_output_command_callback(self._control_dispatch.announce_output)
        self._hub.set_replacement_callbacks(
            self._control_dispatch.request_stream_flush,
            self._control_dispatch.admit_replacement,
        )
        self._hub.set_replacement_task_callbacks(
            self._schedule_sound_flush_task,
            self._sound_flush_task_is_current,
            self._complete_sound_flush_task,
            self._fail_sound_flush_task,
        )

        self._datagram_transport: DatagramSender | None = None

        self._control_server: ControlServer | None = None

        self._flush_driver: asyncio.Task[None] | None = None

        self._agent_tts_tasks: set[asyncio.Task[None]] = set()

        self._preoutput_agent_tts: dict[tuple[str, str], asyncio.Task[None]] = {}

        self._session_runtime: SessionRuntime | None = None

        self._session_runtimes: dict[str, SessionRuntime] = {}

        self._session_runtime_factory: SessionRuntimeFactory | None = None

        self._comment_ingresses: dict[int, AuthenticatedCommentIngress] = {}

        self._frontend_connections: dict[str, FrontendConnection] = {}

        self._active_timelines: dict[str, tuple[CaptionTimelineCommand, TurnId]] = {}

    def set_session_runtime(self, session_runtime: SessionRuntime) -> None:
        self._session_runtime = session_runtime
        self._session_runtimes[str(session_runtime.scheduler.snapshot.session_id)] = (
            session_runtime
        )

        self.set_output_fence(session_runtime.output_fence)

        session_runtime.set_preoutput_tts_cancellation(
            lambda turn_id: self._cancel_preoutput_agent_tts(
                str(session_runtime.scheduler.snapshot.session_id), str(turn_id)
            )
        )

        self._hub.set_voice_evidence_callback(session_runtime.receive_voice_evidence)

        bridge = self._onsite_bridge
        if isinstance(bridge, OnsiteExplainerBridge):
            bridge.set_asr_final_handler(self.receive_onsite_asr_final)

        session_runtime.agent_effect_dispatcher = FrontendEffectDispatcher(
            self._send_frontend_operation
        )

    def set_session_runtime_factory(self, factory: SessionRuntimeFactory) -> None:
        self._session_runtime_factory = factory

    def register_frontend_connection(
        self, session_id: SessionId, connection: FrontendConnection
    ) -> None:
        """Register the narrow outbound control surface used by timelines."""
        self._frontend_connections[str(session_id)] = connection

    async def emit_caption_timeline(
        self,
        timeline: CaptionTimelineCommand,
        session_id: SessionId,
        turn_id: TurnId,
    ) -> None:
        """Deliver a media-admitted timeline without exposing transport internals."""
        _ = await self._send_caption_timeline(timeline, session_id, turn_id)

    async def receive_onsite_asr_final(
        self, stream: StreamKey, event: ASRAudienceEvent
    ) -> bool:
        session_runtime = self._runtime_for_session(stream.session_id)
        if session_runtime is None:
            return False
        correlation = EventCorrelation(
            TraceId(f"asr-{stream.stream_id}-{event.segment_id}"),
            SessionId(stream.session_id),
            EventSequence(event.seq),
        )
        outcome = await session_runtime.receive_asr_final_async(event, correlation)
        self._schedule_agent_tts(session_runtime, outcome, stream, correlation)
        return outcome.accepted

    def _schedule_agent_tts(
        self,
        session_runtime: SessionRuntime,
        outcome: RuntimeOutcome,
        stream: StreamKey,
        correlation: EventCorrelation,
    ) -> None:
        task = asyncio.create_task(
            self._run_agent_tts(session_runtime, outcome, stream, correlation)
        )
        self._agent_tts_tasks.add(task)
        task.add_done_callback(self._agent_tts_tasks.discard)
        turn_id = outcome.turn_id
        if turn_id is None:
            return
        key = (str(session_runtime.scheduler.snapshot.session_id), str(turn_id))
        self._preoutput_agent_tts[key] = task
        task.add_done_callback(
            lambda completed, task_key=key: self._release_preoutput_agent_tts(
                task_key, completed
            )
        )

    def _cancel_preoutput_agent_tts(self, session_id: str, turn_id: str) -> None:
        task = self._preoutput_agent_tts.get((session_id, turn_id))
        if task is not None and not task.done():
            _ = task.cancel()

    def _release_preoutput_agent_tts(
        self, key: tuple[str, str], task: asyncio.Task[None]
    ) -> None:
        if self._preoutput_agent_tts.get(key) is task:
            del self._preoutput_agent_tts[key]

    async def _run_agent_tts(
        self,
        session_runtime: SessionRuntime,
        outcome: RuntimeOutcome,
        stream: StreamKey,
        correlation: EventCorrelation,
    ) -> None:
        bridge = self._onsite_bridge
        if (
            outcome.accepted
            and isinstance(bridge, OnsiteExplainerBridge)
        ):
            turn_id = outcome.turn_id
            if turn_id is None:
                return

            def timeline_started() -> None:
                scheduled = session_runtime.schedule_started_timeline(
                    turn_id,
                    audio_stream_id=f"agent-{turn_id}",
                    correlation=correlation,
                )
                if scheduled is None:
                    return
                task_id, timeline = scheduled
                task = asyncio.create_task(
                    self._run_caption_timeline_delivery(
                        session_runtime, task_id, timeline, turn_id, correlation
                    )
                )
                self._agent_tts_tasks.add(task)
                task.add_done_callback(self._agent_tts_tasks.discard)

            _ = await session_runtime.run_agent_tts_for_turn(
                turn_id,
                lambda text, output_started: bridge.speak_response(
                    stream,
                    text,
                    session_runtime.cancellation_epoch,
                    str(turn_id),
                    output_started,
                ),
                correlation,
                timeline_started,
            )

    def set_observability(self, observability: OnsiteObservability) -> None:
        self._hub.set_observability(observability)

        self._control_dispatch.set_observability(observability)

        bridge = self._onsite_bridge

        if isinstance(bridge, OnsiteExplainerBridge):
            bridge.set_observability(observability)

    def set_output_fence(self, output_fence: SchedulerOutputFence) -> None:
        self._hub.set_output_fence(output_fence)

        self._control_dispatch.set_output_fence(output_fence)

    async def start(self) -> None:
        self._datagram_transport = await self._datagram_listener(
            self._config.udp_bind_host,
            self._config.udp_bind_port,
            self._hub,
        )

        self._hub.attach_transport(self._datagram_transport)

        self._control_server = await self._control_listener(
            self._config, self.handle_control
        )

        self._flush_driver = asyncio.create_task(self._drive_flush_admission())
        _LOGGER.debug(
            "transport_started udp=%s:%d control=%s:%d",
            self._config.udp_bind_host,
            self._config.udp_bind_port,
            self._config.control_bind_host,
            self._config.control_bind_port,
        )

    async def cancel_stream(self, session_id: str, stream_id: str) -> None:
        await self._control_dispatch.cancel_stream(session_id, stream_id)

    async def request_stream_flush(self, flush: StreamFlush) -> None:
        await self._control_dispatch.request_stream_flush(flush)

    async def advance_flush_admission(self) -> None:
        await self._control_dispatch.advance_flush_admission()

    async def admit_replacement(self, flush: StreamFlush) -> bool:
        return await self._control_dispatch.admit_replacement(flush)

    async def begin_onsite_replacement(
        self, stream: StreamKey, segment_id: SegmentId
    ) -> CancellationEpoch | None:
        return await self._hub.begin_onsite_replacement(stream, segment_id)

    def _schedule_sound_flush_task(
        self, stream: StreamKey, replacement: OutputLease, flush: StreamFlush
    ) -> TaskId | Literal[False] | None:
        """Admit a replacement cutover through the owning session runtime."""
        session_runtime = self._runtime_for_session(stream.session_id)
        correlation = flush.correlation
        if session_runtime is None or correlation is None:
            # Standalone transport operation has no scheduler state to own.  The
            # Hub retains its compatibility path for that deliberately narrow
            # contract-test mode.
            return None
        task_id = session_runtime.schedule_sound_flush(
            SchedulerTurnId(str(replacement.turn_id)),
            SchedulerSegmentId(str(replacement.segment_id)),
            request_id=str(flush.request_id),
            correlation=_event_correlation(correlation),
        )
        return False if task_id is None else task_id

    def _sound_flush_task_is_current(self, stream: StreamKey, task_id: TaskId) -> bool:
        session_runtime = self._runtime_for_session(stream.session_id)
        return (
            session_runtime is not None
            and session_runtime.sound_flush_is_current(task_id)
        )

    def _complete_sound_flush_task(self, stream: StreamKey, task_id: TaskId) -> bool:
        session_runtime = self._runtime_for_session(stream.session_id)
        correlation = self._hub.correlation(stream)
        return (
            session_runtime is not None
            and correlation is not None
            and session_runtime.complete_sound_flush(
                task_id, _event_correlation(correlation)
            )
        )

    def _fail_sound_flush_task(
        self, stream: StreamKey, task_id: TaskId, reason: str
    ) -> None:
        session_runtime = self._runtime_for_session(stream.session_id)
        if session_runtime is not None:
            session_runtime.fail_sound_flush(task_id, reason=reason)

    @property
    def flush_failures(self) -> tuple[FlushFailure, ...]:
        return self._control_dispatch.flush_failures

    def readiness(self) -> TransportReadiness:
        listener_ready = (
            self._datagram_transport is not None and self._control_server is not None
        )

        return TransportReadiness(listener_ready, self._hub.route_ready)

    async def close(self) -> None:
        flush_driver = self._flush_driver

        if flush_driver is not None:
            _ = flush_driver.cancel()

            with suppress(asyncio.CancelledError):
                await flush_driver

            self._flush_driver = None

        agent_tts_tasks = tuple(self._agent_tts_tasks)
        for task in agent_tts_tasks:
            _ = task.cancel()
        if agent_tts_tasks:
            _ = await asyncio.gather(*agent_tts_tasks, return_exceptions=True)

        for session_id in tuple(self._active_timelines):
            await self._cancel_caption_timeline(
                SessionId(session_id), reason="transport_closed"
            )

        if self._control_server is not None:
            self._control_server.close()

            await self._control_server.wait_closed()

            self._control_server = None

        if self._datagram_transport is not None:
            self._datagram_transport.close()

            self._datagram_transport = None

        self._hub.clear()

        await self._hub.wait_for_onsite_jobs()

        bridge = self._onsite_bridge
        if isinstance(bridge, OnsiteExplainerBridge):
            await bridge.aclose()

        self._control_dispatch.clear()

    async def handle_control(  # noqa: C901, PLR0912, PLR0915
        self, connection: ControlConnection
    ) -> None:
        peer_ip = _peer_ip(connection)

        try:
            async for message in connection:
                if isinstance(message, str):
                    _LOGGER.debug(
                        "control_received peer=%s bytes=%d", peer_ip, len(message)
                    )
                    if not bearer_token_matches(
                        self._config.control_token,
                        _connection_authorization(connection),
                    ):
                        continue

                    frontend_session = _frontend_registration(message)
                    if frontend_session is not None:
                        self._frontend_connections[frontend_session] = connection
                        continue

                    session_runtime = self._runtime_for_message(message)

                    if session_runtime is not None and parse_comment_proposal(message):
                        _LOGGER.debug(
                            "control_received kind=audience.input peer=%s", peer_ip
                        )
                        await self._receive_comment(
                            connection, session_runtime, message
                        )

                        continue

                    if session_runtime is not None and session_runtime.receive_control(
                        message
                    ):
                        continue

                    # Mic ASR is the only voice ingress. A partial is parsed by
                    # the control dispatcher for diagnostics only; a final is
                    # route/epoch/replay checked before it reaches the shared
                    # Gate and Brain pipeline.
                    try:
                        control_event = parse_control_event(message)
                    except (ControlEnvelopeError, JsonBoundaryError):
                        control_event = None
                    if isinstance(control_event, AsrFinal):
                        _LOGGER.debug(
                            "mic_asr_final_received session=%s stream=%s segment=%s seq=%s text=%r",  # noqa: E501
                            control_event.session_id,
                            control_event.stream_id,
                            control_event.segment_id,
                            control_event.correlation.seq,
                            control_event.text,
                        )
                        if not self._hub.accept_asr_final(control_event):
                            _LOGGER.debug(
                                "mic_asr_final_rejected session=%s stream=%s segment=%s",  # noqa: E501
                                control_event.session_id,
                                control_event.stream_id,
                                control_event.segment_id,
                            )
                            continue
                        runtime = self._runtime_for_session(control_event.session_id)
                        if runtime is None:
                            continue
                        event = ASRAudienceEvent(
                            text=control_event.text,
                            received_at_ms=control_event.received_at_ms,
                            segment_id=control_event.segment_id,
                            seq=control_event.correlation.seq,
                        )
                        correlation = EventCorrelation(
                            TraceId(control_event.correlation.trace_id),
                            SessionId(control_event.session_id),
                            EventSequence(control_event.correlation.seq),
                        )
                        outcome = await runtime.receive_asr_final_async(
                            event, correlation
                        )
                        self._schedule_agent_tts(
                            runtime,
                            outcome,
                            StreamKey(
                                control_event.session_id, control_event.stream_id
                            ),
                            correlation,
                        )
                        _LOGGER.debug(
                            "mic_asr_final_processed session=%s segment=%s accepted=%s turn=%s",  # noqa: E501
                            control_event.session_id,
                            control_event.segment_id,
                            outcome.accepted,
                            outcome.turn_id,
                        )
                        continue

                    if session_runtime is not None:
                        control = parse_session_control(message)

                        if control is not None:
                            outcome = await (
                                session_runtime.receive_session_control_async(control)
                            )
                            if (
                                isinstance(control, SessionEndControl)
                                and outcome.accepted
                            ):
                                await self._cancel_caption_timeline(
                                    session_runtime.scheduler.snapshot.session_id,
                                    reason="session_ended",
                                )

                            continue

                    try:
                        await self._control_dispatch.register(
                            message,
                            peer_ip,
                            connection,
                        )
                        _LOGGER.debug("control_dispatched peer=%s", peer_ip)

                    except (ControlEnvelopeError, JsonBoundaryError):
                        continue

        finally:
            comment_ingress = self._comment_ingresses.pop(id(connection), None)

            if comment_ingress is not None:
                comment_ingress.cancel_pending()

            for session_id, frontend in tuple(self._frontend_connections.items()):
                if frontend is connection:
                    del self._frontend_connections[session_id]

            self._control_dispatch.remove_connection(connection)

    def _runtime_for_message(self, raw_message: str) -> SessionRuntime | None:
        try:
            value = parse_json_value(raw_message)
        except JsonBoundaryError:
            return None
        if not isinstance(value, dict):
            return None
        session_id = value.get("session_id")
        if not isinstance(session_id, str):
            return None
        return self._runtime_for_session(session_id, fallback=self._session_runtime)

    def _runtime_for_session(
        self, session_id: str, *, fallback: SessionRuntime | None = None
    ) -> SessionRuntime | None:
        runtime = self._session_runtimes.get(session_id)
        if runtime is not None:
            return runtime
        factory = self._session_runtime_factory
        if factory is None:
            return fallback
        runtime = factory(SessionId(session_id))
        self.set_session_runtime(runtime)
        return runtime

    async def _send_frontend_operation(
        self,
        event_type: str,
        operation: FrontendOperation,
        session_id: SessionId,
        turn_id: TurnId,
    ) -> None:
        connection = self._frontend_connections.get(str(session_id))
        if connection is None:
            return
        await send_frontend_operation(
            connection.send, event_type, operation, session_id, turn_id
        )

    async def _send_caption_timeline(
        self,
        timeline: CaptionTimelineCommand,
        session_id: SessionId,
        turn_id: TurnId,
    ) -> bool:
        connection = self._frontend_connections.get(str(session_id))
        if connection is None:
            _LOGGER.debug(
                "caption_timeline_dropped frontend_unavailable session=%s turn=%s",
                session_id,
                turn_id,
            )
            return False
        previous = self._active_timelines.get(str(session_id))
        if previous is not None and previous[0].timeline_id != timeline.timeline_id:
            await self._cancel_caption_timeline(
                session_id,
                reason="replaced",
                connection=connection,
            )
        try:
            await send_caption_timeline(connection.send, timeline, session_id, turn_id)
        except (OSError, ValueError):
            _LOGGER.debug(
                "caption_timeline_delivery_failed session=%s turn=%s timeline=%s",
                session_id,
                turn_id,
                timeline.timeline_id,
                exc_info=True,
            )
            return False
        self._active_timelines[str(session_id)] = (timeline, turn_id)
        return True

    async def _run_caption_timeline_delivery(
        self,
        session_runtime: SessionRuntime,
        task_id: TaskId,
        timeline: CaptionTimelineCommand,
        turn_id: TurnId,
        correlation: EventCorrelation,
    ) -> None:
        """Run frontend delivery as a fenced scheduler task.

        Frontend availability is intentionally independent of audio playback:
        any delivery failure only closes this task and is never propagated to
        the Sound/TTS path.
        """
        if not session_runtime.caption_timeline_delivery_is_current(task_id):
            session_runtime.fail_caption_timeline_delivery(task_id, reason="stale")
            return
        try:
            delivered = await self._send_caption_timeline(
                timeline,
                session_runtime.scheduler.snapshot.session_id,
                turn_id,
            )
        except (OSError, ValueError):
            session_runtime.fail_caption_timeline_delivery(
                task_id, reason="frontend_delivery_failed"
            )
            _LOGGER.debug(
                "caption_timeline_task_failed task=%s", task_id, exc_info=True
            )
            return
        if not delivered:
            session_runtime.fail_caption_timeline_delivery(
                task_id, reason="frontend_unavailable"
            )
            return
        if not session_runtime.complete_caption_timeline_delivery(task_id, correlation):
            return

    async def _cancel_caption_timeline(
        self,
        session_id: SessionId,
        *,
        reason: str,
        connection: FrontendConnection | None = None,
    ) -> None:
        active = self._active_timelines.pop(str(session_id), None)
        if active is None:
            return
        timeline, turn_id = active
        target = (
            self._frontend_connections.get(str(session_id))
            if connection is None
            else connection
        )
        if target is None:
            return
        cancel = CaptionTimelineCancel(
            timeline_id=timeline.timeline_id,
            audio_stream_id=timeline.audio_stream_id,
            cancellation_epoch=timeline.cancellation_epoch,
            reason=reason,
        )
        try:
            await send_caption_timeline(target.send, cancel, session_id, turn_id)
        except (OSError, ValueError):
            _LOGGER.debug(
                "caption_timeline_cancel_failed session=%s timeline=%s",
                session_id,
                timeline.timeline_id,
                exc_info=True,
            )

    async def _receive_comment(
        self,
        connection: ControlConnection,
        session_runtime: SessionRuntime,
        message: str,
    ) -> None:
        ingress = self._comment_ingresses.setdefault(
            id(connection),
            AuthenticatedCommentIngress(
                session_runtime.interaction_ingress,
                _comment_ingress_config(self._config),
            ),
        )

        receipt = ingress.receive(
            message,
            _connection_authorization(connection),
            now_ms=monotonic_ns() // 1_000_000,
        )

        if not receipt.accepted:
            return

        while (proposal := ingress.take_next()) is not None:
            _ = await session_runtime.receive_comment_async(proposal)

    def route_datagram(self, data: bytes, peer: tuple[str, int]) -> bool:
        routed = self._hub.route_datagram(data, peer)
        _LOGGER.debug(
            "rtp_received peer=%s:%d bytes=%d routed=%s",
            peer[0],
            peer[1],
            len(data),
            routed,
        )
        return routed

    async def wait_for_onsite_jobs(self) -> None:
        await self._hub.wait_for_onsite_jobs()

    async def _drive_flush_admission(self) -> None:
        while True:
            await asyncio.sleep(0.25)

            await self._control_dispatch.advance_flush_admission()


@final
class _RtpDatagramProtocol(asyncio.DatagramProtocol):
    def __init__(self, hub: RtpHub) -> None:
        self._hub: RtpHub = hub

    @override
    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        _ = self._hub.route_datagram(data, addr)


async def _listen_udp(host: str, port: int, hub: RtpHub) -> DatagramSender:
    loop = asyncio.get_running_loop()

    transport, _ = await loop.create_datagram_endpoint(
        lambda: _RtpDatagramProtocol(hub),
        local_addr=(host, port),
    )

    return transport


async def _listen_control(
    config: TransportConfig, handler: ControlHandler
) -> ControlServer:
    ssl_context = _ssl_context(config)

    def authorize(connection: ControlConnection, request: Request) -> Response | None:
        authorization = request.headers.get("Authorization")

        if bearer_token_matches(config.control_token, authorization):
            return None

        return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")

    return await serve(
        handler,
        config.control_bind_host,
        config.control_bind_port,
        process_request=authorize,
        ssl=ssl_context,
    )


def _event_correlation(correlation: EnvelopeIdentity) -> EventCorrelation:
    """Convert an authenticated transport envelope into scheduler correlation."""
    return EventCorrelation(
        TraceId(correlation.trace_id),
        SessionId(correlation.session_id),
        EventSequence(correlation.seq),
    )


def _comment_ingress_config(config: TransportConfig) -> CommentIngressConfig:
    token = config.control_token

    credential = (
        None
        if token is None
        else CommentAccessToken(CommentTokenValue(token), (1 << 63) - 1)
    )

    return CommentIngressConfig(
        token=credential,
        replay_window=128,
        max_payload_bytes=16_384,
        max_pending=16,
    )


def _connection_authorization(connection: ControlConnection) -> str | None:
    authorization = getattr(connection, "authorization", None)

    if isinstance(authorization, str):
        return authorization

    request = getattr(connection, "request", None)

    headers = getattr(request, "headers", None)

    candidate = getattr(headers, "get", None)

    value = candidate("Authorization") if callable(candidate) else None

    return value if isinstance(value, str) else None


def _ssl_context(config: TransportConfig) -> ssl.SSLContext | None:
    if config.control_scheme == "ws":
        return None

    if config.tls_cert_path is None or config.tls_key_path is None:
        raise ControlEnvelopeError(field_name="tls")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    context.load_cert_chain(config.tls_cert_path, config.tls_key_path)

    return context


def _frontend_registration(raw_message: str) -> str | None:
    try:
        value = parse_json_value(raw_message)
    except JsonBoundaryError:
        return None
    if not isinstance(value, dict):
        return None
    if (
        value.get("event_type") != "frontend.register"
        or value.get("source") != "frontend"
        or value.get("data") != {}
    ):
        return None
    session_id = value.get("session_id")
    return session_id if isinstance(session_id, str) and session_id.strip() else None


def _peer_ip(connection: ControlConnection) -> str:
    remote_address = connection.remote_address

    if remote_address is None:
        raise ControlEnvelopeError(field_name="peer")

    return remote_address[0]
