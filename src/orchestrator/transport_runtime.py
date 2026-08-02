from __future__ import annotations

import asyncio
import logging
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from time import monotonic_ns
from typing import TYPE_CHECKING, Protocol, final, override

from websockets.asyncio.server import serve

from orchestrator.comment_ingress import (
    AuthenticatedCommentIngress,
    CommentAccessToken,
    CommentIngressConfig,
    CommentTokenValue,
)
from orchestrator.control_ingress import parse_session_control
from orchestrator.frontend_effects import (
    FrontendEffectDispatcher,
    send_frontend_operation,
)
from orchestrator.interaction_ingress import parse_comment_proposal
from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_control import ControlEnvelopeError, bearer_token_matches
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
    from orchestrator.ids import SessionId, TurnId
    from orchestrator.observability import OnsiteObservability
    from orchestrator.scheduler_reflex import SchedulerOutputFence
    from orchestrator.scheduler_runtime import SessionRuntime
    from orchestrator.streaming_contracts import (
        CancellationEpoch,
        FlushClock,
        FlushFailure,
        SegmentId,
        StreamFlush,
        StreamKey,
    )


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


type ControlHandler = Callable[[ControlConnection], Awaitable[None]]

type ControlListener = Callable[
    [TransportConfig, ControlHandler], Awaitable[ControlServer]
]


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

        self._datagram_transport: DatagramSender | None = None

        self._control_server: ControlServer | None = None

        self._flush_driver: asyncio.Task[None] | None = None

        self._session_runtime: SessionRuntime | None = None

        self._comment_ingresses: dict[int, AuthenticatedCommentIngress] = {}

        self._frontend_connections: dict[str, ControlConnection] = {}

    def set_session_runtime(self, session_runtime: SessionRuntime) -> None:
        self._session_runtime = session_runtime

        self.set_output_fence(session_runtime.output_fence)

        self._hub.set_voice_evidence_callback(session_runtime.receive_voice_evidence)

        session_runtime.agent_effect_dispatcher = FrontendEffectDispatcher(
            self._send_frontend_operation
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

    async def handle_control(  # noqa: C901, PLR0912
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

                    session_runtime = self._session_runtime

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

                    if session_runtime is not None:
                        control = parse_session_control(message)

                        if control is not None:
                            _ = await session_runtime.receive_session_control_async(
                                control
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
