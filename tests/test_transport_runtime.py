from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

import pytest

from orchestrator.caption_timeline import CaptionTimelineCommand
from orchestrator.config import TrustedLanToken
from orchestrator.control_ingress import PresentationResultControl
from orchestrator.ids import ConnectionId, SessionId, TraceId
from orchestrator.ids import TurnId as AgentTurnId
from orchestrator.interactions import (
    CommandId,
    PresentationCommand,
    PresentationCommandKind,
    PresentationResult,
)
from orchestrator.json_boundary import parse_json_value
from orchestrator.mcp_adapters import DeckDispatchIntent, DeckEffectResultKind
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import ProviderCancellationHandle
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    SegmentId,
    StreamKey,
    TurnId,
)
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_control import (
    AuthenticatedControl,
    EnvelopeCorrelation,
    MicInputRegistration,
    VoiceEvidence,
)
from orchestrator.transport_dispatch import TransportControlDispatch
from orchestrator.transport_hub import DuplicateRouteError, RtpHub
from orchestrator.transport_runtime import (
    ControlHandler,
    ControlListener,
    DatagramListener,
    TransportRuntime,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from http import HTTPStatus

    from websockets.http11 import Response

    from orchestrator.json_boundary import JsonValue


SESSION_ID: Final = "session-transport-001"

STREAM_ID: Final = "mic-stream-001"

SSRC: Final = 0x12345678

SOURCE_PEER: Final = ("192.0.2.10", 41_000)

SINK_PEER: Final = ("192.0.2.11", 41_001)


@dataclass
class FakeDatagramTransport:
    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)

    closed: bool = False

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:

        self.sent.append((data, addr))

    def close(self) -> None:

        self.closed = True


@dataclass
class FakeControlServer:
    closed: bool = False

    waited: bool = False

    def close(self) -> None:

        self.closed = True

    async def wait_closed(self) -> None:

        self.waited = True


@dataclass
class RecordingControlPeer:
    messages: list[str] = field(default_factory=list)

    async def send(self, message: str) -> None:

        self.messages.append(message)


@dataclass
class _RouteWithoutEpochAdvance:
    registrations: int = 0

    def register_control(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs
        self.registrations += 1

    def remove_connection(self, owner: ConnectionId) -> None:
        _ = owner

    def remove_stream(self, session_id: str, stream_id: str) -> None:
        _ = session_id, stream_id

    def output_ssrc(self, stream: StreamKey, cancellation_epoch: int = 0) -> int:
        _ = stream, cancellation_epoch
        return 0

    def correlation(self, stream: StreamKey) -> EnvelopeCorrelation | None:
        return EnvelopeCorrelation("trace", stream.session_id, 1)

    def input_epoch(self, stream: StreamKey) -> int:
        _ = stream
        return 1

    def owns_mic_input(self, stream: StreamKey, owner: ConnectionId) -> bool:
        _ = stream, owner
        return True


@dataclass
class _RecordingOutputFence:
    finishes: list[dict[str, object]] = field(default_factory=list)

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> bool:
        _ = acknowledgement
        return True

    def finish(
        self,
        *,
        stream: StreamKey,
        turn_id: TurnId | None,
        segment_id: SegmentId | None,
        cancellation_epoch: CancellationEpoch | None,
    ) -> bool:
        self.finishes.append(
            {
                "stream": stream,
                "turn_id": turn_id,
                "segment_id": segment_id,
                "cancellation_epoch": cancellation_epoch,
            }
        )
        return True


@dataclass
class _ControlConnection:
    messages: tuple[str, ...]

    sent: list[str] = field(default_factory=list)

    authorization: str | None = None

    @property
    def remote_address(self) -> tuple[str, int]:

        return ("127.0.0.1", 443)

    async def __aiter__(self) -> AsyncIterator[str]:

        for message in self.messages:
            yield message

    async def send(self, message: str) -> None:

        self.sent.append(message)

    def respond(self, status: HTTPStatus, text: str) -> Response:

        _ = status

        _ = text

        raise AssertionError


@dataclass
class _IncrementingClock:
    current: int = 0

    def now(self) -> int:

        value = self.current

        self.current += 10_000

        return value


def test_hub_does_not_forward_valid_mic_rtp_without_an_onsite_bridge() -> None:
    # Given: authenticated Mic and Sound registrations for one canonical stream.

    transport = FakeDatagramTransport()

    hub = RtpHub(transport)

    hub.register_control(_source_registration(), SOURCE_PEER[0])

    hub.register_control(_sink_registration(), SINK_PEER[0])

    packet = _rtp_packet(payload_type=96)

    # When: the first valid Mic packet arrives without an onsite bridge.

    forwarded = hub.route_datagram(packet, SOURCE_PEER)

    # Then: the packet is not sent to Sound.

    assert forwarded is False

    assert transport.sent == []


def test_hub_rejects_mic_rtp_after_control_input_and_sink_readiness() -> None:
    # Given: registered Mic control input and a Sound output route.

    transport = FakeDatagramTransport()

    hub = RtpHub(transport)

    hub.register_control(_source_registration(), SOURCE_PEER[0])

    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: Sound acknowledges its output readiness.

    hub.register_control(_sink_ready(), SINK_PEER[0])

    # Then: readiness cannot enable a raw Mic-to-Sound fallback.

    assert hub.route_datagram(_rtp_packet(payload_type=96), SOURCE_PEER) is False

    assert transport.sent == []


def test_hub_rejects_invalid_rtp() -> None:
    # Given: a registered stream with a malformed RTP payload type.

    transport = FakeDatagramTransport()

    hub = RtpHub(transport)

    hub.register_control(_source_registration(), SOURCE_PEER[0])

    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: the packet arrives from the registered Mic endpoint.

    forwarded = hub.route_datagram(_rtp_packet(payload_type=97), SOURCE_PEER)

    # Then: no UDP bytes are sent to the sink.

    assert forwarded is False

    assert transport.sent == []


def test_hub_rejects_rtp_from_unregistered_peer() -> None:
    # Given: a registered stream and a valid packet from the Sound peer instead of Mic.

    transport = FakeDatagramTransport()

    hub = RtpHub(transport)

    hub.register_control(_source_registration(), SOURCE_PEER[0])

    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: the valid packet arrives from an IP not registered as its source.

    forwarded = hub.route_datagram(_rtp_packet(payload_type=96), SINK_PEER)

    # Then: no UDP bytes are sent to the sink.

    assert forwarded is False

    assert transport.sent == []


def test_hub_registers_authenticated_control_envelopes_and_rejects_duplicates() -> None:
    # Given: a hub receiving the canonical source and sink envelopes from WSS peers.

    hub = RtpHub(FakeDatagramTransport())

    hub.register_control(_source_registration(), SOURCE_PEER[0])

    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: Sound repeats an existing output route.

    # Then: output route duplication is refused, while Mic registration is idempotent.
    hub.register_control(_source_registration(), SOURCE_PEER[0])

    with pytest.raises(DuplicateRouteError):
        hub.register_control(_sink_registration(), SINK_PEER[0])


def test_authenticated_control_registers_only_matching_bearer_token() -> None:
    # Given: a production bearer token protecting a new Mic source route.

    transport = FakeDatagramTransport()

    hub = RtpHub(transport)

    control = AuthenticatedControl(hub, TrustedLanToken("transport-test-token"))

    # When: Mic supplies its canonical envelope with the matching bearer value.

    accepted = control.register(
        _source_registration(),
        SOURCE_PEER[0],
        "Bearer transport-test-token",
    )

    rejected = control.register(
        _sink_registration(),
        SINK_PEER[0],
        "Bearer wrong-token",
    )

    # Then: only the authenticated route is retained and no unauthenticated sink exists.

    assert accepted is True

    assert rejected is False

    assert hub.route_datagram(_rtp_packet(payload_type=96), SOURCE_PEER) is False


def test_hub_removes_stream_routes_when_sound_cancels_stream() -> None:
    # Given: a pinned source route and registered sink route.

    transport = FakeDatagramTransport()

    hub = RtpHub(transport)

    hub.register_control(_source_registration(), SOURCE_PEER[0])

    hub.register_control(_sink_registration(), SINK_PEER[0])

    assert hub.route_datagram(_rtp_packet(payload_type=96), SOURCE_PEER) is False

    # When: Sound reports the canonical cancelled stream state.

    hub.register_control(_stream_state("cancelled"), SINK_PEER[0])

    # Then: the removed route cannot forward further RTP packets.

    assert hub.route_datagram(_rtp_packet(payload_type=96), SOURCE_PEER) is False

    assert transport.sent == []


def test_dispatch_never_releases_retired_mic_rtp_source() -> None:

    async def verify_startup_gate() -> None:

        dispatcher = TransportControlDispatch(RtpHub())

        source = RecordingControlPeer()

        sink = RecordingControlPeer()

        await dispatcher.register(_sink_registration(), SINK_PEER[0], sink)

        await dispatcher.register(_source_registration(), SOURCE_PEER[0], source)

        assert len(source.messages) == 1
        ready = parse_json_value(source.messages[0])
        assert isinstance(ready, dict)
        assert ready["event_type"] == "mic.input.ready"
        assert ready["data"] == {"stream_id": STREAM_ID, "input_epoch": 1}

        # Source registration is compatibility-only and cannot start media.
        assert sink.messages == []

        await dispatcher.register(_sink_ready(), SINK_PEER[0], sink)

        assert len(source.messages) == 1

    asyncio.run(verify_startup_gate())


def test_output_announcement_waits_for_sound_ready_before_admitting_rtp() -> None:
    async def verify_ready_gate() -> None:
        hub = RtpHub()
        dispatcher = TransportControlDispatch(hub)
        source = RecordingControlPeer()
        sink = RecordingControlPeer()

        await dispatcher.register(_sink_registration(), SINK_PEER[0], sink)
        await dispatcher.register(_source_registration(), SOURCE_PEER[0], source)

        announcement = asyncio.create_task(
            dispatcher.announce_output(StreamKey(SESSION_ID, STREAM_ID), 0)
        )
        await asyncio.sleep(0)

        assert len(sink.messages) == 1
        assert announcement.done() is False

        await dispatcher.register(_sink_ready(), SINK_PEER[0], sink)
        await announcement

    asyncio.run(verify_ready_gate())


def test_finished_playback_never_requires_mic_epoch_advance() -> None:
    route = _RouteWithoutEpochAdvance()
    dispatcher = TransportControlDispatch(route)
    fence = _RecordingOutputFence()
    dispatcher.set_output_fence(fence)
    peer = RecordingControlPeer()
    mic = RecordingControlPeer()

    asyncio.run(dispatcher.register(_sink_registration(), SINK_PEER[0], peer))
    asyncio.run(dispatcher.register(_source_registration(), SOURCE_PEER[0], mic))

    asyncio.run(
        dispatcher.register(
            _envelope(
                "media.stream.state",
                "sound",
                {
                    "command_id": f"rtp-{STREAM_ID}-0",
                    "stream_id": STREAM_ID,
                    "state": "finished",
                    "cancellation_epoch": 0,
                },
            ),
            SINK_PEER[0],
            peer,
        )
    )

    assert route.registrations == 3
    assert len(fence.finishes) == 1


def test_finished_playback_notifies_turn_reducer_only_after_fence_accepts() -> None:
    route = _RouteWithoutEpochAdvance()
    dispatcher = TransportControlDispatch(route)
    fence = _RecordingOutputFence()
    dispatcher.set_output_fence(fence)
    finished_streams: list[StreamKey] = []
    dispatcher.set_playback_finished_callback(
        finished_streams.append
    )
    peer = RecordingControlPeer()
    mic = RecordingControlPeer()
    asyncio.run(dispatcher.register(_sink_registration(), SINK_PEER[0], peer))
    asyncio.run(dispatcher.register(_source_registration(), SOURCE_PEER[0], mic))

    asyncio.run(
        dispatcher.register(
            _envelope(
                "media.stream.state",
                "sound",
                {
                    "command_id": f"rtp-{STREAM_ID}-0",
                    "stream_id": STREAM_ID,
                    "state": "finished",
                    "cancellation_epoch": 0,
                },
            ),
            SINK_PEER[0],
            peer,
        )
    )

    assert finished_streams == [
        StreamKey(session_id=SESSION_ID, stream_id=STREAM_ID)
    ]


def test_presentation_dispatch_waits_for_owning_frontend_result() -> None:
    async def verify() -> None:
        runtime = TransportRuntime(_loopback_config())
        session_runtime = SessionRuntime.create(
            session_id=SessionId(SESSION_ID),
            turn_id_prefix="turn",
            task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        )
        runtime.set_session_runtime(session_runtime)
        frontend = _ControlConnection(())
        wrong_frontend = _ControlConnection(())
        runtime.register_frontend_connection(SessionId(SESSION_ID), frontend)
        command = PresentationCommand(
            PresentationCommandKind.LOAD,
            "deck-1",
            1,
            CommandId("presentation-1"),
        )
        task = asyncio.create_task(
            session_runtime.deck_dispatcher.executor.dispatch_async(
                DeckDispatchIntent(command, 10**15),
                ProviderCancellationHandle(),
            )
        )
        await asyncio.sleep(0)
        assert len(frontend.sent) == 1
        envelope = parse_json_value(frontend.sent[0])
        assert isinstance(envelope, dict)
        assert envelope["event_type"] == "presentation.load.command"
        result = PresentationResultControl(
            PresentationResult(command.command_id, succeeded=True),
            EventCorrelation(
                TraceId("frontend-result"),
                SessionId(SESSION_ID),
                EventSequence(1),
            ),
        )
        assert not runtime.accept_presentation_result(result, wrong_frontend)
        assert runtime.accept_presentation_result(result, frontend)
        assert (await task).kind is DeckEffectResultKind.SUCCEEDED

    asyncio.run(verify())


def test_runtime_reports_ready_after_listeners_start_and_closes_them() -> None:
    # Given: injected control and datagram listeners for an explicit loopback runtime.

    datagram_transport = FakeDatagramTransport()

    control_server = FakeControlServer()

    runtime = TransportRuntime(
        _loopback_config(),
        datagram_listener=_fake_datagram_listener(datagram_transport),
        control_listener=_fake_control_listener(control_server),
    )

    # When: the runtime starts then receives its shutdown signal.

    asyncio.run(runtime.start())

    ready = runtime.readiness()

    asyncio.run(runtime.close())

    # Then: readiness requires both listeners and shutdown closes each resource.

    assert ready.ready is True

    assert control_server.closed is True

    assert control_server.waited is True

    assert datagram_transport.closed is True


def test_hub_routes_voice_evidence_only_to_the_matching_session_runtime() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId(SESSION_ID),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )
    hub = RtpHub()
    hub.set_voice_evidence_callback(SESSION_ID, runtime.receive_voice_evidence)
    evidence = VoiceEvidence(
        session_id=SESSION_ID,
        evidence_id="voice-evidence-1",
        stream_id=STREAM_ID,
        input_epoch=1,
        rtp_start_timestamp=10,
        rtp_end_timestamp=330,
        embedding_model_revision="camplusplus-onnx-v1",
        embedding=(0.1, 0.2),
        speech_ms=20,
        quality_score=0.9,
        correlation=EnvelopeCorrelation("trace", SESSION_ID, 1),
    )

    hub.register_control(evidence, SOURCE_PEER[0])

    assert runtime.voice_evidence_ranges == ((STREAM_ID, 10, 330),)


def test_control_connection_rejects_comments_without_response_coordinator() -> None:
    # Given: one real transport control loop bound to its production session runtime.

    runtime = TransportRuntime(_loopback_config())

    session_runtime = SessionRuntime.create(
        session_id=SessionId(SESSION_ID),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )

    runtime.set_session_runtime(session_runtime)

    valid = _audience_comment(SESSION_ID, "trace-comment", 7)

    foreign = _audience_comment("foreign-session", "trace-foreign", 8)

    connection = _ControlConnection((valid, valid, "{", "not-control", foreign))

    # When: valid, replayed, malformed, non-media, and foreign frames share one loop.

    asyncio.run(runtime.handle_control(connection))

    # Then: no incomplete legacy fallback is available at this boundary.

    observables = session_runtime.observables

    assert observables.dispatches == ()

    assert observables.snapshot.revision == 0

    assert observables.snapshot.active_turn_id is None

    assert [rejection.correlation.trace_id for rejection in observables.rejections] == [
        "trace-comment"
    ]

    assert observables.task_commits == ()

    assert observables.generated_rtp == ()

    assert observables.sound_transitions == ()

    assert connection.sent == []


def test_onsite_asr_final_is_routed_to_matching_session_runtime() -> None:
    runtime = TransportRuntime(_loopback_config())
    session_runtime = SessionRuntime.create(
        session_id=SessionId(SESSION_ID),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )
    runtime.set_session_runtime(session_runtime)

    async def run() -> bool:
        return await runtime.receive_onsite_asr_final(
            StreamKey(SESSION_ID, STREAM_ID),
            ASRAudienceEvent("请介绍 BitNet", 20, "asr-1", 1),
        )

    assert not asyncio.run(run())
    assert session_runtime.observables.dispatches == ()


def test_onsite_asr_final_cannot_create_an_unregistered_session_runtime() -> None:
    runtime = TransportRuntime(_loopback_config())
    created: list[SessionRuntime] = []

    def factory(session_id: SessionId) -> SessionRuntime:
        session = SessionRuntime.create(
            session_id=session_id,
            turn_id_prefix="turn",
            task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        )
        created.append(session)
        return session

    runtime.set_session_runtime_factory(factory)

    async def run() -> bool:
        return await runtime.receive_onsite_asr_final(
            StreamKey("session-new", STREAM_ID),
            ASRAudienceEvent("请介绍 BitNet", 20, "asr-1", 1),
        )

    assert not asyncio.run(run())
    assert created == []


def test_session_capacity_is_checked_before_factory_construction() -> None:
    runtime = TransportRuntime(replace(_loopback_config(), max_sessions=1))
    constructed: list[str] = []

    def factory(session_id: SessionId) -> SessionRuntime:
        constructed.append(str(session_id))
        return SessionRuntime.create(
            session_id=session_id,
            turn_id_prefix="turn",
            task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        )

    runtime.set_session_runtime_factory(factory)

    first = runtime._runtime_for_session(  # pyright: ignore[reportPrivateUsage]
        "session-1", allow_create=True
    )
    second = runtime._runtime_for_session(  # pyright: ignore[reportPrivateUsage]
        "session-2", allow_create=True
    )

    assert first is not None
    assert second is None
    assert constructed == ["session-1"]


def test_output_fences_are_isolated_per_session() -> None:
    first = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="first",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
    )
    second = SessionRuntime.create(
        session_id=SessionId("session-2"),
        turn_id_prefix="second",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
    )
    hub = RtpHub()
    hub.set_output_fence(first.output_fence, "session-1")
    hub.set_output_fence(second.output_fence, "session-2")
    hub.register_control(
        MicInputRegistration(
            "session-1",
            "mic",
            EnvelopeCorrelation("trace-1", "session-1", 1),
        ),
        "127.0.0.1",
    )

    _ = hub.authorize_onsite_output(
        StreamKey("session-1", "mic"), CancellationEpoch(0)
    )

    assert first.scheduler.snapshot.revision == 1
    assert second.scheduler.snapshot.revision == 0


def test_idle_sweeper_exempts_active_playback_then_erases_session() -> None:
    async def scenario() -> None:
        runtime = TransportRuntime(
            replace(_loopback_config(), session_idle_ttl_seconds=1)
        )
        session = SessionRuntime.create(
            session_id=SessionId("session-1"),
            turn_id_prefix="turn",
            task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        )
        runtime.set_session_runtime(session)
        active = session.output_fence.activate(
            stream=StreamKey("session-1", "mic"),
            segment_id=SegmentId("segment"),
            correlation=EnvelopeCorrelation("trace", "session-1", 1),
        )

        assert await runtime.sweep_sessions(now_ms=10**15) == ()
        lease = session.output_fence
        assert lease.finish(
            stream=active.stream,
            turn_id=active.turn_id,
            segment_id=active.segment_id,
            cancellation_epoch=active.cancellation_epoch,
        )
        assert await runtime.sweep_sessions(now_ms=10**15) == ("session-1",)

    asyncio.run(scenario())


def test_control_connection_refuses_comments_without_valid_credential() -> None:
    # Given: a production-token transport and three valid-looking comments.

    runtime = TransportRuntime(_token_config())

    session_runtime = SessionRuntime.create(
        session_id=SessionId(SESSION_ID),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )

    runtime.set_session_runtime(session_runtime)

    comment = _audience_comment(SESSION_ID, "trace-comment", 7)

    # When: direct, absent, and invalid per-message credentials submit the frame.

    direct = session_runtime.receive_control(comment)

    asyncio.run(runtime.handle_control(_ControlConnection((comment,))))

    asyncio.run(
        runtime.handle_control(_ControlConnection((comment,), authorization=None))
    )

    asyncio.run(
        runtime.handle_control(
            _ControlConnection((comment,), authorization="Bearer wrong")
        )
    )

    # Then: no frame reaches the scheduler or creates an outbound effect.

    observables = session_runtime.observables

    assert direct is False

    assert observables.snapshot.revision == 0

    assert observables.dispatches == ()

    assert observables.task_commits == ()

    assert observables.generated_rtp == ()

    assert observables.sound_transitions == ()


def test_control_connection_rejects_mixed_roles_on_one_authenticated_connection() -> (
    None
):
    # Given: one authenticated control connection carrying only canonical commands.

    runtime = TransportRuntime(_token_config())

    session_runtime = SessionRuntime.create(
        session_id=SessionId(SESSION_ID),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 1),
    )

    runtime.set_session_runtime(session_runtime)

    connection = _ControlConnection(
        (
            _audience_comment(SESSION_ID, "trace-turn", 0),
            _profile_enrollment(SESSION_ID, "trace-profile", 1),
            _profile_revoke(SESSION_ID, "trace-profile", 2),
            _action_command(SESSION_ID, "trace-action", 3),
            _presentation_command(SESSION_ID, "trace-deck", 4),
            _presentation_result(SESSION_ID, "trace-deck", 4),
            "{",
        ),
        authorization="Bearer transport-test-token",
    )

    # When: the transport receives the authenticated lifecycle and command sequence.

    asyncio.run(runtime.handle_control(connection))

    # Then: the first source binds the connection role and other roles are rejected.

    assert session_runtime.interaction_ingress.reducer.presentation_state is None

    stages = [record.stage for record in session_runtime.operational_journal.records]

    assert stages == []

    assert session_runtime.deck_dispatcher.journal == ()

    assert "template-sensitive" not in repr(session_runtime.operational_journal.records)


def test_control_connection_skips_expired_presentation_mcp() -> None:
    # Given: an authenticated root runtime whose task deadline expires before selection.

    clock = _IncrementingClock()

    runtime = TransportRuntime(_token_config())

    session_runtime = SessionRuntime.create(
        session_id=SessionId(SESSION_ID),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 1),
        clock=clock.now,
    )

    runtime.set_session_runtime(session_runtime)

    connection = _ControlConnection(
        (
            _audience_comment(SESSION_ID, "trace-deadline-turn", 0),
            _presentation_command(SESSION_ID, "trace-deadline", 1),
        ),
        authorization="Bearer transport-test-token",
    )

    # When: control handles a presentation whose generated task expires immediately.

    asyncio.run(runtime.handle_control(connection))

    # Then: no adapter issue or presentation state can be committed.

    assert session_runtime.deck_dispatcher.journal == ()

    assert session_runtime.interaction_ingress.reducer.presentation_state is None


def _source_registration() -> str:

    return _envelope(
        "mic.input.register",
        "mic",
        {"stream_id": STREAM_ID},
    )


def _audience_comment(session_id: str, trace_id: str, sequence: int) -> str:

    return json.dumps(
        {
            "event_type": "audience.input",
            "source": "comments",
            "trace_id": trace_id,
            "session_id": session_id,
            "seq": sequence,
            "data": {"text": "解释量化"},
        }
    )


def _profile_enrollment(session_id: str, trace_id: str, sequence: int) -> str:

    return _envelope(
        "profile.enroll.command",
        "orchestrator",
        {
            "profile_id": "profile-transport",
            "preferred_name": "private-name",
            "evidence_id": "voice-evidence-1",
            "consented": True,
        },
        session_id=session_id,
        trace_id=trace_id,
        sequence=sequence,
    )


def _profile_revoke(session_id: str, trace_id: str, sequence: int) -> str:

    return _envelope(
        "profile.revoke.command",
        "orchestrator",
        {"profile_id": "profile-transport"},
        session_id=session_id,
        trace_id=trace_id,
        sequence=sequence,
    )


def _action_command(session_id: str, trace_id: str, sequence: int) -> str:

    return _envelope(
        "action.command",
        "orchestrator",
        {"command_id": "action-transport", "action": "speak"},
        session_id=session_id,
        trace_id=trace_id,
        sequence=sequence,
    )


def _presentation_command(session_id: str, trace_id: str, sequence: int) -> str:

    return _envelope(
        "presentation.load.command",
        "orchestrator",
        {
            "command_id": "deck-transport",
            "deck_id": "deck-transport",
            "deck_version": "v1",
            "page": 1,
        },
        session_id=session_id,
        trace_id=trace_id,
        sequence=sequence,
    )


def _presentation_result(session_id: str, trace_id: str, sequence: int) -> str:

    return _envelope(
        "presentation.result",
        "frontend",
        {"command_id": "deck-transport", "succeeded": True},
        session_id=session_id,
        trace_id=trace_id,
        sequence=sequence,
    )


def _sink_registration() -> str:

    return _envelope(
        "media.rtp.sink.register",
        "sound",
        {"stream_id": STREAM_ID, "codec": _codec(), "rtp_endpoint": _endpoint(5006)},
    )


def _sink_ready() -> str:

    return _envelope(
        "media.rtp.sink.ready",
        "sound",
        {"stream_id": STREAM_ID},
    )


def _stream_state(state: str) -> str:

    return _envelope(
        "media.stream.state",
        "sound",
        {
            "command_id": f"rtp-{STREAM_ID}-0",
            "stream_id": STREAM_ID,
            "state": state,
            "cancellation_epoch": 0,
        },
    )


def test_new_caption_timeline_cancels_the_previous_active_timeline() -> None:
    runtime = TransportRuntime(_loopback_config())
    peer = RecordingControlPeer()
    runtime.register_frontend_connection(SessionId(SESSION_ID), peer)
    first = CaptionTimelineCommand(
        "timeline-1", "第一句", "agent-turn-1", 1, 96_000
    )
    second = CaptionTimelineCommand(
        "timeline-2", "第二句", "agent-turn-2", 2, 96_320
    )

    asyncio.run(
        runtime.emit_caption_timeline(
            first, SessionId(SESSION_ID), AgentTurnId("turn-1")
        )
    )
    asyncio.run(
        runtime.emit_caption_timeline(
            second, SessionId(SESSION_ID), AgentTurnId("turn-2")
        )
    )

    envelopes: list[dict[str, object]] = [
        json.loads(message) for message in peer.messages
    ]
    assert [item["event_type"] for item in envelopes] == [
        "vtuber.caption.timeline.command",
        "vtuber.caption.timeline.cancel",
        "vtuber.caption.timeline.command",
    ]
    cancelled_data = envelopes[1]["data"]
    assert isinstance(cancelled_data, dict)
    assert cancelled_data["timeline_id"] == "timeline-1"
    assert cancelled_data["reason"] == "replaced"


def _envelope(  # noqa: PLR0913
    event_type: str,
    source: str,
    data: dict[str, JsonValue],
    *,
    session_id: str = SESSION_ID,
    trace_id: str = "trace-001",
    sequence: int = 1,
) -> str:

    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": event_type,
            "event_id": f"evt-{event_type}",
            "source": source,
            "time": "2026-07-08T00:00:00Z",
            "trace_id": trace_id,
            "session_id": session_id,
            "seq": sequence,
            "data": data,
        }
    )


def _codec() -> dict[str, JsonValue]:

    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _endpoint(port: int) -> dict[str, JsonValue]:

    return {"host": "declared.example.test", "port": port}


def _rtp_packet(payload_type: int) -> bytes:

    header = bytes((0x80, payload_type, 0, 1, 0, 0, 0, 1, 0x12, 0x34, 0x56, 0x78))

    return header + (b"\x00\x01" * 320)


def _loopback_config() -> TransportConfig:

    return TransportConfig(
        "127.0.0.1",
        8765,
        "127.0.0.1",
        5004,
        "127.0.0.1",
        8765,
        5004,
        "ws",
        None,
        None,
        None,
    )


def _token_config() -> TransportConfig:

    return TransportConfig(
        "127.0.0.1",
        8765,
        "127.0.0.1",
        5004,
        "127.0.0.1",
        8765,
        5004,
        "wss",
        TrustedLanToken("transport-test-token"),
        None,
        None,
    )


def _fake_datagram_listener(transport: FakeDatagramTransport) -> DatagramListener:

    async def listen(_host: str, _port: int, _hub: RtpHub) -> FakeDatagramTransport:

        return transport

    return listen


def _fake_control_listener(server: FakeControlServer) -> ControlListener:

    async def listen(
        _config: TransportConfig, _handler: ControlHandler
    ) -> FakeControlServer:

        return server

    return listen
