from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import pytest

from orchestrator.agent_pipeline import AgentPipeline, BrainStateSnapshot, GateDecision
from orchestrator.config import TrustedLanToken
from orchestrator.ids import SessionId
from orchestrator.json_boundary import parse_json_value
from orchestrator.mcp_adapters import DeckJournalKind
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.streaming_contracts import StreamKey
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_control import (
    AuthenticatedControl,
    EnvelopeCorrelation,
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


@dataclass
class _RecordingOutputFence:
    finishes: list[dict[str, object]] = field(default_factory=list)

    def finish(self, **kwargs: object) -> bool:
        self.finishes.append(kwargs)
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
class _HeldControlConnection:
    messages: tuple[str, ...]

    release: asyncio.Event

    sent: list[str] = field(default_factory=list)

    @property
    def remote_address(self) -> tuple[str, int]:
        return ("127.0.0.1", 443)

    async def __aiter__(self) -> AsyncIterator[str]:
        for message in self.messages:
            yield message
        _ = await self.release.wait()

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def respond(self, status: HTTPStatus, text: str) -> Response:
        _ = status, text
        raise AssertionError


class _AcceptGate:
    def evaluate(self, *args: object, **kwargs: object) -> GateDecision:
        _ = args, kwargs
        return GateDecision.ACCEPT


class _CaptionBrain:
    def plan(self, snapshot: BrainStateSnapshot, **kwargs: object) -> str:
        _ = kwargs
        return json.dumps(
            {
                "response_text": "欢迎来到活动",
                "expected_revision": snapshot.revision,
                "frontend_operations": [
                    {"kind": "caption", "value": "欢迎来到活动"}
                ],
            }
        )

    def repair(self, snapshot: BrainStateSnapshot, invalid_plan: str) -> str:
        _ = snapshot, invalid_plan
        return "{}"


class _NoTools:
    def execute(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs


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


def test_hub_accepts_canonical_source_and_sink_ready_events() -> None:
    # Given: registered Mic and Sound routes for a canonical media stream.

    transport = FakeDatagramTransport()

    hub = RtpHub(transport)

    hub.register_control(_source_registration(), SOURCE_PEER[0])

    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: both peers acknowledge their canonical RTP readiness events.

    hub.register_control(_source_ready(), SOURCE_PEER[0])

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

    # When: a second route claims either existing session-stream route.

    with pytest.raises(DuplicateRouteError):
        hub.register_control(_source_registration(), SOURCE_PEER[0])

    # Then: the duplicate is refused rather than replacing the authenticated route.

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

        assert source.messages == []

        # Source registration is compatibility-only and cannot start media.
        assert sink.messages == []

        await dispatcher.register(_sink_ready(), SINK_PEER[0], sink)

        assert source.messages == []

    asyncio.run(verify_startup_gate())


def test_finished_playback_never_requires_mic_epoch_advance() -> None:
    route = _RouteWithoutEpochAdvance()
    dispatcher = TransportControlDispatch(route)  # type: ignore[arg-type]
    fence = _RecordingOutputFence()
    dispatcher.set_output_fence(fence)  # type: ignore[arg-type]

    asyncio.run(
        dispatcher.register(
            _envelope(
                "media.stream.state",
                "sound",
                {
                    "stream_id": STREAM_ID,
                    "state": "finished",
                    "cancellation_epoch": 0,
                },
            ),
            SINK_PEER[0],
            RecordingControlPeer(),
        )
    )

    assert route.registrations == 1
    assert len(fence.finishes) == 1


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
    hub.set_voice_evidence_callback(runtime.receive_voice_evidence)
    evidence = VoiceEvidence(
        session_id=SESSION_ID,
        stream_id=STREAM_ID,
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


def test_control_connection_routes_comments_through_scheduler_runtime() -> None:
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

    # Then: only valid input dispatches, while every reject leaves effects unchanged.

    observables = session_runtime.observables

    assert len(observables.dispatches) == 1

    assert observables.dispatches[0].correlation.trace_id == "trace-comment"

    assert observables.snapshot.revision == 1

    assert observables.snapshot.active_turn_id == "turn-0001"

    assert len(observables.rejections) == 1

    assert observables.rejections[0].correlation.trace_id == "trace-foreign"

    assert observables.task_commits == ()

    assert observables.generated_rtp == ()

    assert observables.sound_transitions == ()

    assert connection.sent == []


def test_control_connection_dispatches_brain_frontend_effect_to_registered_client() -> (
    None
):
    runtime = TransportRuntime(_loopback_config())
    session_runtime = SessionRuntime.create(
        session_id=SessionId(SESSION_ID),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        agent_pipeline=AgentPipeline(_AcceptGate(), _CaptionBrain(), _NoTools()),
    )
    runtime.set_session_runtime(session_runtime)
    async def run() -> _HeldControlConnection:
        release = asyncio.Event()
        connection = _HeldControlConnection(
            (
                _envelope("frontend.register", "frontend", {}),
                _audience_comment(SESSION_ID, "trace-caption", 1),
            ),
            release,
        )
        task = asyncio.create_task(runtime.handle_control(connection))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        await task
        return connection

    connection = asyncio.run(run())

    assert len(connection.sent) == 1
    envelope = parse_json_value(connection.sent[0])
    assert isinstance(envelope, dict)
    assert envelope["event_type"] == "vtuber.caption.command"
    assert envelope["source"] == "orchestrator"
    assert envelope["session_id"] == SESSION_ID
    data = envelope["data"]
    assert isinstance(data, dict)
    assert data["text"] == "欢迎来到活动"


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

    assert asyncio.run(run())
    dispatch = session_runtime.observables.dispatches[0]
    assert dispatch.correlation.session_id == SessionId(SESSION_ID)


def test_onsite_asr_final_creates_an_isolated_session_runtime() -> None:
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

    assert asyncio.run(run())
    assert len(created) == 1
    assert created[0].scheduler.snapshot.session_id == SessionId("session-new")
    assert created[0].observables.dispatches[0].correlation.session_id == SessionId(
        "session-new"
    )


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


def test_control_connection_routes_authenticated_profile_action_and_presentation() -> (
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

    # Then: typed reducer paths record outcomes and the exact ACK commits deck state.

    assert session_runtime.interaction_ingress.reducer.presentation_state == (
        "deck-transport",
        "v1",
        1,
    )

    stages = [record.stage for record in session_runtime.operational_journal.records]

    assert stages == [
        "profile_enrolled",
        "profile_revoked",
        "action",
        "presentation_command",
        "task_scheduled",
        "deck_task",
        "deck_dispatch",
        "task_result",
        "presentation_ack",
    ]

    assert [entry.kind for entry in session_runtime.deck_dispatcher.journal] == [
        DeckJournalKind.DISPATCHED,
        DeckJournalKind.SUCCEEDED,
    ]

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
        "media.rtp.source.register",
        "mic",
        {
            "stream_id": STREAM_ID,
            "ssrc": SSRC,
            "codec": _codec(),
            "rtp_endpoint": _endpoint(5004),
        },
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
            "encrypted_template": "template-sensitive",
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


def _source_ready() -> str:

    return _envelope(
        "media.rtp.source.ready",
        "mic",
        {"stream_id": STREAM_ID, "ssrc": SSRC},
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
        {"stream_id": STREAM_ID, "state": state},
    )


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
