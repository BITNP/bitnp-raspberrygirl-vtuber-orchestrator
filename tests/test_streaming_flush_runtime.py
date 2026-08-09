
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.config import load_fake_config
from orchestrator.ids import SessionId
from orchestrator.json_boundary import parse_json_value
from orchestrator.observability import OnsiteObservability
from orchestrator.scheduler_reflex import OutputLease, SchedulerOutputFence
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import SessionScheduler
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    FlushRequestId,
    GeneratedSsrc,
    SegmentId,
    StreamFlush,
    StreamKey,
    TurnId,
)
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind, TaskState
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_control import EnvelopeCorrelation
from orchestrator.transport_runtime import TransportRuntime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from http import HTTPStatus

    from websockets.http11 import Response

    from orchestrator.json_boundary import JsonValue
    from orchestrator.transport_hub import RtpHub
    from orchestrator.transport_runtime import ControlHandler


@dataclass
class _Clock:

    now_ms: int = 0

    def advance(self, milliseconds: int) -> None:

        self.now_ms += milliseconds


@dataclass
class _DatagramTransport:

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:

        _ = data

        _ = addr

    def close(self) -> None:

        return


@dataclass
class _ControlServer:

    def close(self) -> None:

        return

    async def wait_closed(self) -> None:

        return


@dataclass
class _WssConnection:

    peer_ip: str

    incoming: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)

    sent: list[str] = field(default_factory=list)

    @property
    def remote_address(self) -> tuple[str, int]:

        return (self.peer_ip, 443)

    def __aiter__(self) -> AsyncIterator[str]:

        return self

    async def __anext__(self) -> str:

        message = await self.incoming.get()

        if message is None:
            raise StopAsyncIteration

        return message

    async def send(self, message: str) -> None:

        self.sent.append(message)

    def respond(self, status: HTTPStatus, text: str) -> Response:

        _ = status

        _ = text

        raise AssertionError


def test_runtime_admits_replacement_only_after_matching_sound_ack() -> None:

    asyncio.run(_matching_ack_proof())


def test_runtime_rejects_invalid_ack_and_missing_ack_timeout() -> None:

    asyncio.run(_rejection_proof())


def test_runtime_holds_replacement_until_sound_acknowledges_flush() -> None:
    asyncio.run(_delayed_replacement_proof())


def test_runtime_registers_session_owned_flush_before_admitting_replacement() -> None:
    asyncio.run(_session_owned_replacement_proof())


def test_acknowledged_flush_can_restore_old_lease_before_task_commit() -> None:
    scheduler = SessionScheduler(
        session_id=SessionId("session-001"), turn_id_prefix="turn-reflex"
    )
    fence = SchedulerOutputFence(scheduler)
    stream = StreamKey(session_id="session-001", stream_id="stream-001")
    correlation = EnvelopeCorrelation("trace-source-001", "session-001", 29)
    active = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-active"),
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
        correlation=correlation,
    )
    replacement, flush = fence.interrupt(
        stream=stream,
        segment_id=SegmentId("segment-new"),
        correlation=correlation,
    )

    assert fence.acknowledge(FlushAcknowledgement.from_flush(flush))
    assert fence.can_emit(stream, replacement.cancellation_epoch)

    assert fence.abandon_replacement(stream)
    assert fence.can_emit(stream, active.cancellation_epoch)
    assert not fence.can_emit(stream, replacement.cancellation_epoch)


async def _matching_ack_proof() -> None:
    # Given: live Mic and Sound WSS sessions registered on one runtime-owned stream.


    clock = _Clock()

    observability = OnsiteObservability(load_fake_config())

    runtime = TransportRuntime(
        _config(),
        datagram_listener=_datagram_listener,
        control_listener=_control_listener,
        clock=clock,
    )

    runtime.set_observability(observability)

    source, sink, tasks = await _registered_runtime(runtime)

    fence = SchedulerOutputFence(
        SessionScheduler(
            session_id=SessionId("session-001"), turn_id_prefix="turn-reflex"
        )
    )

    runtime.set_output_fence(fence)

    stream = StreamKey(session_id="session-001", stream_id="stream-001")

    correlation = EnvelopeCorrelation("trace-source-001", "session-001", 29)

    _ = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-active"),
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
        correlation=correlation,
    )

    replacement, flush = fence.interrupt(
        stream=stream,
        segment_id=SegmentId("segment-replacement"),
        correlation=correlation,
    )

    # When: runtime sends a flush and Sound returns the exact acknowledgement over WSS.

    gate_started_at = clock.now_ms

    await runtime.request_stream_flush(flush)

    assert fence.can_emit(stream, replacement.cancellation_epoch) is False

    await sink.incoming.put(_acknowledgement(flush, session_id="stale-session"))

    await asyncio.sleep(0)

    assert fence.can_emit(stream, replacement.cancellation_epoch) is False

    await sink.incoming.put(_acknowledgement(flush))

    await asyncio.sleep(0)

    admission = asyncio.create_task(runtime.admit_replacement(replacement, flush))
    await _acknowledge_replacement_command(sink)
    admitted = await admission

    mismatched = await runtime.admit_replacement(
        replacement,
        StreamFlush(
            stream=flush.stream,
            turn_id=flush.turn_id,
            segment_id=SegmentId("segment-other"),
            cancellation_epoch=flush.cancellation_epoch,
            request_id=FlushRequestId("flush-other"),
            target_generated_ssrc=flush.target_generated_ssrc,
        )
    )

    # Then: only the matching ack permits a second generated stream command.

    assert clock.now_ms - gate_started_at <= 20

    assert _event_types(sink.sent).count("media.stream.flush") == 1

    assert _event_types(source.sent).count("media.stream.flush") == 0

    assert admitted is True

    assert mismatched is False

    assert fence.can_emit(stream, replacement.cancellation_epoch) is True

    assert _event_types(sink.sent).count("media.stream.command") == 2

    flush_envelope = next(
        _envelope_value(message)
        for message in sink.sent
        if _envelope_value(message)["event_type"] == "media.stream.flush"
    )

    replacement_envelope = _envelope_value(sink.sent[-1])

    replacement_data = replacement_envelope["data"]

    assert isinstance(replacement_data, dict)

    assert _correlation(flush_envelope) == ("trace-source-001", "session-001", 29)

    assert _correlation(replacement_envelope) == (
        "trace-source-001",
        "session-001",
        29,
    )

    assert (
        replacement_envelope["turn_id"],
        replacement_envelope["segment_id"],
        replacement_data["cancellation_epoch"],
    ) == (
        str(replacement.turn_id),
        str(replacement.segment_id),
        int(replacement.cancellation_epoch),
    )

    assert observability.records[-1].stage == "flush_ack"

    assert (
        observability.records[-1].trace_id,
        observability.records[-1].session_id,
        observability.records[-1].seq,
        observability.records[-1].turn_id,
        observability.records[-1].segment_id,
        observability.records[-1].cancellation_epoch,
    ) == (
        "trace-source-001",
        "session-001",
        29,
        str(flush.turn_id),
        str(flush.segment_id),
        int(flush.cancellation_epoch),
    )

    await sink.incoming.put(_finished_state(replacement))
    await asyncio.sleep(0)

    assert fence.has_active_lease(stream) is False

    await _close_runtime(runtime, source, sink, tasks)


async def _rejection_proof() -> None:
    # Given: a registered Sound control peer and a pending generated-media flush.


    clock = _Clock()

    runtime = TransportRuntime(
        _config(),
        datagram_listener=_datagram_listener,
        control_listener=_control_listener,
        clock=clock,
    )

    source, sink, tasks = await _registered_runtime(runtime)

    flush = _flush()

    # When: Sound sends a stale-session acknowledgement, then time reaches 750ms.

    await runtime.request_stream_flush(flush)

    await sink.incoming.put(_acknowledgement(flush, session_id="stale-session"))

    await asyncio.sleep(0)

    clock.advance(250)

    await runtime.advance_flush_admission()

    clock.advance(500)

    await runtime.advance_flush_admission()

    # Then: retry occurs once and invalid/missing acknowledgement blocks replacement.

    assert _event_types(sink.sent).count("media.stream.flush") == 2

    assert await runtime.admit_replacement(_replacement(flush), flush) is False

    assert [failure.reason for failure in runtime.flush_failures] == ["timeout"]

    await _close_runtime(runtime, source, sink, tasks)


async def _delayed_replacement_proof() -> None:
    clock = _Clock()
    runtime = TransportRuntime(
        _config(),
        datagram_listener=_datagram_listener,
        control_listener=_control_listener,
        clock=clock,
    )
    source, sink, tasks = await _registered_runtime(runtime)
    fence = SchedulerOutputFence(
        SessionScheduler(
            session_id=SessionId("session-001"), turn_id_prefix="turn-reflex"
        )
    )
    runtime.set_output_fence(fence)
    stream = StreamKey(session_id="session-001", stream_id="stream-001")
    correlation = EnvelopeCorrelation("trace-source-001", "session-001", 29)
    active = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-active"),
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
        correlation=correlation,
    )

    # A replacement first frame is ready.  The async request must leave old
    # audio eligible until the exact Sound acknowledgement arrives.
    prepared = asyncio.create_task(
        runtime.begin_onsite_replacement(stream, SegmentId("segment-new"))
    )
    await asyncio.sleep(0)
    assert fence.can_emit(stream, active.cancellation_epoch) is True
    assert not prepared.done()
    flush = next(
        _flush_from_message(message)
        for message in sink.sent
        if _envelope_value(message)["event_type"] == "media.stream.flush"
    )

    await sink.incoming.put(_acknowledgement(flush))
    await _acknowledge_replacement_command(sink)
    replacement_epoch = await prepared

    assert replacement_epoch == flush.cancellation_epoch
    assert fence.can_emit(stream, active.cancellation_epoch) is False
    assert fence.can_emit(stream, flush.cancellation_epoch) is True
    assert _event_types(sink.sent).count("media.stream.command") == 2

    await _close_runtime(runtime, source, sink, tasks)


async def _session_owned_replacement_proof() -> None:
    runtime = TransportRuntime(
        _config(),
        datagram_listener=_datagram_listener,
        control_listener=_control_listener,
    )
    session_runtime = SessionRuntime.create(
        session_id=SessionId("session-001"),
        turn_id_prefix="turn-reflex",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 4),
    )
    runtime.set_session_runtime(session_runtime)
    source, sink, tasks = await _registered_runtime(runtime)
    fence = session_runtime.output_fence
    stream = StreamKey(session_id="session-001", stream_id="stream-001")
    correlation = EnvelopeCorrelation("trace-source-001", "session-001", 29)
    active = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-active"),
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
        correlation=correlation,
    )

    prepared = asyncio.create_task(
        runtime.begin_onsite_replacement(stream, SegmentId("segment-new"))
    )
    await asyncio.sleep(0)
    flush_task = next(
        record
        for record in session_runtime.task_registry.records
        if str(record.request.task_id).startswith("sound-flush-")
    )
    assert flush_task.state is TaskState.RUNNING
    assert fence.can_emit(stream, active.cancellation_epoch)

    flush = next(
        _flush_from_message(message)
        for message in sink.sent
        if _envelope_value(message)["event_type"] == "media.stream.flush"
    )
    await sink.incoming.put(_acknowledgement(flush))
    await _acknowledge_replacement_command(sink)

    assert await prepared == flush.cancellation_epoch
    completed = session_runtime.task_registry.task(flush_task.request.task_id)
    assert completed is not None
    assert completed.state is TaskState.COMPLETED
    assert session_runtime.observables.task_commits[-1].effect.effect_type == (
        "sound.flush.admitted"
    )

    await _close_runtime(runtime, source, sink, tasks)


async def _registered_runtime(
    runtime: TransportRuntime,
) -> tuple[
    _WssConnection, _WssConnection, tuple[asyncio.Task[None], asyncio.Task[None]]
]:

    await runtime.start()

    source = _WssConnection(peer_ip="192.0.2.10")

    sink = _WssConnection(peer_ip="192.0.2.11")

    source_task = asyncio.create_task(runtime.handle_control(source))

    sink_task = asyncio.create_task(runtime.handle_control(sink))

    await source.incoming.put(_source_registration())

    await sink.incoming.put(_sink_registration())

    await asyncio.sleep(0)

    return source, sink, (source_task, sink_task)


async def _close_runtime(
    runtime: TransportRuntime,
    source: _WssConnection,
    sink: _WssConnection,
    tasks: tuple[asyncio.Task[None], asyncio.Task[None]],
) -> None:

    await source.incoming.put(None)

    await sink.incoming.put(None)

    await tasks[0]

    await tasks[1]

    await runtime.close()


async def _acknowledge_replacement_command(sink: _WssConnection) -> None:
    async with asyncio.timeout(1.0):
        # The ready acknowledgement must follow the replacement command; an
        # unmatched early ready is deliberately rejected by the transport.
        while _event_types(sink.sent).count("media.stream.command") < 2:  # noqa: ASYNC110
            await asyncio.sleep(0)
    await sink.incoming.put(_sink_ready())


def _flush() -> StreamFlush:

    return StreamFlush(
        stream=StreamKey(session_id="session-001", stream_id="stream-001"),
        turn_id=TurnId("turn-001"),
        segment_id=SegmentId("segment-001"),
        cancellation_epoch=CancellationEpoch(3),
        request_id=FlushRequestId("flush-request-001"),
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
    )


def _replacement(flush: StreamFlush) -> OutputLease:
    return OutputLease(
        stream=flush.stream,
        turn_id=TurnId("turn-replacement"),
        segment_id=SegmentId("segment-replacement"),
        cancellation_epoch=flush.cancellation_epoch,
        generation=int(flush.cancellation_epoch),
        target_generated_ssrc=GeneratedSsrc(0x8765_4321),
    )


def _source_registration() -> str:

    return _envelope(
        _EnvelopeFields(
            event_type="mic.input.register",
            source="mic",
            data={"stream_id": "stream-001"},
            trace_id="trace-source-001",
            seq=29,
        )
    )


def _sink_registration() -> str:

    return _envelope(
        _EnvelopeFields(
            event_type="media.rtp.sink.register",
            source="sound",
            data={
                "stream_id": "stream-001",
                "codec": _codec(),
                "rtp_endpoint": {"host": "192.0.2.11", "port": 5006},
            },
        )
    )


def _sink_ready() -> str:
    return _envelope(
        _EnvelopeFields(
            event_type="media.rtp.sink.ready",
            source="sound",
            data={"stream_id": "stream-001"},
        )
    )


def _acknowledgement(flush: StreamFlush, *, session_id: str = "session-001") -> str:

    return _envelope(
        _EnvelopeFields(
            event_type="media.stream.flush.ack",
            source="sound",
            session_id=session_id,
            turn_id=str(flush.turn_id),
            segment_id=str(flush.segment_id),
            data={
                "stream_id": flush.stream.stream_id,
                "cancellation_epoch": int(flush.cancellation_epoch),
                "request_id": str(flush.request_id),
                "target_generated_ssrc": int(flush.target_generated_ssrc),
                "disposition": "APPLIED",
            },
            trace_id="trace-source-001",
            seq=29,
        )
    )


def _finished_state(replacement: OutputLease) -> str:
    return _envelope(
        _EnvelopeFields(
            event_type="media.stream.state",
            source="sound",
            turn_id=str(replacement.turn_id),
            segment_id=str(replacement.segment_id),
            data={
                "command_id": (
                    f"rtp-{replacement.stream.stream_id}-"
                    f"{int(replacement.cancellation_epoch)}"
                ),
                "stream_id": replacement.stream.stream_id,
                "state": "finished",
                "cancellation_epoch": int(replacement.cancellation_epoch),
            },
            trace_id="trace-source-001",
            seq=29,
        )
    )


def _flush_from_message(message: str) -> StreamFlush:
    envelope = _envelope_value(message)
    data = envelope["data"]
    assert isinstance(data, dict)
    stream_id = data["stream_id"]
    epoch = data["cancellation_epoch"]
    request_id = data["request_id"]
    target_ssrc = data["target_generated_ssrc"]
    turn_id = envelope["turn_id"]
    segment_id = envelope["segment_id"]
    assert isinstance(stream_id, str)
    assert isinstance(epoch, int)
    assert isinstance(request_id, str)
    assert isinstance(target_ssrc, int)
    assert isinstance(turn_id, str)
    assert isinstance(segment_id, str)
    return StreamFlush(
        stream=StreamKey("session-001", stream_id),
        turn_id=TurnId(turn_id),
        segment_id=SegmentId(segment_id),
        cancellation_epoch=CancellationEpoch(epoch),
        request_id=FlushRequestId(request_id),
        target_generated_ssrc=GeneratedSsrc(target_ssrc),
        correlation=EnvelopeCorrelation("trace-source-001", "session-001", 29),
    )


@dataclass(frozen=True)
class _EnvelopeFields:

    data: dict[str, JsonValue]

    event_type: str

    source: str

    session_id: str = "session-001"

    turn_id: str | None = None

    segment_id: str | None = None

    trace_id: str = "trace-001"

    seq: int = 1


def _envelope(fields: _EnvelopeFields) -> str:

    envelope: dict[str, JsonValue] = {
        "schema_version": "1.0.0",
        "event_type": fields.event_type,
        "event_id": fields.event_type,
        "source": fields.source,
        "time": "2026-07-28T00:00:00Z",
        "trace_id": fields.trace_id,
        "session_id": fields.session_id,
        "seq": fields.seq,
        "data": fields.data,
    }

    if fields.turn_id is not None:
        envelope["turn_id"] = fields.turn_id

    if fields.segment_id is not None:
        envelope["segment_id"] = fields.segment_id

    return json.dumps(envelope)


def _codec() -> dict[str, JsonValue]:

    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _event_types(messages: list[str]) -> list[str]:

    return [json.loads(message)["event_type"] for message in messages]


def _correlation(envelope: dict[str, JsonValue]) -> tuple[str, str, int]:

    trace_id = envelope["trace_id"]

    session_id = envelope["session_id"]

    seq = envelope["seq"]

    assert isinstance(trace_id, str)

    assert isinstance(session_id, str)

    assert isinstance(seq, int)

    return trace_id, session_id, seq


def _envelope_value(message: str) -> dict[str, JsonValue]:

    value = parse_json_value(message)

    assert isinstance(value, dict)

    return value


def _config() -> TransportConfig:

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


async def _datagram_listener(
    _host: str, _port: int, _hub: RtpHub
) -> _DatagramTransport:

    return _DatagramTransport()


async def _control_listener(
    _config: TransportConfig, _handler: ControlHandler
) -> _ControlServer:

    return _ControlServer()
