from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, final

from orchestrator.ids import ConnectionId
from orchestrator.observability import (
    OnsiteObservability,
    OnsiteStage,
    StageCorrelation,
)
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    FlushAdmission,
    FlushClock,
    FlushFailure,
    SegmentId,
    StreamFlush,
    StreamKey,
    TurnId,
)
from orchestrator.transport_control import (
    ControlEvent,
    EnvelopeCorrelation,
    MicInputRegistration,
    SinkRegistration,
    StreamReady,
    StreamState,
    VoiceEvidence,
    parse_control_event,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_CODEC = {
    "format": "L16",
    "clock_rate_hz": 16_000,
    "channels": 1,
    "payload_type": 96,
    "samples_per_frame": 320,
}
_LOGGER = logging.getLogger(__name__)


class ControlPeer(Protocol):
    async def send(self, message: str) -> None: ...


class RouteRegistry(Protocol):
    def register_control(
        self,
        raw_message: ControlEvent | str,
        peer_ip: str,
        owner: ConnectionId | None = None,
    ) -> None: ...

    def remove_connection(self, owner: ConnectionId) -> None: ...

    def remove_stream(self, session_id: str, stream_id: str) -> None: ...

    def output_ssrc(self, stream: StreamKey, cancellation_epoch: int = 0) -> int: ...

    def correlation(self, stream: StreamKey) -> EnvelopeCorrelation | None: ...

    def input_epoch(self, stream: StreamKey) -> int: ...

    def owns_mic_input(self, stream: StreamKey, owner: ConnectionId) -> bool: ...


class OutputFence(Protocol):
    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> bool: ...

    def finish(
        self,
        *,
        stream: StreamKey,
        turn_id: TurnId | None,
        segment_id: SegmentId | None,
        cancellation_epoch: CancellationEpoch | None,
    ) -> bool: ...

@dataclass(frozen=True, slots=True)
class _SinkPeer:
    connection: ControlPeer

    host: str

    udp_port: int


@final
class TransportControlDispatch:
    def __init__(
        self,
        hub: RouteRegistry,
        *,
        clock: FlushClock | None = None,
        observability: OnsiteObservability | None = None,
        rtp_sender_endpoint: tuple[str, int] = ("127.0.0.1", 5004),
    ) -> None:
        self._hub: RouteRegistry = hub

        self._observability = observability
        self._rtp_sender_endpoint = rtp_sender_endpoint

        self._sinks: dict[StreamKey, _SinkPeer] = {}

        self._dispatched: set[StreamKey] = set()

        self._ready_sinks: set[StreamKey] = set()

        self._leases: dict[StreamKey, tuple[str, int]] = {}

        self._flush_outbox: list[StreamFlush] = []

        self._flush_admission = FlushAdmission(
            clock=_MonotonicFlushClock() if clock is None else clock,
            sender=self,
        )

        self._output_fence: OutputFence | None = None

        self._output_fences: dict[str, OutputFence] = {}

        # Called only after OutputFence accepted an exact Sound finished event.
        # The callback is observational state reduction, never transport I/O.
        self._playback_finished_callback: Callable[[StreamKey], None] | None = None

    def set_playback_finished_callback(
        self, callback: Callable[[StreamKey], None] | None
    ) -> None:
        """Receive only physically verified playback completion events."""
        self._playback_finished_callback = callback

    async def register(  # noqa: C901, PLR0911, PLR0912
        self, raw_message: str, peer_ip: str, connection: ControlPeer
    ) -> None:
        event = parse_control_event(raw_message)

        _LOGGER.debug(
            "control_register event=%s peer=%s", type(event).__name__, peer_ip
        )

        owner = _connection_id(connection)
        if isinstance(event, VoiceEvidence):
            evidence_stream = StreamKey(event.session_id, event.stream_id)
            if (
                not self._hub.owns_mic_input(evidence_stream, owner)
                or event.input_epoch != self._hub.input_epoch(evidence_stream)
            ):
                return
        if isinstance(event, StreamState):
            stream = StreamKey(event.session_id, event.stream_id)
            if (
                event.cancellation_epoch is None
                or self._leases.get(stream)
                != (event.command_id, int(event.cancellation_epoch))
            ):
                return

        self._hub.register_control(event, peer_ip, owner)

        if isinstance(event, StreamState):
            self._record_playback(event)

        match event:
            case MicInputRegistration(session_id=session_id, stream_id=stream_id):
                stream = StreamKey(session_id, stream_id)
                await connection.send(
                    _mic_input_ready_envelope(
                        stream, self._hub.input_epoch(stream), event.correlation
                    )
                )
                await self._dispatch_start(StreamKey(session_id, stream_id))
                return

            case SinkRegistration(
                session_id=session_id, stream_id=stream_id, udp_port=port
            ):
                self._sinks[StreamKey(session_id, stream_id)] = _SinkPeer(
                    connection, peer_ip, port
                )

            case StreamReady(session_id=session_id, stream_id=stream_id):
                stream = StreamKey(session_id, stream_id)

                sink = self._sinks.get(stream)

                if sink is None or sink.connection is not connection:
                    return

                self._ready_sinks.add(stream)
                return

            case StreamState(
                session_id=session_id,
                stream_id=stream_id,
                state="cancelled" | "error",
            ):
                sink = self._sinks.get(StreamKey(session_id, stream_id))
                if sink is None or sink.connection is not connection:
                    return
                self._discard(StreamKey(session_id, stream_id))

            case StreamState(
                session_id=session_id,
                stream_id=stream_id,
                state="finished",
                turn_id=turn_id,
                segment_id=segment_id,
                cancellation_epoch=cancellation_epoch,
            ):
                sink = self._sinks.get(StreamKey(session_id, stream_id))
                if sink is None or sink.connection is not connection:
                    return
                self._finish_playback(
                    stream=StreamKey(session_id, stream_id),
                    turn_id=turn_id,
                    segment_id=segment_id,
                    cancellation_epoch=cancellation_epoch,
                )
                return

            case StreamState():
                return

            case FlushAcknowledgement() as acknowledgement:
                sink = self._sinks.get(acknowledgement.stream)
                if sink is None or sink.connection is not connection:
                    return
                self._record_flush_acknowledgement(acknowledgement)

                admitted = self._flush_admission.acknowledge(acknowledgement)

                output_fence = self._fence_for(acknowledgement.stream)

                if admitted and output_fence is not None:
                    _ = output_fence.acknowledge(acknowledgement)

                return

            case _:
                return

        await self._dispatch_start(StreamKey(event.session_id, event.stream_id))

    def _finish_playback(
        self,
        *,
        stream: StreamKey,
        turn_id: TurnId | None,
        segment_id: SegmentId | None,
        cancellation_epoch: CancellationEpoch | None,
    ) -> None:
        output_fence = self._fence_for(stream)
        if output_fence is None:
            return
        finished = output_fence.finish(
            stream=stream,
            turn_id=turn_id,
            segment_id=segment_id,
            cancellation_epoch=cancellation_epoch,
        )
        if finished:
            callback = self._playback_finished_callback
            if callback is not None:
                callback(stream)

        # A normal playback finish releases only its output lease.  Mic has no
        # epoch-update control message, so advancing the route generation here
        # would reject every later ASR final carrying Mic's active input epoch.

    async def request_stream_flush(self, flush: StreamFlush) -> None:
        correlation = flush.correlation or self._hub.correlation(flush.stream)

        if correlation is None:
            return

        flush = replace(flush, correlation=correlation)

        self._record_flush(flush)

        self._flush_admission.begin(flush)

        await self._deliver_flushes()

    async def advance_flush_admission(self) -> None:
        self._flush_admission.advance()

        await self._deliver_flushes()

    async def admit_replacement(self, flush: StreamFlush) -> bool:
        if not self._flush_admission.admitted(flush):
            return False

        sink = self._sinks.get(flush.stream)

        if sink is None:
            return False

        await sink.connection.send(
            _stream_command_envelope(
                flush.stream,
                self._hub.output_ssrc(flush.stream, int(flush.cancellation_epoch)),
                self._rtp_sender_endpoint,
                self._hub.correlation(flush.stream),
                flush,
            )
        )
        self._leases[flush.stream] = (
            _media_command_id(flush.stream, int(flush.cancellation_epoch)),
            int(flush.cancellation_epoch),
        )
        _LOGGER.debug(
            "control_sent event=media.stream.command session=%s stream=%s epoch=%d",
            flush.stream.session_id,
            flush.stream.stream_id,
            int(flush.cancellation_epoch),
        )

        return True

    async def finish_generated_stream(self, stream: StreamKey, epoch: int) -> None:
        sink = self._sinks.get(stream)
        correlation = self._hub.correlation(stream)
        if sink is None or correlation is None:
            return
        await sink.connection.send(
            _stream_end_envelope(
                stream,
                epoch,
                self._hub.output_ssrc(stream, epoch),
                correlation,
            )
        )
        self._leases[stream] = (_media_command_id(stream, epoch), epoch)
        _LOGGER.debug(
            "control_sent event=media.stream.end session=%s stream=%s epoch=%d",
            stream.session_id,
            stream.stream_id,
            epoch,
        )

    async def announce_output(self, stream: StreamKey, epoch: int) -> None:
        sink = self._sinks.get(stream)
        correlation = self._hub.correlation(stream)
        if sink is None or correlation is None:
            return
        await sink.connection.send(
            _stream_command_envelope(
                stream,
                self._hub.output_ssrc(stream, epoch),
                self._rtp_sender_endpoint,
                correlation,
                epoch=epoch,
            )
        )
        self._leases[stream] = (_media_command_id(stream, epoch), epoch)
        _LOGGER.debug(
            "control_sent event=media.stream.command session=%s stream=%s epoch=%d",
            stream.session_id,
            stream.stream_id,
            epoch,
        )

    @property
    def flush_failures(self) -> tuple[FlushFailure, ...]:
        return tuple(self._flush_admission.failures)

    def send_flush(self, flush: StreamFlush) -> None:
        self._flush_outbox.append(flush)

    async def cancel_stream(self, session_id: str, stream_id: str) -> None:
        stream = StreamKey(session_id, stream_id)

        self._record_transport_transition("cancellation", stream)

        sink = self._sinks.get(stream)

        correlation = self._hub.correlation(stream)

        self._hub.remove_stream(session_id, stream_id)

        self._discard(stream)

        if sink is not None and correlation is not None:
            await sink.connection.send(_cancel_envelope(stream_id, correlation))
            _LOGGER.debug(
                "control_sent event=media.stream.cancel session=%s stream=%s",
                session_id,
                stream_id,
            )

    def clear(self) -> None:
        self._sinks.clear()

        self._dispatched.clear()

        self._ready_sinks.clear()

        self._leases.clear()

        self._flush_outbox.clear()

    def set_observability(self, observability: OnsiteObservability) -> None:
        self._observability = observability

    def set_output_fence(
        self, output_fence: OutputFence, session_id: str | None = None
    ) -> None:
        if session_id is None:
            self._output_fence = output_fence
        else:
            self._output_fences[session_id] = output_fence

    def _fence_for(self, stream: StreamKey) -> OutputFence | None:
        return self._output_fences.get(stream.session_id, self._output_fence)

    def remove_connection(self, connection: ControlPeer) -> None:
        self._hub.remove_connection(_connection_id(connection))

        for stream, sink in tuple(self._sinks.items()):
            if sink.connection is connection:
                del self._sinks[stream]

                self._dispatched.discard(stream)

                self._ready_sinks.discard(stream)

                _ = self._leases.pop(stream, None)

    def remove_session(self, session_id: str) -> None:
        _ = self._output_fences.pop(session_id, None)
        for stream in tuple(self._sinks):
            if stream.session_id == session_id:
                self._discard(stream)
        self._flush_outbox[:] = [
            flush
            for flush in self._flush_outbox
            if flush.stream.session_id != session_id
        ]

    async def _dispatch_start(self, stream: StreamKey) -> None:
        sink = self._sinks.get(stream)

        if sink is None or stream in self._dispatched:
            return

        self._dispatched.add(stream)

        correlation = self._hub.correlation(stream)

        if correlation is None:
            return

        await sink.connection.send(
            _stream_command_envelope(
                stream,
                self._hub.output_ssrc(stream),
                self._rtp_sender_endpoint,
                correlation,
            )
        )
        self._leases[stream] = (_media_command_id(stream, 0), 0)

    async def _deliver_flushes(self) -> None:
        while self._flush_outbox:
            flush = self._flush_outbox.pop(0)

            sink = self._sinks.get(flush.stream)

            correlation = self._hub.correlation(flush.stream)

            if correlation is not None:
                envelope = _flush_envelope(flush, correlation)

                if sink is not None:
                    await sink.connection.send(envelope)

    def _discard(self, stream: StreamKey) -> None:
        _ = self._sinks.pop(stream, None)

        self._dispatched.discard(stream)

        self._ready_sinks.discard(stream)

        _ = self._leases.pop(stream, None)

    def _record_playback(self, event: StreamState) -> None:
        observability = self._observability

        if observability is not None:
            stream = StreamKey(event.session_id, event.stream_id)

            observability.record_stream(
                "playback_state",
                stream,
                command=StageCorrelation(
                    trace_id=event.correlation.trace_id,
                    session_id=event.correlation.session_id,
                    seq=event.correlation.seq,
                    turn_id=str(event.turn_id) if event.turn_id is not None else None,
                    segment_id=(
                        str(event.segment_id) if event.segment_id is not None else None
                    ),
                    cancellation_epoch=(
                        int(event.cancellation_epoch)
                        if event.cancellation_epoch is not None
                        else None
                    ),
                ),
            )

    def _record_flush(self, flush: StreamFlush) -> None:
        observability = self._observability

        if observability is not None:
            correlation = flush.correlation or self._hub.correlation(flush.stream)

            if correlation is not None:
                observability.record_stream(
                    "flush",
                    flush.stream,
                    command=StageCorrelation(
                        trace_id=correlation.trace_id,
                        session_id=correlation.session_id,
                        seq=correlation.seq,
                        turn_id=str(flush.turn_id),
                        segment_id=str(flush.segment_id),
                        cancellation_epoch=int(flush.cancellation_epoch),
                    ),
                )

    def _record_flush_acknowledgement(
        self, acknowledgement: FlushAcknowledgement
    ) -> None:
        observability = self._observability

        correlation = acknowledgement.correlation

        if observability is not None and correlation is not None:
            observability.record_stream(
                "flush_ack",
                acknowledgement.stream,
                command=StageCorrelation(
                    trace_id=correlation.trace_id,
                    session_id=correlation.session_id,
                    seq=correlation.seq,
                    turn_id=str(acknowledgement.turn_id),
                    segment_id=str(acknowledgement.segment_id),
                    cancellation_epoch=int(acknowledgement.cancellation_epoch),
                ),
            )

    def _record_transport_transition(
        self, stage: OnsiteStage, stream: StreamKey
    ) -> None:
        observability = self._observability

        if observability is not None:
            observability.record_stream(stage, stream)


def _stream_command_envelope(  # noqa: PLR0913
    stream: StreamKey,
    ssrc: int,
    rtp_sender_endpoint: tuple[str, int],
    correlation: EnvelopeCorrelation | None,
    flush: StreamFlush | None = None,
    epoch: int | None = None,
) -> str:
    if correlation is None:
        message = "stream correlation is required"

        raise RuntimeError(message)

    cancellation_epoch = (
        int(flush.cancellation_epoch)
        if flush is not None
        else (0 if epoch is None else epoch)
    )
    data: dict[str, object] = {
        "command_id": _media_command_id(stream, cancellation_epoch),
        "stream_id": stream.stream_id,
        "start_rtp_timestamp": 96_000,
        "ssrc": ssrc,
        "codec": _CODEC,
        "cancellation_epoch": cancellation_epoch,
        "rtp_sender_endpoint": {
            "host": rtp_sender_endpoint[0],
            "port": rtp_sender_endpoint[1],
        },
    }

    return _envelope(
        event_type="media.stream.command",
        correlation=correlation,
        turn_id=str(flush.turn_id) if flush is not None else None,
        segment_id=str(flush.segment_id) if flush is not None else None,
        data=data,
    )


def _cancel_envelope(stream_id: str, correlation: EnvelopeCorrelation) -> str:
    return _envelope(
        event_type="cancel",
        correlation=correlation,
        segment_id=stream_id,
        data={"reason": "transport_cancelled"},
    )


def _mic_input_ready_envelope(
    stream: StreamKey, input_epoch: int, correlation: EnvelopeCorrelation
) -> str:
    return _envelope(
        event_type="mic.input.ready",
        correlation=correlation,
        data={"stream_id": stream.stream_id, "input_epoch": input_epoch},
    )


def _stream_end_envelope(
    stream: StreamKey, epoch: int, ssrc: int, correlation: EnvelopeCorrelation
) -> str:
    return _envelope(
        event_type="media.stream.end",
        correlation=correlation,
        data={
            "command_id": _media_command_id(stream, epoch),
            "stream_id": stream.stream_id,
            "cancellation_epoch": epoch,
            "ssrc": ssrc,
        },
    )


def _flush_envelope(flush: StreamFlush, correlation: EnvelopeCorrelation) -> str:
    return _envelope(
        event_type="media.stream.flush",
        correlation=correlation,
        turn_id=str(flush.turn_id),
        segment_id=str(flush.segment_id),
        data={
            "stream_id": flush.stream.stream_id,
            "cancellation_epoch": int(flush.cancellation_epoch),
            "request_id": str(flush.request_id),
            "target_generated_ssrc": int(flush.target_generated_ssrc),
        },
    )


def _connection_id(connection: ControlPeer) -> ConnectionId:
    return ConnectionId(str(id(connection)))


def _media_command_id(stream: StreamKey, epoch: int) -> str:
    return f"rtp-{stream.stream_id}-{epoch}"


def _envelope(
    *,
    event_type: str,
    correlation: EnvelopeCorrelation,
    data: dict[str, object],
    turn_id: str | None = None,
    segment_id: str | None = None,
) -> str:
    envelope: dict[str, object] = {
        "schema_version": "1.0.0",
        "event_type": event_type,
        "event_id": f"transport-{event_type}-{correlation.session_id}",
        "source": "orchestrator",
        "time": "2026-07-27T00:00:00Z",
        "trace_id": correlation.trace_id,
        "session_id": correlation.session_id,
        "seq": correlation.seq,
        "data": data,
    }

    if segment_id is not None:
        envelope["segment_id"] = segment_id

    if turn_id is not None:
        envelope["turn_id"] = turn_id

    return json.dumps(envelope, separators=(",", ":"))


@final
class _MonotonicFlushClock:
    @property
    def now_ms(self) -> int:
        return int(time.monotonic() * 1_000)
