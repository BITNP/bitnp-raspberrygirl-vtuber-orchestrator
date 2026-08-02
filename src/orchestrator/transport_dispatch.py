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
    FlushAcknowledgement,
    FlushAdmission,
    FlushClock,
    FlushFailure,
    StreamFlush,
    StreamKey,
)
from orchestrator.transport_control import (
    ControlEvent,
    EnvelopeCorrelation,
    MicInputRegistration,
    SinkRegistration,
    SourceRegistration,
    StreamReady,
    StreamState,
    parse_control_event,
)

if TYPE_CHECKING:
    from orchestrator.scheduler_reflex import SchedulerOutputFence


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

    def advance_onsite_epoch(self, stream: StreamKey, epoch: int) -> None: ...


@dataclass(frozen=True, slots=True)
class _SourcePeer:
    connection: ControlPeer

    ssrc: int


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
    ) -> None:
        self._hub: RouteRegistry = hub

        self._observability = observability

        self._sources: dict[StreamKey, _SourcePeer] = {}

        self._sinks: dict[StreamKey, _SinkPeer] = {}

        self._dispatched: set[StreamKey] = set()

        self._ready_sinks: set[StreamKey] = set()

        self._released_sources: set[StreamKey] = set()

        self._flush_outbox: list[StreamFlush] = []

        self._flush_admission = FlushAdmission(
            clock=_MonotonicFlushClock() if clock is None else clock,
            sender=self,
        )

        self._output_fence: SchedulerOutputFence | None = None

    async def register(  # noqa: C901, PLR0911, PLR0912
        self, raw_message: str, peer_ip: str, connection: ControlPeer
    ) -> None:
        event = parse_control_event(raw_message)

        _LOGGER.debug(
            "control_register event=%s peer=%s", type(event).__name__, peer_ip
        )

        self._hub.register_control(event, peer_ip, _connection_id(connection))

        if isinstance(event, StreamState):
            self._record_playback(event)

        match event:
            case MicInputRegistration(session_id=session_id, stream_id=stream_id):
                await self._dispatch_start(StreamKey(session_id, stream_id))
                return

            case SourceRegistration(
                session_id=session_id, stream_id=stream_id, ssrc=ssrc
            ):
                self._sources[StreamKey(session_id, stream_id)] = _SourcePeer(
                    connection, ssrc
                )

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
                self._discard(StreamKey(session_id, stream_id))

            case StreamState(
                session_id=session_id,
                stream_id=stream_id,
                state="finished",
                turn_id=turn_id,
                segment_id=segment_id,
                cancellation_epoch=cancellation_epoch,
            ):
                output_fence = self._output_fence

                if output_fence is not None:
                    released = output_fence.finish(
                        stream=StreamKey(session_id, stream_id),
                        turn_id=turn_id,
                        segment_id=segment_id,
                        cancellation_epoch=cancellation_epoch,
                    )

                    if released and cancellation_epoch is not None:
                        # Retire the actor that owns the completed packetizer.
                        # The next microphone frame constructs a fresh actor whose
                        # input epoch equals the next scheduler lease epoch.
                        self._hub.advance_onsite_epoch(
                            StreamKey(session_id, stream_id),
                            int(cancellation_epoch) + 1,
                        )

                return

            case StreamState():
                return

            case FlushAcknowledgement() as acknowledgement:
                self._record_flush_acknowledgement(acknowledgement)

                admitted = self._flush_admission.acknowledge(acknowledgement)

                output_fence = self._output_fence

                if admitted and output_fence is not None:
                    _ = output_fence.acknowledge(acknowledgement)

                return

            case _:
                return

        await self._dispatch_start(StreamKey(event.session_id, event.stream_id))

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

        source = self._sources.get(flush.stream)

        sink = self._sinks.get(flush.stream)

        if source is None or sink is None:
            return False

        await sink.connection.send(
            _stream_command_envelope(
                flush.stream,
                self._hub.output_ssrc(flush.stream, int(flush.cancellation_epoch)),
                sink,
                self._hub.correlation(flush.stream),
                flush,
            )
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
                stream, epoch, self._hub.output_ssrc(stream, epoch), correlation
            )
        )
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
                sink,
                correlation,
                epoch=epoch,
            )
        )
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
        self._sources.clear()

        self._sinks.clear()

        self._dispatched.clear()

        self._ready_sinks.clear()

        self._released_sources.clear()

        self._flush_outbox.clear()

    def set_observability(self, observability: OnsiteObservability) -> None:
        self._observability = observability

    def set_output_fence(self, output_fence: SchedulerOutputFence) -> None:
        self._output_fence = output_fence

    def remove_connection(self, connection: ControlPeer) -> None:
        self._hub.remove_connection(_connection_id(connection))

        for stream, source in tuple(self._sources.items()):
            if source.connection is connection:
                del self._sources[stream]

                self._dispatched.discard(stream)

                self._ready_sinks.discard(stream)

                self._released_sources.discard(stream)

        for stream, sink in tuple(self._sinks.items()):
            if sink.connection is connection:
                del self._sinks[stream]

                self._dispatched.discard(stream)

                self._ready_sinks.discard(stream)

                self._released_sources.discard(stream)

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
                stream, self._hub.output_ssrc(stream), sink, correlation
            )
        )

    async def _release_source(self, stream: StreamKey) -> None:
        source = self._sources.get(stream)

        if (
            source is None
            or stream not in self._ready_sinks
            or stream in self._released_sources
        ):
            return

        correlation = self._hub.correlation(stream)

        if correlation is None:
            return

        self._released_sources.add(stream)

        await source.connection.send(
            _source_ready_envelope(stream, source.ssrc, correlation)
        )

    async def _deliver_flushes(self) -> None:
        while self._flush_outbox:
            flush = self._flush_outbox.pop(0)

            source = self._sources.get(flush.stream)

            sink = self._sinks.get(flush.stream)

            correlation = self._hub.correlation(flush.stream)

            if correlation is not None:
                envelope = _flush_envelope(flush, correlation)

                if source is not None:
                    await source.connection.send(envelope)

                if sink is not None:
                    await sink.connection.send(envelope)

    def _discard(self, stream: StreamKey) -> None:
        _ = self._sources.pop(stream, None)

        _ = self._sinks.pop(stream, None)

        self._dispatched.discard(stream)

        self._ready_sinks.discard(stream)

        self._released_sources.discard(stream)

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


def _source_ready_envelope(
    stream: StreamKey, ssrc: int, correlation: EnvelopeCorrelation
) -> str:
    return _envelope(
        event_type="media.rtp.source.ready",
        correlation=correlation,
        data={"stream_id": stream.stream_id, "ssrc": ssrc},
    )


def _stream_command_envelope(  # noqa: PLR0913
    stream: StreamKey,
    ssrc: int,
    sink: _SinkPeer,
    correlation: EnvelopeCorrelation | None,
    flush: StreamFlush | None = None,
    epoch: int | None = None,
) -> str:
    if correlation is None:
        message = "stream correlation is required"

        raise RuntimeError(message)

    data: dict[str, object] = {
        "command_id": f"rtp-{stream.stream_id}",
        "stream_id": stream.stream_id,
        "start_rtp_timestamp": 96_000,
        "ssrc": ssrc,
        "codec": _CODEC,
        "rtp_endpoint": {"host": sink.host, "port": sink.udp_port},
    }

    if flush is not None:
        data["cancellation_epoch"] = int(flush.cancellation_epoch)
    elif epoch is not None:
        data["cancellation_epoch"] = epoch

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


def _stream_end_envelope(
    stream: StreamKey, epoch: int, ssrc: int, correlation: EnvelopeCorrelation
) -> str:
    return _envelope(
        event_type="media.stream.end",
        correlation=correlation,
        data={"stream_id": stream.stream_id, "cancellation_epoch": epoch, "ssrc": ssrc},
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
