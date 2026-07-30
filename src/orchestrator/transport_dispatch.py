"""Outbound control dispatch for the Orchestrator-owned RTP transport."""

from __future__ import annotations

import json
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


class ControlPeer(Protocol):
    """A live control session that can receive canonical text envelopes."""

    async def send(self, message: str) -> None:
        """Send one canonical control envelope."""


class RouteRegistry(Protocol):
    """Applies inbound route registrations and removes cancelled stream routes."""

    def register_control(
        self,
        raw_message: ControlEvent | str,
        peer_ip: str,
        owner: ConnectionId | None = None,
    ) -> None:
        """Apply one authenticated inbound control envelope."""

    def remove_connection(self, owner: ConnectionId) -> None:
        """Remove route components owned by one closed WSS connection."""

    def remove_stream(self, session_id: str, stream_id: str) -> None:
        """Remove one stream route before cancellation reaches Sound."""

    def output_ssrc(self, stream: StreamKey, cancellation_epoch: int = 0) -> int:
        """Return the SSRC announced to Sound for the active media mode."""
        ...

    def correlation(self, stream: StreamKey) -> EnvelopeCorrelation | None:
        """Return source-envelope correlation retained for a live route."""
        ...


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
    """Coordinates Mic and Sound sessions through the hub control boundary."""

    def __init__(
        self,
        hub: RouteRegistry,
        *,
        clock: FlushClock | None = None,
        observability: OnsiteObservability | None = None,
    ) -> None:
        """Create a dispatcher bound to the authoritative hub route registry."""
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

    async def register(
        self, raw_message: str, peer_ip: str, connection: ControlPeer
    ) -> None:
        """Dispatch startup messages after both peers have registered."""
        event = parse_control_event(raw_message)
        self._hub.register_control(event, peer_ip, _connection_id(connection))
        if isinstance(event, StreamState):
            self._record_playback(event)
        match event:
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
                await self._release_source(stream)
                return
            case StreamState(
                session_id=session_id,
                stream_id=stream_id,
                state="cancelled" | "finished" | "error",
            ):
                self._discard(StreamKey(session_id, stream_id))
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
        """Send Sound a flush and block replacement."""
        correlation = flush.correlation or self._hub.correlation(flush.stream)
        if correlation is None:
            return
        flush = replace(flush, correlation=correlation)
        self._record_flush(flush)
        self._flush_admission.begin(flush)
        await self._deliver_flushes()

    async def advance_flush_admission(self) -> None:
        """Deliver a retry or record timeout from the runtime clock."""
        self._flush_admission.advance()
        await self._deliver_flushes()

    async def admit_replacement(self, flush: StreamFlush) -> bool:
        """Create replacement media only after Sound admission."""
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
        return True

    @property
    def flush_failures(self) -> tuple[FlushFailure, ...]:
        """Expose typed failed replacement admissions to the runtime boundary."""
        return tuple(self._flush_admission.failures)

    def send_flush(self, flush: StreamFlush) -> None:
        """Queue a canonical flush synchronously for the admission state machine."""
        self._flush_outbox.append(flush)

    async def cancel_stream(self, session_id: str, stream_id: str) -> None:
        """Stop forwarding first, then signal Sound to suppress late RTP."""
        stream = StreamKey(session_id, stream_id)
        self._record_transport_transition("cancellation", stream)
        sink = self._sinks.get(stream)
        correlation = self._hub.correlation(stream)
        self._hub.remove_stream(session_id, stream_id)
        self._discard(stream)
        if sink is not None and correlation is not None:
            await sink.connection.send(_cancel_envelope(stream_id, correlation))

    def clear(self) -> None:
        """Release retained live-session references during runtime shutdown."""
        self._sources.clear()
        self._sinks.clear()
        self._dispatched.clear()
        self._ready_sinks.clear()
        self._released_sources.clear()
        self._flush_outbox.clear()

    def set_observability(self, observability: OnsiteObservability) -> None:
        """Attach the runtime-shared onsite recorder."""
        self._observability = observability

    def set_output_fence(self, output_fence: SchedulerOutputFence) -> None:
        """Forward Sound acknowledgements to the scheduler-owned output fence."""
        self._output_fence = output_fence

    def remove_connection(self, connection: ControlPeer) -> None:
        """Discard only control and RTP state owned by one closed WSS peer."""
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
        source = self._sources.get(stream)
        sink = self._sinks.get(stream)
        if source is None or sink is None or stream in self._dispatched:
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
        """Record Sound's actual acknowledgement envelope and command identity."""
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


def _stream_command_envelope(
    stream: StreamKey,
    ssrc: int,
    sink: _SinkPeer,
    correlation: EnvelopeCorrelation | None,
    flush: StreamFlush | None = None,
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
    """Build a new outbound envelope.

    Correlation deliberately excludes event identity.
    """
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
