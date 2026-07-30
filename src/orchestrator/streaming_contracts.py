"""Versioned turn and generated-media interruption contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NewType, Protocol, final

StreamingContractVersion = NewType("StreamingContractVersion", str)
TurnId = NewType("TurnId", str)
SegmentId = NewType("SegmentId", str)
CancellationEpoch = NewType("CancellationEpoch", int)
FlushRequestId = NewType("FlushRequestId", str)
GeneratedSsrc = NewType("GeneratedSsrc", int)

STREAMING_CONTRACT_VERSION: Final = StreamingContractVersion("1.0.0")
_RETRY_AFTER_MS: Final = 250
_TIMEOUT_AFTER_MS: Final = 750


class EnvelopeIdentity(Protocol):
    """Structural envelope identity retained by control-bound stream events."""

    @property
    def trace_id(self) -> str:
        """Return the transport trace identity."""
        ...

    @property
    def session_id(self) -> str:
        """Return the transport session identity."""
        ...

    @property
    def seq(self) -> int:
        """Return the transport sequence identity."""
        ...


@dataclass(frozen=True, slots=True)
class StreamKey:
    """Stable identity of one session-local generated media stream."""

    session_id: str
    stream_id: str


@dataclass(frozen=True, slots=True)
class StreamFlush:
    """Request Sound to stop one generated SSRC before replacement admission."""

    stream: StreamKey
    turn_id: TurnId
    segment_id: SegmentId
    cancellation_epoch: CancellationEpoch
    request_id: FlushRequestId
    target_generated_ssrc: GeneratedSsrc
    version: StreamingContractVersion = STREAMING_CONTRACT_VERSION
    correlation: EnvelopeIdentity | None = None


@dataclass(frozen=True, slots=True)
class FlushAcknowledgement:
    """Sound's epoch-correlated confirmation that the generated SSRC is rejected."""

    stream: StreamKey
    turn_id: TurnId
    segment_id: SegmentId
    cancellation_epoch: CancellationEpoch
    request_id: FlushRequestId
    target_generated_ssrc: GeneratedSsrc
    version: StreamingContractVersion = STREAMING_CONTRACT_VERSION
    correlation: EnvelopeIdentity | None = None

    @classmethod
    def from_flush(cls, flush: StreamFlush) -> FlushAcknowledgement:
        """Build the only acknowledgement valid for a flush request."""
        return cls(
            stream=flush.stream,
            turn_id=flush.turn_id,
            segment_id=flush.segment_id,
            cancellation_epoch=flush.cancellation_epoch,
            request_id=flush.request_id,
            target_generated_ssrc=flush.target_generated_ssrc,
            version=flush.version,
            correlation=flush.correlation,
        )


@dataclass(frozen=True, slots=True)
class FlushFailure:
    """Typed replacement-admission failure retained by the Orchestrator."""

    flush: StreamFlush
    reason: str


class FlushClock(Protocol):
    """Monotonic millisecond source used for deterministic flush deadlines."""

    @property
    def now_ms(self) -> int:
        """Return the current monotonic time in milliseconds."""
        ...


class FlushSender(Protocol):
    """Transport seam for one canonical flush envelope delivery."""

    def send_flush(self, flush: StreamFlush) -> None:
        """Deliver one canonical flush request."""
        ...


@dataclass(frozen=True, slots=True)
class _PendingFlush:
    flush: StreamFlush
    started_at_ms: int
    retried: bool = False


@final
class FlushAdmission:
    """Gate generated replacement on Sound acknowledgement."""

    def __init__(self, *, clock: FlushClock, sender: FlushSender) -> None:
        """Create a clock-driven admission gate with a flush transport seam."""
        self._clock = clock
        self._sender = sender
        self._pending: dict[StreamKey, _PendingFlush] = {}
        self._admitted: set[StreamFlush] = set()
        self.failures: list[FlushFailure] = []

    def begin(self, flush: StreamFlush) -> None:
        """Send the first flush and block replacement admission for its stream."""
        self._admitted = {
            admitted for admitted in self._admitted if admitted.stream != flush.stream
        }
        self._pending[flush.stream] = _PendingFlush(flush, self._clock.now_ms)
        self._sender.send_flush(flush)

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> bool:
        """Accept only the exact pending epoch-correlated acknowledgement."""
        pending = self._pending.get(acknowledgement.stream)
        if (
            pending is None
            or acknowledgement != FlushAcknowledgement.from_flush(pending.flush)
        ):
            flush = (
                pending.flush
                if pending is not None
                else self._flush_for_request(acknowledgement)
            )
            self.failures.append(FlushFailure(flush=flush, reason="invalid_ack"))
            return False
        del self._pending[acknowledgement.stream]
        self._admitted.add(pending.flush)
        return True

    def _flush_for_request(self, acknowledgement: FlushAcknowledgement) -> StreamFlush:
        for pending in self._pending.values():
            if pending.flush.request_id == acknowledgement.request_id:
                return pending.flush
        return StreamFlush(
                stream=acknowledgement.stream,
                turn_id=acknowledgement.turn_id,
                segment_id=acknowledgement.segment_id,
                cancellation_epoch=acknowledgement.cancellation_epoch,
                request_id=acknowledgement.request_id,
                target_generated_ssrc=acknowledgement.target_generated_ssrc,
                version=acknowledgement.version,
                correlation=acknowledgement.correlation,
        )

    def advance(self) -> None:
        """Retry once at 250ms and reject each still-pending admission at 750ms."""
        now_ms = self._clock.now_ms
        for stream, pending in tuple(self._pending.items()):
            elapsed_ms = now_ms - pending.started_at_ms
            if elapsed_ms >= _TIMEOUT_AFTER_MS:
                del self._pending[stream]
                self.failures.append(
                    FlushFailure(flush=pending.flush, reason="timeout")
                )
            elif elapsed_ms >= _RETRY_AFTER_MS and not pending.retried:
                self._sender.send_flush(pending.flush)
                self._pending[stream] = _PendingFlush(
                    flush=pending.flush,
                    started_at_ms=pending.started_at_ms,
                    retried=True,
                )

    def admitted(self, flush: StreamFlush) -> bool:
        """Return whether this exact replacement flush may begin."""
        return flush in self._admitted
