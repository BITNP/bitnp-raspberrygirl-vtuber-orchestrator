
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, NewType, Protocol, final

StreamingContractVersion = NewType("StreamingContractVersion", str)

TurnId = NewType("TurnId", str)

SegmentId = NewType("SegmentId", str)

CancellationEpoch = NewType("CancellationEpoch", int)

FlushRequestId = NewType("FlushRequestId", str)

GeneratedSsrc = NewType("GeneratedSsrc", int)


class FlushDisposition(StrEnum):
    APPLIED = "APPLIED"
    REPLAYED = "REPLAYED"


STREAMING_CONTRACT_VERSION: Final = StreamingContractVersion("1.0.0")

_RETRY_AFTER_MS: Final = 250

_TIMEOUT_AFTER_MS: Final = 750


class EnvelopeIdentity(Protocol):

    @property
    def trace_id(self) -> str:
        ...

    @property
    def session_id(self) -> str:
        ...

    @property
    def seq(self) -> int:
        ...


@dataclass(frozen=True, slots=True)
class StreamKey:

    session_id: str

    stream_id: str


@dataclass(frozen=True, slots=True)
class StreamFlush:

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

    stream: StreamKey

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch

    request_id: FlushRequestId

    target_generated_ssrc: GeneratedSsrc

    disposition: FlushDisposition = FlushDisposition.APPLIED

    version: StreamingContractVersion = STREAMING_CONTRACT_VERSION

    correlation: EnvelopeIdentity | None = None

    @classmethod
    def from_flush(
        cls,
        flush: StreamFlush,
        disposition: FlushDisposition = FlushDisposition.APPLIED,
    ) -> FlushAcknowledgement:
        return cls(
            stream=flush.stream,
            turn_id=flush.turn_id,
            segment_id=flush.segment_id,
            cancellation_epoch=flush.cancellation_epoch,
            request_id=flush.request_id,
            target_generated_ssrc=flush.target_generated_ssrc,
            disposition=disposition,
            version=flush.version,
            correlation=flush.correlation,
        )


@dataclass(frozen=True, slots=True)
class FlushFailure:

    flush: StreamFlush

    reason: str


class FlushClock(Protocol):

    @property
    def now_ms(self) -> int:
        ...


class FlushSender(Protocol):

    def send_flush(self, flush: StreamFlush) -> None:
        ...


@dataclass(frozen=True, slots=True)
class _PendingFlush:

    flush: StreamFlush

    started_at_ms: int

    retried: bool = False


@final
class FlushAdmission:

    def __init__(self, *, clock: FlushClock, sender: FlushSender) -> None:
        self._clock = clock

        self._sender = sender

        self._pending: dict[StreamKey, _PendingFlush] = {}

        self._admitted: set[StreamFlush] = set()

        self.failures: list[FlushFailure] = []

    def begin(self, flush: StreamFlush) -> None:
        self._admitted = {
            admitted for admitted in self._admitted if admitted.stream != flush.stream
        }

        self._pending[flush.stream] = _PendingFlush(flush, self._clock.now_ms)

        self._sender.send_flush(flush)

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> bool:
        pending = self._pending.get(acknowledgement.stream)

        if pending is None or acknowledgement not in {
            FlushAcknowledgement.from_flush(pending.flush, FlushDisposition.APPLIED),
            FlushAcknowledgement.from_flush(pending.flush, FlushDisposition.REPLAYED),
        }:
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
        return flush in self._admitted
