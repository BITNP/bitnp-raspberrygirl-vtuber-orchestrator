
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from orchestrator.ids import TraceId
from orchestrator.sessions import (
    EventCorrelation,
    EventSequence,
    SchedulerEvent,
    SessionScheduler,
    StartTurn,
    TransitionAccepted,
)
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
from orchestrator.tts_rtp import generated_ssrc

if TYPE_CHECKING:
    from orchestrator.transport_control import EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class OutputLease:

    stream: StreamKey

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch

    generation: int

    target_generated_ssrc: GeneratedSsrc


@dataclass(frozen=True, slots=True)
class _PendingReplacement:

    lease: OutputLease

    flush: StreamFlush


@dataclass(frozen=True, slots=True)
class SchedulerReflexRejectionError(Exception):
    ...


@final
class SchedulerOutputFence:

    def __init__(self, scheduler: SessionScheduler) -> None:
        self._scheduler = scheduler

        self._leases: dict[StreamKey, OutputLease] = {}

        # Completed leases are removed, but their generation is retained so a
        # later natural turn cannot accidentally reuse the prior SSRC/epoch.
        self._last_generation: dict[StreamKey, int] = {}

        self._pending: dict[StreamKey, _PendingReplacement] = {}

        self._flush_sequence = 0

    def activate(
        self,
        *,
        stream: StreamKey,
        segment_id: SegmentId,
        target_generated_ssrc: GeneratedSsrc,
        correlation: EnvelopeCorrelation,
    ) -> OutputLease:
        previous = self._leases.get(stream)

        generation = (
            self._last_generation.get(stream, -1) + 1
            if previous is None
            else previous.generation + 1
        )

        epoch = CancellationEpoch(generation)

        lease = OutputLease(
            stream=stream,
            turn_id=self._start_turn(correlation),
            segment_id=segment_id,
            cancellation_epoch=epoch,
            generation=generation,
            target_generated_ssrc=target_generated_ssrc,
        )

        self._leases[stream] = lease

        self._last_generation[stream] = generation

        _ = self._pending.pop(stream, None)

        return lease

    def interrupt(
        self,
        *,
        stream: StreamKey,
        segment_id: SegmentId,
        correlation: EnvelopeCorrelation,
    ) -> tuple[OutputLease, StreamFlush]:
        active = self._leases[stream]

        replacement = self.activate(
            stream=stream,
            segment_id=segment_id,
            target_generated_ssrc=GeneratedSsrc(
                generated_ssrc(
                    stream, CancellationEpoch(int(active.cancellation_epoch) + 1)
                )
            ),
            correlation=correlation,
        )

        self._flush_sequence += 1

        flush = StreamFlush(
            stream=stream,
            turn_id=active.turn_id,
            segment_id=active.segment_id,
            cancellation_epoch=replacement.cancellation_epoch,
            request_id=FlushRequestId(
                f"{stream.session_id}:{stream.stream_id}:flush:{self._flush_sequence}"
            ),
            target_generated_ssrc=active.target_generated_ssrc,
            correlation=correlation,
        )

        self._pending[stream] = _PendingReplacement(replacement, flush)

        return replacement, flush

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> bool:
        pending = self._pending.get(acknowledgement.stream)

        if pending is None or acknowledgement != FlushAcknowledgement.from_flush(
            pending.flush
        ):
            return False

        del self._pending[acknowledgement.stream]

        return True

    def can_emit(self, stream: StreamKey, epoch: CancellationEpoch) -> bool:
        lease = self._leases.get(stream)

        return (
            lease is not None
            and stream not in self._pending
            and epoch == lease.cancellation_epoch
            and str(lease.turn_id) == str(self._scheduler.snapshot.active_turn_id)
        )

    def finish(
        self,
        *,
        stream: StreamKey,
        turn_id: TurnId | None,
        segment_id: SegmentId | None,
        cancellation_epoch: CancellationEpoch | None,
    ) -> bool:
        """Release only the exact lease Sound reports as physically consumed.

        A finished notification is advisory until all correlation fields match.
        In particular, it must not release a replacement that is awaiting a
        flush acknowledgement.
        """
        lease = self._leases.get(stream)

        if (
            lease is None
            or stream in self._pending
            or turn_id != lease.turn_id
            or segment_id != lease.segment_id
            or cancellation_epoch != lease.cancellation_epoch
        ):
            return False

        del self._leases[stream]

        return True

    def _start_turn(self, correlation: EnvelopeCorrelation) -> TurnId:
        result = self._scheduler.apply(
            StartTurn(
                expected_revision=self._scheduler.snapshot.revision,
                event=SchedulerEvent(
                    event_type="asr.final",
                    correlation=EventCorrelation(
                        trace_id=TraceId(correlation.trace_id),
                        session_id=self._scheduler.snapshot.session_id,
                        sequence=EventSequence(correlation.seq),
                    ),
                ),
            )
        )

        if not isinstance(result, TransitionAccepted):
            raise SchedulerReflexRejectionError

        return TurnId(str(result.accepted_event.turn_id))
