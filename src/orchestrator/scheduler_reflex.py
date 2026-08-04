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
    previous: OutputLease

    lease: OutputLease

    flush: StreamFlush


@dataclass(frozen=True, slots=True)
class SchedulerReflexRejectionError(Exception): ...


@final
class SchedulerOutputFence:
    def __init__(self, scheduler: SessionScheduler) -> None:
        self._scheduler = scheduler

        self._leases: dict[StreamKey, OutputLease] = {}

        # Completed leases are removed, but their generation is retained so a
        # later natural turn cannot accidentally reuse the prior SSRC/epoch.
        self._last_generation: dict[StreamKey, int] = {}

        self._pending: dict[StreamKey, _PendingReplacement] = {}

        # An ACK permits a prepared first frame to proceed, but it is not the
        # scheduler commit.  Retain the prior lease until the flush task has
        # crossed its result fence so cancellation immediately after ACK can
        # restore uninterrupted old playback.
        self._acknowledged: dict[StreamKey, _PendingReplacement] = {}

        # A rejected cutover restores an old turn which is no longer the
        # scheduler's active turn.  Keep that lease explicitly eligible until
        # Sound reports its natural finish or another replacement supersedes it.
        self._retained: dict[StreamKey, OutputLease] = {}

        self._flush_sequence = 0

    def activate(
        self,
        *,
        stream: StreamKey,
        segment_id: SegmentId,
        target_generated_ssrc: GeneratedSsrc | None = None,
        correlation: EnvelopeCorrelation,
    ) -> OutputLease:
        previous = self._leases.get(stream)

        generation = (
            self._last_generation.get(stream, -1) + 1
            if previous is None
            else previous.generation + 1
        )

        epoch = CancellationEpoch(generation)

        if target_generated_ssrc is None:
            target_generated_ssrc = GeneratedSsrc(generated_ssrc(stream, epoch))

        return self._activate_lease(
            stream=stream,
            segment_id=segment_id,
            turn_id=self._start_turn(correlation),
            generation=generation,
            target_generated_ssrc=target_generated_ssrc,
        )

    def activate_for_turn(
        self,
        *,
        stream: StreamKey,
        segment_id: SegmentId,
        turn_id: str,
    ) -> OutputLease | None:
        """Activate output for an already-admitted task without changing state.

        Agent-plan TTS belongs to the turn that created its task. Starting a
        second turn here would make that task stale before it can commit.
        """
        if str(self._scheduler.snapshot.active_turn_id) != turn_id:
            return None
        previous = self._leases.get(stream)
        generation = (
            self._last_generation.get(stream, -1) + 1
            if previous is None
            else previous.generation + 1
        )
        return self._activate_lease(
            stream=stream,
            segment_id=segment_id,
            turn_id=TurnId(turn_id),
            generation=generation,
            target_generated_ssrc=GeneratedSsrc(
                generated_ssrc(stream, CancellationEpoch(generation))
            ),
        )

    def _activate_lease(
        self,
        *,
        stream: StreamKey,
        segment_id: SegmentId,
        turn_id: TurnId,
        generation: int,
        target_generated_ssrc: GeneratedSsrc,
    ) -> OutputLease:
        lease = OutputLease(
            stream=stream,
            turn_id=turn_id,
            segment_id=segment_id,
            cancellation_epoch=CancellationEpoch(generation),
            generation=generation,
            target_generated_ssrc=target_generated_ssrc,
        )
        self._leases[stream] = lease
        self._last_generation[stream] = generation
        _ = self._pending.pop(stream, None)
        _ = self._acknowledged.pop(stream, None)
        _ = self._retained.pop(stream, None)
        return lease

    def interrupt(
        self,
        *,
        stream: StreamKey,
        segment_id: SegmentId,
        correlation: EnvelopeCorrelation,
    ) -> tuple[OutputLease, StreamFlush]:
        active = self._leases[stream]

        generation = active.generation + 1
        replacement = OutputLease(
            stream=stream,
            turn_id=self._start_turn(correlation),
            segment_id=segment_id,
            cancellation_epoch=CancellationEpoch(generation),
            generation=generation,
            target_generated_ssrc=GeneratedSsrc(
                generated_ssrc(stream, CancellationEpoch(generation))
            ),
        )
        self._last_generation[stream] = generation

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

        # Retain the old lease until Sound has acknowledged the flush.  This is
        # the continuity boundary: deep work is cancelled immediately, but the
        # audience keeps hearing the already-buffered old response until a new
        # first frame is ready and Sound has committed the replacement.
        self._pending[stream] = _PendingReplacement(active, replacement, flush)
        _ = self._retained.pop(stream, None)

        return replacement, flush

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> bool:
        pending = self._pending.get(acknowledgement.stream)

        if pending is None or acknowledgement != FlushAcknowledgement.from_flush(
            pending.flush
        ):
            return False

        self._leases[acknowledgement.stream] = pending.lease
        del self._pending[acknowledgement.stream]
        self._acknowledged[acknowledgement.stream] = pending

        return True

    def abandon_replacement(self, stream: StreamKey) -> bool:
        """Keep the current lease when a prepared replacement cannot be admitted."""
        if self._pending.pop(stream, None) is not None:
            return True
        acknowledged = self._acknowledged.pop(stream, None)
        if acknowledged is None:
            return False
        if self._leases.get(stream) == acknowledged.lease:
            self._leases[stream] = acknowledged.previous
            self._retained[stream] = acknowledged.previous
        return True

    def commit_replacement(
        self, stream: StreamKey, epoch: CancellationEpoch
    ) -> bool:
        """Make an ACKed replacement irreversible after reducer acceptance."""
        acknowledged = self._acknowledged.get(stream)
        if (
            acknowledged is None
            or acknowledged.lease.cancellation_epoch != epoch
            or self._leases.get(stream) != acknowledged.lease
        ):
            return False
        del self._acknowledged[stream]
        _ = self._retained.pop(stream, None)
        return True

    def can_emit(self, stream: StreamKey, epoch: CancellationEpoch) -> bool:
        lease = self._leases.get(stream)

        if lease is None:
            return False
        pending = self._pending.get(stream)
        if pending is not None:
            return epoch == pending.previous.cancellation_epoch
        retained = self._retained.get(stream)
        if retained == lease:
            return epoch == lease.cancellation_epoch
        return epoch == lease.cancellation_epoch and str(lease.turn_id) == str(
            self._scheduler.snapshot.active_turn_id
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
            or (turn_id is not None and turn_id != lease.turn_id)
            or (segment_id is not None and segment_id != lease.segment_id)
            or cancellation_epoch != lease.cancellation_epoch
        ):
            return False

        del self._leases[stream]
        _ = self._retained.pop(stream, None)

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
