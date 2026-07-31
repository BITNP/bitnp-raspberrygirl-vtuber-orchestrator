from dataclasses import replace

import pytest

from orchestrator.ids import SessionId, TurnId
from orchestrator.interactions import (
    ActionCapabilityRegistry,
    ActionProposal,
    CommandId,
    InteractionAccepted,
    InteractionRejection,
    InteractionRejectionReason,
    SessionInteractionReducer,
)
from orchestrator.scheduler_reflex import SchedulerOutputFence
from orchestrator.sessions import (
    SessionScheduler,
    SessionSnapshot,
    StateRevision,
)
from orchestrator.state_snapshots import MemoryRevision, TaskStateSnapshot
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    GeneratedSsrc,
    SegmentId,
    StreamKey,
)
from orchestrator.task_reducer import (
    TaskEffect,
    TaskResult,
    TaskResultAccepted,
    TaskResultReducer,
    TaskResultRejected,
    TaskResultRejection,
)
from orchestrator.task_registry import (
    IdempotencyKey,
    SchedulerTaskConfig,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRegistrationAccepted,
    TaskRegistry,
    TaskRequest,
)
from orchestrator.transport_control import EnvelopeCorrelation


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        ("unknown", TaskResultRejection.TASK_NOT_FOUND),
        ("timed_out", TaskResultRejection.DEADLINE_EXCEEDED),
        ("session", TaskResultRejection.SESSION_MISMATCH),
        ("turn", TaskResultRejection.TURN_MISMATCH),
        ("revision", TaskResultRejection.STALE_REVISION),
        ("data", TaskResultRejection.STALE_DATA_SNAPSHOT),
    ],
)
def test_task_reducer_rejects_each_uncommittable_result(
    scenario: str, reason: TaskResultRejection
) -> None:
    # Given: one registered task and a baseline scheduler snapshot.
    registry = _registry()
    request = _request()
    _register(registry, request)
    reducer = TaskResultReducer(registry)
    result = _result()
    snapshot = _snapshot()
    data_snapshot = TaskStateSnapshot.initial()

    # When: a result crosses one rejection boundary.
    match scenario:
        case "unknown":
            result = replace(result, task_id=TaskId("missing"))
        case "timed_out":
            _ = registry.timeout(request.task_id)
        case "session":
            snapshot = replace(snapshot, session_id=SessionId("other-session"))
        case "turn":
            snapshot = replace(snapshot, active_turn_id=TurnId("other-turn"))
        case "revision":
            snapshot = replace(snapshot, revision=StateRevision(8))
        case "data":
            data_snapshot = replace(
                TaskStateSnapshot.initial(), memory_revision=MemoryRevision(1)
            )
        case _:
            pytest.fail(f"unknown characterization scenario: {scenario}")

    outcome = reducer.reduce(
        result, snapshot=snapshot, now_ms=100, data_snapshot=data_snapshot
    )

    # Then: no uncommittable result produces an accepted effect.
    match outcome:
        case TaskResultRejected(reason=actual_reason):
            assert actual_reason is reason
        case TaskResultAccepted():
            pytest.fail(f"{scenario} result committed an effect")


def test_output_fence_correlates_flush_epoch_41_and_never_resumes_stale_audio() -> None:
    # Given: an output lease emitting generated audio under epoch 40.
    scheduler = SessionScheduler(
        session_id=SessionId("session-1"), turn_id_prefix="turn"
    )
    fence = SchedulerOutputFence(scheduler)
    stream = StreamKey(session_id="session-1", stream_id="stream-1")
    for sequence in range(41):
        _ = fence.activate(
            stream=stream,
            segment_id=SegmentId("segment-40"),
            target_generated_ssrc=GeneratedSsrc(0x1234_5678),
            correlation=EnvelopeCorrelation("trace-1", "session-1", sequence),
        )

    # When: interruption creates epoch 41, then a later interruption creates epoch 42.
    epoch_41, flush_41 = fence.interrupt(
        stream=stream,
        segment_id=SegmentId("segment-41"),
        correlation=EnvelopeCorrelation("trace-1", "session-1", 41),
    )
    acknowledged_41 = fence.acknowledge(FlushAcknowledgement.from_flush(flush_41))
    assert epoch_41.cancellation_epoch == CancellationEpoch(41)
    assert flush_41.cancellation_epoch == CancellationEpoch(41)
    assert acknowledged_41 is True
    assert fence.can_emit(stream, CancellationEpoch(41)) is True
    epoch_42, flush_42 = fence.interrupt(
        stream=stream,
        segment_id=SegmentId("segment-42"),
        correlation=EnvelopeCorrelation("trace-1", "session-1", 42),
    )

    # Then: each Sound acknowledgement is epoch-correlated.
    # Stale epoch 41 cannot emit.
    assert epoch_42.cancellation_epoch == CancellationEpoch(42)
    assert flush_42.cancellation_epoch == CancellationEpoch(42)
    assert fence.can_emit(stream, CancellationEpoch(41)) is False
    assert fence.can_emit(stream, CancellationEpoch(42)) is False
    assert fence.acknowledge(FlushAcknowledgement.from_flush(flush_42)) is True
    assert fence.can_emit(stream, CancellationEpoch(42)) is True


def test_output_fence_releases_only_exact_finished_lease_and_advances_epoch() -> None:
    scheduler = SessionScheduler(
        session_id=SessionId("session-1"), turn_id_prefix="turn"
    )
    fence = SchedulerOutputFence(scheduler)
    stream = StreamKey(session_id="session-1", stream_id="stream-1")
    correlation = EnvelopeCorrelation("trace-1", "session-1", 1)
    first = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-1"),
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
        correlation=correlation,
    )

    # A forged completion cannot free an active output lease.
    assert (
        fence.finish(
            stream=stream,
            turn_id=first.turn_id,
            segment_id=SegmentId("other-segment"),
            cancellation_epoch=first.cancellation_epoch,
        )
        is False
    )
    assert fence.can_emit(stream, first.cancellation_epoch) is True

    assert (
        fence.finish(
            stream=stream,
            turn_id=first.turn_id,
            segment_id=first.segment_id,
            cancellation_epoch=first.cancellation_epoch,
        )
        is True
    )

    # A natural next turn gets a distinct epoch instead of reviving old RTP.
    second = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-2"),
        target_generated_ssrc=GeneratedSsrc(0x8765_4321),
        correlation=EnvelopeCorrelation("trace-1", "session-1", 2),
    )
    assert second.cancellation_epoch == CancellationEpoch(1)


def test_action_reducer_allows_one_allowlisted_command_and_rejects_replay() -> None:
    # Given: a reducer that permits only the finite wave avatar action.
    reducer = SessionInteractionReducer(
        scheduler=SessionScheduler(
            session_id=SessionId("session-1"), turn_id_prefix="turn"
        ),
        actions=ActionCapabilityRegistry(frozenset({"wave"})),
        mcp_capabilities=frozenset(),
    )
    proposal = ActionProposal("wave", CommandId("wave-1"))

    # When: the same allowlisted command is proposed twice.
    dispatched = reducer.reduce_action(proposal)
    replayed = reducer.reduce_action(proposal)

    # Then: exactly one typed action becomes dispatchable.
    assert dispatched == InteractionAccepted(command_id=CommandId("wave-1"))
    assert replayed == InteractionRejection(InteractionRejectionReason.DUPLICATE)


def _registry() -> TaskRegistry:
    return TaskRegistry(
        session_id=SessionId("session-1"),
        config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )


def _request() -> TaskRequest:
    return TaskRequest(
        task_id=TaskId("task-1"),
        session_id=SessionId("session-1"),
        turn_id=TurnId("turn-1"),
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(200),
        snapshot_revision=StateRevision(7),
        idempotency_key=IdempotencyKey("answer-1"),
        kind=TaskKind.INTERACTIVE,
    )


def _register(registry: TaskRegistry, request: TaskRequest) -> None:
    match registry.register(request):
        case TaskRegistrationAccepted(record=record):
            assert record == registry.task(request.task_id)
        case _:
            pytest.fail("task fixture was not admitted")


def _result() -> TaskResult:
    return TaskResult(
        task_id=TaskId("task-1"),
        session_id=SessionId("session-1"),
        turn_id=TurnId("turn-1"),
        snapshot_revision=StateRevision(7),
        effect=TaskEffect("answer", "accepted"),
    )


def _snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        session_id=SessionId("session-1"),
        revision=StateRevision(7),
        active_turn_id=TurnId("turn-1"),
    )
