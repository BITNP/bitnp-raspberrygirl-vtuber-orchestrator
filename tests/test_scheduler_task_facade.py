from dataclasses import replace

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.interactions import CommentProposal
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence, StateRevision
from orchestrator.state_snapshots import MemoryRevision
from orchestrator.task_reducer import TaskEffect, TaskResult
from orchestrator.task_registry import (
    IdempotencyKey,
    SchedulerTaskConfig,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRequest,
)


def test_runtime_rejects_stale_snapshot_before_registry_or_lane_mutation() -> None:
    # Given: a live runtime and a request captured before its active revision.
    runtime = _runtime()
    request = replace(_request(runtime), snapshot_revision=StateRevision(0))

    # When: stale work reaches the only runtime admission surface.
    outcome = runtime.schedule_task(request, _correlation("task", 2))

    # Then: it is neither retained nor available to a worker.
    assert outcome.accepted is False
    assert runtime.task_registry.records == ()
    assert runtime.next_task(now_ms=0) is None


def test_runtime_rejects_inactive_turn_before_registry_or_lane_mutation() -> None:
    # Given: a live runtime and a request for another turn.
    runtime = _runtime()
    request = replace(_request(runtime), turn_id=TurnId("inactive-turn"))

    # When: inactive work reaches runtime admission.
    outcome = runtime.schedule_task(request, _correlation("task", 2))

    # Then: it is neither retained nor available to a worker.
    assert outcome.accepted is False
    assert runtime.task_registry.records == ()
    assert runtime.next_task(now_ms=0) is None


def test_runtime_rejects_explicitly_stale_data_snapshot_before_enqueue() -> None:
    # Given: a live runtime and a caller-supplied stale data revision.
    runtime = _runtime()
    stale_snapshot = replace(
        runtime.interaction_ingress.data.task_snapshot,
        memory_revision=MemoryRevision(1),
    )
    request = replace(_request(runtime), data_snapshot=stale_snapshot)

    # When: the stale data request reaches runtime admission.
    outcome = runtime.schedule_task(request, _correlation("task", 2))

    # Then: it cannot create registry or lane state.
    assert outcome.accepted is False
    assert runtime.task_registry.records == ()
    assert runtime.next_task(now_ms=0) is None


def test_runtime_rejects_stale_result_after_newer_turn_without_commit() -> None:
    # Given: admitted work for a current runtime turn.
    runtime = _runtime()
    request = _request(runtime)
    correlation = _correlation("task", 2)
    assert runtime.schedule_task(request, correlation).accepted

    # When: a newer turn invalidates the original result.
    _ = runtime.receive_comment(CommentProposal("newer", _correlation("newer", 3)))
    outcome = runtime.reduce_task(_result(request), correlation)

    # Then: no effect is committed.
    assert outcome.accepted is False
    assert runtime.observables.task_commits == ()


def _runtime() -> SessionRuntime:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        clock=lambda: 0,
    )
    opened = runtime.receive_comment(CommentProposal("start", _correlation("turn", 1)))
    assert opened.accepted
    return runtime


def _request(runtime: SessionRuntime) -> TaskRequest:
    turn_id = runtime.scheduler.snapshot.active_turn_id
    assert turn_id is not None
    return TaskRequest(
        task_id=TaskId("task-1"),
        session_id=SessionId("session-1"),
        turn_id=turn_id,
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(100),
        snapshot_revision=runtime.scheduler.snapshot.revision,
        idempotency_key=IdempotencyKey("answer-1"),
        kind=TaskKind.INTERACTIVE,
    )


def _result(request: TaskRequest) -> TaskResult:
    return TaskResult(
        task_id=request.task_id,
        session_id=request.session_id,
        turn_id=request.turn_id,
        snapshot_revision=request.snapshot_revision,
        effect=TaskEffect("answer", "accepted"),
    )


def _correlation(trace_id: str, sequence: int) -> EventCorrelation:
    return EventCorrelation(
        TraceId(trace_id), SessionId("session-1"), EventSequence(sequence)
    )
