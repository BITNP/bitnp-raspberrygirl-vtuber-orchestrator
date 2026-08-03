from orchestrator.ids import SessionId, TurnId
from orchestrator.sessions import StateRevision
from orchestrator.task_lifecycle import TaskResultFence, result_is_current
from orchestrator.task_registry import (
    IdempotencyKey,
    SchedulerTaskConfig,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRegistry,
    TaskRequest,
)


def _running_record() -> tuple[TaskRegistry, TaskId]:
    registry = TaskRegistry(
        session_id=SessionId("session-1"),
        config=SchedulerTaskConfig(frozenset(TaskKind), max_children_per_task=2),
    )
    task_id = TaskId("task-1")
    result = registry.register(
        TaskRequest(
            task_id=task_id,
            session_id=SessionId("session-1"),
            turn_id=TurnId("turn-1"),
            parent_task_id=None,
            deadline_ms=TaskDeadlineMs(100),
            snapshot_revision=StateRevision(2),
            idempotency_key=IdempotencyKey("task-1"),
            kind=TaskKind.INTERACTIVE,
        )
    )
    assert result.__class__.__name__ == "TaskRegistrationAccepted"
    assert registry.claim(task_id) is not None
    return registry, task_id


def test_result_fence_rejects_cancelled_or_late_provider_work() -> None:
    registry, task_id = _running_record()
    fence = TaskResultFence(
        SessionId("session-1"), TurnId("turn-1"), StateRevision(2), 0, 100
    )
    assert result_is_current(registry.task(task_id), fence)
    _ = registry.cancel(task_id, reason="interrupt")
    assert not result_is_current(registry.task(task_id), fence)


def test_result_fence_rejects_a_different_cancellation_epoch() -> None:
    registry, task_id = _running_record()
    fence = TaskResultFence(
        SessionId("session-1"), TurnId("turn-1"), StateRevision(2), 1, 100
    )

    assert not result_is_current(registry.task(task_id), fence)


def test_registry_can_mark_provider_failure_terminal() -> None:
    registry, task_id = _running_record()
    record = registry.fail(task_id, reason="provider_unavailable")
    assert record is not None
    assert record.state.value == "failed"
    assert registry.claim(task_id) is None


def test_terminal_task_is_retained_until_explicit_tombstone_cleanup() -> None:
    registry, task_id = _running_record()
    assert registry.cancel(task_id, reason="interrupt") is not None
    assert registry.task(task_id) is not None
    removed = registry.clear_terminal_tombstones()
    assert tuple(record.request.task_id for record in removed) == (task_id,)
    assert registry.task(task_id) is None
