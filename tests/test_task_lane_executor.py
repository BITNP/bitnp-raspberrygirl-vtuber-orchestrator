"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import replace

from orchestrator.ids import SessionId, TurnId
from orchestrator.sessions import StateRevision
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.task_executor import TaskLaneExecutor
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


def test_reflex_runs_before_queued_lower_priority_lanes() -> None:
    # Given: queued work in every lane plus a reflex task submitted last.

    """函数契约说明.

    功能: 验证 reflex runs before queued
    lower priority lanes 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = _registry()

    executor = TaskLaneExecutor(registry, max_pending_per_lane=2)

    requests = tuple(_request(kind, index) for index, kind in enumerate(TaskKind))

    for request in requests:
        _admit(registry, request)

        assert executor.enqueue(request) is True

    # When: the executor selects the next scheduler-authorized work item.

    selected = executor.next(now_ms=0)

    # Then: reflex work preempts all queued LLM-capable lanes.

    assert selected == requests[0]


def test_executor_bounds_each_lane_and_expires_work_before_execution() -> None:
    # Given: one-slot lanes and a deadline already elapsed on an interactive task.

    """函数契约说明.

    功能: 验证 executor bounds each lane and
    expires work before execution
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = _registry()

    executor = TaskLaneExecutor(registry, max_pending_per_lane=1)

    expired = _request(TaskKind.INTERACTIVE, 1, deadline_ms=10)

    overflow = _request(TaskKind.INTERACTIVE, 2, deadline_ms=20)

    _admit(registry, expired)

    _admit(registry, overflow)

    assert executor.enqueue(expired) is True

    assert executor.enqueue(overflow) is False

    # When: the fake clock advances past the pending task deadline.

    selected = executor.next(now_ms=11)

    # Then: no expired work is handed to an LLM-capable caller.

    assert selected is None

    record = registry.task(expired.task_id)

    assert record is not None

    assert record.state.value == "timed_out"


def test_executor_skips_superseded_task_before_worker_selection() -> None:
    # Given: a queued task replaced by work for a newer turn.

    """函数契约说明.

    功能: 验证 executor skips superseded
    task before worker selection
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = _registry()

    executor = TaskLaneExecutor(registry, max_pending_per_lane=1)

    original = _request(TaskKind.DELIBERATIVE, 1)

    replacement = replace(_request(TaskKind.DELIBERATIVE, 2), turn_id=TurnId("turn-2"))

    _admit(registry, original)

    _admit(registry, replacement)

    assert executor.enqueue(original) is True

    assert (
        registry.supersede(original.task_id, replacement_task_id=replacement.task_id)
        is not None
    )

    # When: a worker asks for deliberative work.

    selected = executor.next(now_ms=0)

    # Then: superseded queued work is never selected.

    assert selected is None


def _registry() -> TaskRegistry:
    """函数契约说明.

    功能: 执行 _registry 的同步逻辑,并协调
    TaskRegistry, SessionId,
    SchedulerTaskConfig, frozenset。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `TaskRegistry`。
    """

    return TaskRegistry(
        session_id=SessionId("session-1"),
        config=SchedulerTaskConfig(frozenset(TaskKind), 3),
    )


def _request(kind: TaskKind, index: int, *, deadline_ms: int = 100) -> TaskRequest:
    """函数契约说明.

    功能: 执行 _request 的同步逻辑,并协调
    TaskRequest, TaskId, SessionId,
    TurnId。
    参数: kind: TaskKind。 必填。 index: int。
    必填。 deadline_ms: int。 可省略。
    契约: 同步调用。 返回 `TaskRequest`。
    """

    return TaskRequest(
        task_id=TaskId(f"task-{index}"),
        session_id=SessionId("session-1"),
        turn_id=TurnId("turn-1"),
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(deadline_ms),
        snapshot_revision=StateRevision(1),
        idempotency_key=IdempotencyKey(f"key-{index}"),
        kind=kind,
        data_snapshot=TaskStateSnapshot.initial(),
    )


def _admit(registry: TaskRegistry, request: TaskRequest) -> None:
    """函数契约说明.

    功能: 执行 _admit 的同步逻辑,并协调 isinstance,
    register。
    参数: registry: TaskRegistry。 必填。
    request: TaskRequest。 必填。
    契约: 同步调用。 返回 `None`。
    """

    assert isinstance(registry.register(request), TaskRegistrationAccepted)
