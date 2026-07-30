"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import replace
from typing import Literal

import pytest

from orchestrator.ids import SessionId, TurnId
from orchestrator.sessions import SessionSnapshot, StateRevision
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
    TaskRecord,
    TaskRegistrationAccepted,
    TaskRegistrationDuplicate,
    TaskRegistrationRejected,
    TaskRegistrationRejection,
    TaskRegistrationResult,
    TaskRegistry,
    TaskRequest,
    TaskState,
)


def test_registry_returns_existing_task_for_duplicate_idempotency_key() -> None:
    # Given: a scheduler-owned registry with one allowed interactive task kind.

    """函数契约说明.

    功能: 验证 registry returns existing
    task for duplicate idempotency key
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = _registry()

    request = _request(task_id="task-1", key="answer-1")

    # When: scheduler configuration receives the same idempotency key twice.

    first = registry.register(request)

    duplicate = registry.register(_request(task_id="task-2", key="answer-1"))

    # Then: exactly one lifecycle record is retained for the work.

    record = _accepted_record(first)

    assert record.request == request

    assert registry.records == (record,)

    match duplicate:
        case TaskRegistrationDuplicate(record=duplicate_record):
            assert duplicate_record == record

        case TaskRegistrationAccepted():
            pytest.fail("duplicate idempotency key created another task")

        case TaskRegistrationRejected():
            pytest.fail("duplicate idempotency key was rejected")


@pytest.mark.parametrize(
    ("prepare", "expected_reason"),
    [
        ("stale", TaskResultRejection.STALE_REVISION),
        ("cancelled", TaskResultRejection.CANCELLED),
        ("superseded", TaskResultRejection.SUPERSEDED),
    ],
)
def test_reducer_rejects_result_when_task_is_no_longer_current(
    prepare: Literal["stale", "cancelled", "superseded"],
    expected_reason: TaskResultRejection,
) -> None:
    # Given: accepted scheduler work whose state becomes stale or terminal.

    """函数契约说明.

    功能: 验证 reducer rejects result when
    task is no longer current
    的回归场景和可观察结果。
    参数: prepare: Literal['stale',
    'cancelled', 'superseded']。 必填。
    expected_reason:
    TaskResultRejection。 必填。
    契约: 同步调用。 返回 `None`。
    """

    registry = _registry()

    request = _request(task_id="task-1", key="answer-1")

    _register(registry, request)

    reducer = TaskResultReducer(registry)

    match prepare:
        case "cancelled":
            _ = registry.cancel(TaskId("task-1"), reason="user_interrupt")

            result = _result(snapshot_revision=StateRevision(7))

        case "superseded":
            replacement = replace(
                _request(task_id="task-2", key="answer-2"),
                turn_id=TurnId("turn-2"),
            )

            _register(registry, replacement)

            _ = registry.supersede(
                TaskId("task-1"), replacement_task_id=TaskId("task-2")
            )

            result = _result(snapshot_revision=StateRevision(7))

        case "stale":
            result = _result(snapshot_revision=StateRevision(8))

    # When: a physical worker completes after the task is invalidated.

    outcome = reducer.reduce(result, snapshot=_snapshot(), now_ms=100)

    # Then: the reducer rejects the proposed effect and never commits it.

    match outcome:
        case TaskResultRejected(reason=reason):
            assert reason is expected_reason

        case TaskResultAccepted():
            pytest.fail("invalid task result committed an effect")


def test_reducer_commits_current_result_once_and_rejects_duplicate_delivery() -> None:
    # Given: one active task and a matching worker result.

    """函数契约说明.

    功能: 验证 reducer commits current
    result once and rejects duplicate
    delivery 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = _registry()

    _register(registry, _request(task_id="task-1", key="answer-1"))

    reducer = TaskResultReducer(registry)

    result = _result(snapshot_revision=StateRevision(7))

    # When: the same completed result is delivered twice.

    accepted = reducer.reduce(result, snapshot=_snapshot(), now_ms=100)

    duplicate = reducer.reduce(result, snapshot=_snapshot(), now_ms=100)

    # Then: only the first reducer call is eligible to commit the effect.

    match accepted:
        case TaskResultAccepted(effect=effect):
            assert effect == TaskEffect(effect_type="answer", payload="accepted")

        case TaskResultRejected():
            pytest.fail("current task result was rejected")

    match duplicate:
        case TaskResultRejected(reason=TaskResultRejection.ALREADY_COMPLETED):
            pass

        case TaskResultAccepted():
            pytest.fail("duplicate delivery committed an effect")

        case TaskResultRejected():
            pytest.fail("duplicate delivery was rejected for the wrong reason")


def test_reducer_marks_hung_task_timed_out_before_late_result_can_commit() -> None:
    # Given: a task whose deadline has already passed while its worker is hung.

    """函数契约说明.

    功能: 验证 reducer marks hung task timed
    out before late result can commit
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = _registry()

    _register(
        registry,
        replace(_request(task_id="task-1", key="answer-1"), deadline_ms=100),
    )

    reducer = TaskResultReducer(registry)

    # When: the late worker eventually delivers its result.

    outcome = reducer.reduce(
        _result(snapshot_revision=StateRevision(7)), snapshot=_snapshot(), now_ms=101
    )

    # Then: the deadline lifecycle state decisively blocks its effect.

    match outcome:
        case TaskResultRejected(reason=TaskResultRejection.DEADLINE_EXCEEDED):
            pass

        case TaskResultAccepted():
            pytest.fail("timed-out task result committed an effect")

        case TaskResultRejected():
            pytest.fail("timed-out task was rejected for the wrong reason")

    record = registry.task(TaskId("task-1"))

    assert record is not None

    assert record.state is TaskState.TIMED_OUT


@pytest.mark.parametrize(
    ("scenario", "expected_reason"),
    [
        ("unknown", TaskResultRejection.TASK_NOT_FOUND),
        ("session", TaskResultRejection.SESSION_MISMATCH),
        ("turn", TaskResultRejection.TURN_MISMATCH),
    ],
)
def test_reducer_rejects_unknown_or_mismatched_result_identity(
    scenario: Literal["unknown", "session", "turn"],
    expected_reason: TaskResultRejection,
) -> None:
    # Given: a reducer with no matching task or result identity.

    registry = _registry()

    result = _result(snapshot_revision=StateRevision(7))

    if scenario != "unknown":
        _register(registry, _request(task_id="task-1", key="answer-1"))

    match scenario:
        case "unknown":
            result = replace(result, task_id=TaskId("missing"))

        case "session":
            result = replace(result, session_id=SessionId("other"))

        case "turn":
            result = replace(result, turn_id=TurnId("other"))

    # When: the physical result reaches the reducer.

    outcome = TaskResultReducer(registry).reduce(
        result,
        snapshot=_snapshot(),
        now_ms=100,
    )

    # Then: it is refused by its explicit correlation boundary.

    assert outcome == TaskResultRejected(expected_reason)


def test_registry_enforces_scheduler_configured_task_kind_and_child_fanout() -> None:
    # Given: a registry that permits one interactive child per parent task.

    """函数契约说明.

    功能: 验证 registry enforces scheduler
    configured task kind and child
    fanout 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = _registry()

    _register(registry, _request(task_id="parent", key="parent"))

    _register(
        registry,
        replace(
            _request(task_id="child-1", key="child-1"),
            parent_task_id=TaskId("parent"),
        ),
    )

    # When: new work exceeds the configured graph bounds or task-kind authority.

    fan_out = registry.register(
        replace(
            _request(task_id="child-2", key="child-2"),
            parent_task_id=TaskId("parent"),
        )
    )

    forbidden = registry.register(
        replace(_request(task_id="deep", key="deep"), kind=TaskKind.DELIBERATIVE)
    )

    # Then: only scheduler-configured topology and lanes can become task records.

    match fan_out:
        case TaskRegistrationRejected(reason=TaskRegistrationRejection.FAN_OUT_LIMIT):
            pass

        case TaskRegistrationAccepted() | TaskRegistrationDuplicate():
            pytest.fail("fan-out limit allowed another child task")

        case TaskRegistrationRejected():
            pytest.fail("fan-out task was rejected for the wrong reason")

    match forbidden:
        case TaskRegistrationRejected(
            reason=TaskRegistrationRejection.TASK_KIND_FORBIDDEN
        ):
            pass

        case TaskRegistrationAccepted() | TaskRegistrationDuplicate():
            pytest.fail("unconfigured task kind was accepted")

        case TaskRegistrationRejected():
            pytest.fail("task kind was rejected for the wrong reason")


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
        config=SchedulerTaskConfig(
            allowed_kinds=frozenset({TaskKind.INTERACTIVE}),
            max_children_per_task=1,
        ),
    )


def _request(
    *,
    task_id: str,
    key: str,
) -> TaskRequest:
    """函数契约说明.

    功能: 执行 _request 的同步逻辑,并协调
    TaskRequest, TaskId, SessionId,
    TurnId。
    参数: task_id: str。 必填。 key: str。 必填。
    契约: 同步调用。 返回 `TaskRequest`。
    """

    return TaskRequest(
        task_id=TaskId(task_id),
        session_id=SessionId("session-1"),
        turn_id=TurnId("turn-1"),
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(200),
        snapshot_revision=StateRevision(7),
        idempotency_key=IdempotencyKey(key),
        kind=TaskKind.INTERACTIVE,
    )


def _result(*, snapshot_revision: StateRevision) -> TaskResult:
    """函数契约说明.

    功能: 执行 _result 的同步逻辑,并协调 TaskResult,
    TaskId, SessionId, TurnId。
    参数: snapshot_revision:
    StateRevision。 必填。
    契约: 同步调用。 返回 `TaskResult`。
    """

    return TaskResult(
        task_id=TaskId("task-1"),
        session_id=SessionId("session-1"),
        turn_id=TurnId("turn-1"),
        snapshot_revision=snapshot_revision,
        effect=TaskEffect(effect_type="answer", payload="accepted"),
    )


def _snapshot() -> SessionSnapshot:
    """函数契约说明.

    功能: 执行 _snapshot 的同步逻辑,并协调
    SessionSnapshot, SessionId,
    StateRevision, TurnId。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `SessionSnapshot`。
    """

    return SessionSnapshot(
        session_id=SessionId("session-1"),
        revision=StateRevision(7),
        active_turn_id=TurnId("turn-1"),
    )


def _register(registry: TaskRegistry, request: TaskRequest) -> None:
    """函数契约说明.

    功能: 执行 _register 的同步逻辑,并协调 register,
    fail。
    参数: registry: TaskRegistry。 必填。
    request: TaskRequest。 必填。
    契约: 同步调用。 返回 `None`。
    """

    match registry.register(request):
        case TaskRegistrationAccepted():
            pass

        case TaskRegistrationDuplicate():
            pytest.fail("test setup reused an idempotency key")

        case TaskRegistrationRejected():
            pytest.fail("test setup task was rejected")


def _accepted_record(result: TaskRegistrationResult) -> TaskRecord:
    """函数契约说明.

    功能: 执行 _accepted_record 的同步逻辑,并协调
    fail。
    参数: result: TaskRegistrationResult。
    必填。
    契约: 同步调用。 返回 `TaskRecord`。
    """

    match result:
        case TaskRegistrationAccepted(record=record):
            return record

        case TaskRegistrationDuplicate():
            pytest.fail("first task registration was treated as duplicate")

        case TaskRegistrationRejected():
            pytest.fail("first task registration was rejected")
