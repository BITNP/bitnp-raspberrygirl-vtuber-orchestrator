"""模块契约说明.

职责: 提供 orchestrator.task_reducer
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import final

from orchestrator.ids import SessionId, TurnId
from orchestrator.sessions import SessionSnapshot, StateRevision
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.task_registry import TaskId, TaskRecord, TaskRegistry, TaskState


@dataclass(frozen=True, slots=True)
class TaskEffect:
    """类契约说明.

    职责: 保存 TaskEffect
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: effect_type、payload。
    """

    effect_type: str

    payload: str


@dataclass(frozen=True, slots=True)
class TaskResult:
    """类契约说明.

    职责: 保存 TaskResult
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: task_id、session_id、turn_id、s
    napshot_revision、effect。
    """

    task_id: TaskId

    session_id: SessionId

    turn_id: TurnId

    snapshot_revision: StateRevision

    effect: TaskEffect


@unique
class TaskResultRejection(StrEnum):
    """类契约说明.

    职责: 定义 TaskResultRejection
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    TASK_NOT_FOUND = "task_not_found"

    CANCELLED = "cancelled"

    SUPERSEDED = "superseded"

    DEADLINE_EXCEEDED = "deadline_exceeded"

    ALREADY_COMPLETED = "already_completed"

    SESSION_MISMATCH = "session_mismatch"

    TURN_MISMATCH = "turn_mismatch"

    STALE_REVISION = "stale_revision"

    STALE_DATA_SNAPSHOT = "stale_data_snapshot"


@dataclass(frozen=True, slots=True)
class TaskResultAccepted:
    """类契约说明.

    职责: 保存 TaskResultAccepted
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: record、effect。
    """

    record: TaskRecord

    effect: TaskEffect


@dataclass(frozen=True, slots=True)
class TaskResultRejected:
    """类契约说明.

    职责: 保存 TaskResultRejected
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。
    """

    reason: TaskResultRejection


type TaskReductionResult = TaskResultAccepted | TaskResultRejected


@final
class TaskResultReducer:
    """类契约说明.

    职责: 定义 TaskResultReducer
    的状态、行为和对外协作边界。
    契约: 方法: __init__、reduce。
    """

    def __init__(self, registry: TaskRegistry) -> None:
        """函数契约说明.

        功能: 初始化 TaskResultReducer
        的字段并建立实例不变式。
        参数: self 表示当前实例。 registry:
        TaskRegistry。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._registry = registry

    def reduce(
        self,
        result: TaskResult,
        *,
        snapshot: SessionSnapshot,
        now_ms: int,
        data_snapshot: TaskStateSnapshot | None = None,
    ) -> TaskReductionResult:
        """函数契约说明.

        功能: 执行 reduce 的同步逻辑,并协调 task,
        _lifecycle_rejection,
        _snapshot_rejection,
        _result_rejection。
        参数: self 表示当前实例。 result:
        TaskResult。 必填。 snapshot:
        SessionSnapshot。 必填。 now_ms:
        int。 必填。 data_snapshot:
        TaskStateSnapshot | None。 可省略。
        契约: 同步调用。 返回
        `TaskReductionResult`。
        """
        record = self._registry.task(result.task_id)

        if record is None:
            return TaskResultRejected(TaskResultRejection.TASK_NOT_FOUND)

        rejection = _lifecycle_rejection(record)

        if rejection is not None:
            return TaskResultRejected(rejection)

        rejection = _snapshot_rejection(record, snapshot, data_snapshot)

        if rejection is not None:
            return TaskResultRejected(rejection)

        if now_ms > record.request.deadline_ms:
            _ = self._registry.timeout(result.task_id)

            return TaskResultRejected(TaskResultRejection.DEADLINE_EXCEEDED)

        rejection = _result_rejection(record, result)

        if rejection is not None:
            return TaskResultRejected(rejection)

        return TaskResultAccepted(
            self._registry.complete(result.task_id),
            result.effect,
        )


def _lifecycle_rejection(record: TaskRecord) -> TaskResultRejection | None:
    """函数契约说明.

    功能: 执行 _lifecycle_rejection
    的同步逻辑,并维持签名契约。
    参数: record: TaskRecord。 必填。
    契约: 同步调用。 返回 `TaskResultRejection |
    None`。
    """
    match record.state:
        case TaskState.PENDING:
            return None

        case TaskState.CANCELLED:
            return TaskResultRejection.CANCELLED

        case TaskState.SUPERSEDED:
            return TaskResultRejection.SUPERSEDED

        case TaskState.TIMED_OUT:
            return TaskResultRejection.DEADLINE_EXCEEDED

        case TaskState.COMPLETED:
            return TaskResultRejection.ALREADY_COMPLETED


def _snapshot_rejection(
    record: TaskRecord,
    snapshot: SessionSnapshot,
    data_snapshot: TaskStateSnapshot | None,
) -> TaskResultRejection | None:
    """函数契约说明.

    功能: 执行 _snapshot_rejection 的同步逻辑,并协调
    initial。
    参数: record: TaskRecord。 必填。
    snapshot: SessionSnapshot。 必填。
    data_snapshot: TaskStateSnapshot |
    None。 必填。
    契约: 同步调用。 返回 `TaskResultRejection |
    None`。
    """
    if snapshot.session_id != record.request.session_id:
        return TaskResultRejection.SESSION_MISMATCH

    if snapshot.revision != record.request.snapshot_revision:
        return TaskResultRejection.STALE_REVISION

    if snapshot.active_turn_id != record.request.turn_id:
        return TaskResultRejection.TURN_MISMATCH

    current_data_snapshot = data_snapshot or TaskStateSnapshot.initial()

    if record.request.data_snapshot != current_data_snapshot:
        return TaskResultRejection.STALE_DATA_SNAPSHOT

    return None


def _result_rejection(
    record: TaskRecord, result: TaskResult
) -> TaskResultRejection | None:
    """函数契约说明.

    功能: 执行 _result_rejection
    的同步逻辑,并维持签名契约。
    参数: record: TaskRecord。 必填。 result:
    TaskResult。 必填。
    契约: 同步调用。 返回 `TaskResultRejection |
    None`。
    """
    if result.session_id != record.request.session_id:
        return TaskResultRejection.SESSION_MISMATCH

    if result.turn_id != record.request.turn_id:
        return TaskResultRejection.TURN_MISMATCH

    if result.snapshot_revision != record.request.snapshot_revision:
        return TaskResultRejection.STALE_REVISION

    return None
