"""模块契约说明.

职责: 提供 orchestrator.task_registry
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass, field, replace
from enum import StrEnum, unique
from typing import NewType, final

from orchestrator.ids import SessionId, TurnId
from orchestrator.sessions import StateRevision
from orchestrator.state_snapshots import TaskStateSnapshot

TaskId = NewType("TaskId", str)

IdempotencyKey = NewType("IdempotencyKey", str)

TaskDeadlineMs = NewType("TaskDeadlineMs", int)


@unique
class TaskKind(StrEnum):
    """类契约说明.

    职责: 定义 TaskKind 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    REFLEX = "reflex"

    INTERACTIVE = "interactive"

    DELIBERATIVE = "deliberative"

    MAINTENANCE = "maintenance"


@unique
class TaskState(StrEnum):
    """类契约说明.

    职责: 定义 TaskState 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    PENDING = "pending"

    CANCELLED = "cancelled"

    SUPERSEDED = "superseded"

    TIMED_OUT = "timed_out"

    COMPLETED = "completed"


@unique
class TaskRegistrationRejection(StrEnum):
    """类契约说明.

    职责: 定义 TaskRegistrationRejection
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    SESSION_MISMATCH = "session_mismatch"

    DUPLICATE_TASK_ID = "duplicate_task_id"

    TASK_KIND_FORBIDDEN = "task_kind_forbidden"

    RETRY_LIMIT = "retry_limit"

    PARENT_NOT_FOUND = "parent_not_found"

    PARENT_TURN_MISMATCH = "parent_turn_mismatch"

    FAN_OUT_LIMIT = "fan_out_limit"

    ACTIVE_TURN_MISMATCH = "active_turn_mismatch"

    STALE_SNAPSHOT = "stale_snapshot"


@dataclass(frozen=True, slots=True)
class SchedulerTaskConfig:
    """类契约说明.

    职责: 保存 SchedulerTaskConfig
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: allowed_kinds、max_children_p
    er_task、max_retries。
    """

    allowed_kinds: frozenset[TaskKind]

    max_children_per_task: int

    max_retries: int = 0


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """类契约说明.

    职责: 保存 TaskRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: task_id、session_id、turn_id、p
    arent_task_id、deadline_ms、snapshot_r
    evision。
    """

    task_id: TaskId

    session_id: SessionId

    turn_id: TurnId

    parent_task_id: TaskId | None

    deadline_ms: TaskDeadlineMs

    snapshot_revision: StateRevision

    idempotency_key: IdempotencyKey

    kind: TaskKind

    retry_attempt: int = 0

    data_snapshot: TaskStateSnapshot = field(default_factory=TaskStateSnapshot.initial)


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """类契约说明.

    职责: 保存 TaskRecord
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: request、state、cancellation_r
    eason、superseded_by。
    """

    request: TaskRequest

    state: TaskState

    cancellation_reason: str | None

    superseded_by: TaskId | None


@dataclass(frozen=True, slots=True)
class TaskRegistrationAccepted:
    """类契约说明.

    职责: 保存 TaskRegistrationAccepted
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: record。
    """

    record: TaskRecord


@dataclass(frozen=True, slots=True)
class TaskRegistrationDuplicate:
    """类契约说明.

    职责: 保存 TaskRegistrationDuplicate
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: record。
    """

    record: TaskRecord


@dataclass(frozen=True, slots=True)
class TaskRegistrationRejected:
    """类契约说明.

    职责: 保存 TaskRegistrationRejected
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。
    """

    reason: TaskRegistrationRejection


type TaskRegistrationResult = (
    TaskRegistrationAccepted | TaskRegistrationDuplicate | TaskRegistrationRejected
)


@final
class TaskRegistry:
    """类契约说明.

    职责: 定义 TaskRegistry 的状态、行为和对外协作边界。
    契约: 方法: __init__、records、task、regist
    er、_registration_rejection、cancel。
    """

    def __init__(self, *, session_id: SessionId, config: SchedulerTaskConfig) -> None:
        """函数契约说明.

        功能: 初始化 TaskRegistry
        的字段并建立实例不变式。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。 config:
        SchedulerTaskConfig。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._session_id = session_id

        self._config = config

        self._records: dict[TaskId, TaskRecord] = {}

        self._idempotency: dict[IdempotencyKey, TaskId] = {}

    @property
    def records(self) -> tuple[TaskRecord, ...]:
        """函数契约说明.

        功能: 执行 records 的同步逻辑,并协调 tuple,
        values。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[TaskRecord,
        ...]`。
        """
        return tuple(self._records.values())

    def task(self, task_id: TaskId) -> TaskRecord | None:
        """函数契约说明.

        功能: 执行 task 的同步逻辑,并协调 get。
        参数: self 表示当前实例。 task_id:
        TaskId。 必填。
        契约: 同步调用。 返回 `TaskRecord |
        None`。
        """
        return self._records.get(task_id)

    def register(self, request: TaskRequest) -> TaskRegistrationResult:
        """函数契约说明.

        功能: 执行 register 的同步逻辑,并协调 get,
        _registration_rejection,
        TaskRecord,
        TaskRegistrationAccepted。
        参数: self 表示当前实例。 request:
        TaskRequest。 必填。
        契约: 同步调用。 返回
        `TaskRegistrationResult`。
        """
        existing_task_id = self._idempotency.get(request.idempotency_key)

        if existing_task_id is not None:
            return TaskRegistrationDuplicate(record=self._records[existing_task_id])

        rejection = self._registration_rejection(request)

        if rejection is not None:
            return TaskRegistrationRejected(rejection)

        record = TaskRecord(request, TaskState.PENDING, None, None)

        self._records[request.task_id] = record

        self._idempotency[request.idempotency_key] = request.task_id

        return TaskRegistrationAccepted(record)

    def _registration_rejection(
        self, request: TaskRequest
    ) -> TaskRegistrationRejection | None:
        """函数契约说明.

        功能: 执行 _registration_rejection
        的同步逻辑,并协调 _parent_rejection。
        参数: self 表示当前实例。 request:
        TaskRequest。 必填。
        契约: 同步调用。 返回
        `TaskRegistrationRejection |
        None`。
        """
        if request.session_id != self._session_id:
            return TaskRegistrationRejection.SESSION_MISMATCH

        if request.task_id in self._records:
            return TaskRegistrationRejection.DUPLICATE_TASK_ID

        if request.kind not in self._config.allowed_kinds:
            return TaskRegistrationRejection.TASK_KIND_FORBIDDEN

        if request.retry_attempt > self._config.max_retries:
            return TaskRegistrationRejection.RETRY_LIMIT

        return self._parent_rejection(request)

    def cancel(self, task_id: TaskId, *, reason: str) -> TaskRecord | None:
        """函数契约说明.

        功能: 执行 cancel 的同步逻辑,并协调 get,
        _store, replace。
        参数: self 表示当前实例。 task_id:
        TaskId。 必填。 reason: str。 必填。
        契约: 同步调用。 返回 `TaskRecord |
        None`。
        """
        record = self._records.get(task_id)

        if record is None or record.state is not TaskState.PENDING:
            return None

        return self._store(
            replace(
                record,
                state=TaskState.CANCELLED,
                cancellation_reason=reason,
                superseded_by=None,
            )
        )

    def withdraw(self, task_id: TaskId) -> TaskRecord | None:
        """函数契约说明.

        功能: 执行 withdraw 的同步逻辑,并协调 get。
        参数: self 表示当前实例。 task_id:
        TaskId。 必填。
        契约: 同步调用。 返回 `TaskRecord |
        None`。
        """
        record = self._records.get(task_id)

        if record is None or record.state is not TaskState.PENDING:
            return None

        del self._records[task_id]

        del self._idempotency[record.request.idempotency_key]

        return record

    def cancel_pending(self, *, reason: str) -> tuple[TaskRecord, ...]:
        """函数契约说明.

        功能: 执行 cancel_pending 的同步逻辑,并协调
        tuple, cancel。
        参数: self 表示当前实例。 reason: str。
        必填。
        契约: 同步调用。 返回 `tuple[TaskRecord,
        ...]`。
        """
        return tuple(
            record
            for task_id in tuple(self._records)
            if (record := self.cancel(task_id, reason=reason)) is not None
        )

    def supersede(
        self, task_id: TaskId, *, replacement_task_id: TaskId
    ) -> TaskRecord | None:
        """函数契约说明.

        功能: 执行 supersede 的同步逻辑,并协调 get,
        _store, replace。
        参数: self 表示当前实例。 task_id:
        TaskId。 必填。 replacement_task_id:
        TaskId。 必填。
        契约: 同步调用。 返回 `TaskRecord |
        None`。
        """
        record = self._records.get(task_id)

        replacement = self._records.get(replacement_task_id)

        if (
            record is None
            or replacement is None
            or record.state is not TaskState.PENDING
            or record.request.turn_id == replacement.request.turn_id
        ):
            return None

        return self._store(
            replace(
                record,
                state=TaskState.SUPERSEDED,
                cancellation_reason=None,
                superseded_by=replacement_task_id,
            )
        )

    def timeout(self, task_id: TaskId) -> TaskRecord | None:
        """函数契约说明.

        功能: 执行 timeout 的同步逻辑,并协调 get,
        _store, replace。
        参数: self 表示当前实例。 task_id:
        TaskId。 必填。
        契约: 同步调用。 返回 `TaskRecord |
        None`。
        """
        record = self._records.get(task_id)

        if record is None or record.state is not TaskState.PENDING:
            return None

        return self._store(
            replace(
                record,
                state=TaskState.TIMED_OUT,
                cancellation_reason=None,
                superseded_by=None,
            )
        )

    def complete(self, task_id: TaskId) -> TaskRecord:
        """函数契约说明.

        功能: 执行 complete 的同步逻辑,并协调
        _store, replace。
        参数: self 表示当前实例。 task_id:
        TaskId。 必填。
        契约: 同步调用。 返回 `TaskRecord`。
        """
        record = self._records[task_id]

        return self._store(
            replace(
                record,
                state=TaskState.COMPLETED,
                cancellation_reason=None,
                superseded_by=None,
            )
        )

    def _parent_rejection(
        self, request: TaskRequest
    ) -> TaskRegistrationRejection | None:
        """函数契约说明.

        功能: 执行 _parent_rejection
        的同步逻辑,并协调 get, sum, values。
        参数: self 表示当前实例。 request:
        TaskRequest。 必填。
        契约: 同步调用。 返回
        `TaskRegistrationRejection |
        None`。
        """
        if request.parent_task_id is None:
            return None

        parent = self._records.get(request.parent_task_id)

        if parent is None:
            return TaskRegistrationRejection.PARENT_NOT_FOUND

        if parent.request.turn_id != request.turn_id:
            return TaskRegistrationRejection.PARENT_TURN_MISMATCH

        child_count = sum(
            record.request.parent_task_id == request.parent_task_id
            for record in self._records.values()
        )

        if child_count >= self._config.max_children_per_task:
            return TaskRegistrationRejection.FAN_OUT_LIMIT

        return None

    def _store(self, record: TaskRecord) -> TaskRecord:
        """函数契约说明.

        功能: 执行 _store 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 record:
        TaskRecord。 必填。
        契约: 同步调用。 返回 `TaskRecord`。
        """
        self._records[record.request.task_id] = record

        return record
