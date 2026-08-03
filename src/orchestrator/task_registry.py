
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

    REFLEX = "reflex"

    INTERACTIVE = "interactive"

    DELIBERATIVE = "deliberative"

    MAINTENANCE = "maintenance"


@unique
class TaskState(StrEnum):

    PENDING = "pending"

    RUNNING = "running"

    CANCELLED = "cancelled"

    SUPERSEDED = "superseded"

    TIMED_OUT = "timed_out"

    COMPLETED = "completed"

    FAILED = "failed"


@unique
class TaskRegistrationRejection(StrEnum):

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

    allowed_kinds: frozenset[TaskKind]

    max_children_per_task: int

    max_retries: int = 0


@dataclass(frozen=True, slots=True)
class TaskRequest:

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

    request: TaskRequest

    state: TaskState

    cancellation_reason: str | None

    superseded_by: TaskId | None


@dataclass(frozen=True, slots=True)
class TaskRegistrationAccepted:

    record: TaskRecord


@dataclass(frozen=True, slots=True)
class TaskRegistrationDuplicate:

    record: TaskRecord


@dataclass(frozen=True, slots=True)
class TaskRegistrationRejected:

    reason: TaskRegistrationRejection


type TaskRegistrationResult = (
    TaskRegistrationAccepted | TaskRegistrationDuplicate | TaskRegistrationRejected
)


@final
class TaskRegistry:

    def __init__(self, *, session_id: SessionId, config: SchedulerTaskConfig) -> None:
        self._session_id = session_id

        self._config = config

        self._records: dict[TaskId, TaskRecord] = {}

        self._idempotency: dict[IdempotencyKey, TaskId] = {}

    @property
    def records(self) -> tuple[TaskRecord, ...]:
        return tuple(self._records.values())

    def task(self, task_id: TaskId) -> TaskRecord | None:
        return self._records.get(task_id)

    def register(self, request: TaskRequest) -> TaskRegistrationResult:
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
        record = self._records.get(task_id)

        if record is None or record.state not in {TaskState.PENDING, TaskState.RUNNING}:
            return None

        return self._store(
            replace(
                record,
                state=TaskState.CANCELLED,
                cancellation_reason=reason,
                superseded_by=None,
            )
        )

    def claim(self, task_id: TaskId) -> TaskRecord | None:
        """Atomically transfer an admitted task from a queue to a worker."""
        record = self._records.get(task_id)
        if record is None or record.state is not TaskState.PENDING:
            return None
        return self._store(
            replace(
                record,
                state=TaskState.RUNNING,
                cancellation_reason=None,
                superseded_by=None,
            )
        )

    def withdraw(self, task_id: TaskId) -> TaskRecord | None:
        record = self._records.get(task_id)

        if record is None or record.state is not TaskState.PENDING:
            return None

        del self._records[task_id]

        del self._idempotency[record.request.idempotency_key]

        return record

    def cancel_pending(self, *, reason: str) -> tuple[TaskRecord, ...]:
        return tuple(
            record
            for task_id in tuple(self._records)
            if (record := self.cancel(task_id, reason=reason)) is not None
        )

    def supersede(
        self, task_id: TaskId, *, replacement_task_id: TaskId
    ) -> TaskRecord | None:
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
        record = self._records.get(task_id)

        if record is None or record.state not in {TaskState.PENDING, TaskState.RUNNING}:
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
        record = self._records[task_id]

        return self._store(
            replace(
                record,
                state=TaskState.COMPLETED,
                cancellation_reason=None,
                superseded_by=None,
            )
        )

    def fail(self, task_id: TaskId, *, reason: str) -> TaskRecord | None:
        """Mark a running task terminal without admitting a later result."""
        record = self._records.get(task_id)
        if record is None or record.state is not TaskState.RUNNING:
            return None
        return self._store(
            replace(
                record,
                state=TaskState.FAILED,
                cancellation_reason=reason,
                superseded_by=None,
            )
        )

    def _parent_rejection(
        self, request: TaskRequest
    ) -> TaskRegistrationRejection | None:
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
        self._records[record.request.task_id] = record

        return record
