"""Scheduler-owned task lifecycle records with bounded creation authority."""

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
    """Closed scheduler-owned work lanes."""

    REFLEX = "reflex"
    INTERACTIVE = "interactive"
    DELIBERATIVE = "deliberative"
    MAINTENANCE = "maintenance"


@unique
class TaskState(StrEnum):
    """Closed lifecycle states retained for every registered task."""

    PENDING = "pending"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    TIMED_OUT = "timed_out"
    COMPLETED = "completed"


@unique
class TaskRegistrationRejection(StrEnum):
    """Closed reasons scheduler configuration may refuse new work."""

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
    """Bounds scheduler-authorized task kinds, retries, and child fan-out."""

    allowed_kinds: frozenset[TaskKind]
    max_children_per_task: int
    max_retries: int = 0


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """One scheduler-created unit of work tied to its input snapshot."""

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
    """Immutable lifecycle state for one scheduler-authorized task."""

    request: TaskRequest
    state: TaskState
    cancellation_reason: str | None
    superseded_by: TaskId | None


@dataclass(frozen=True, slots=True)
class TaskRegistrationAccepted:
    """A scheduler configuration accepted a new task record."""

    record: TaskRecord


@dataclass(frozen=True, slots=True)
class TaskRegistrationDuplicate:
    """A duplicate delivery maps to the existing idempotent task record."""

    record: TaskRecord


@dataclass(frozen=True, slots=True)
class TaskRegistrationRejected:
    """A scheduler configuration refused a task proposal."""

    reason: TaskRegistrationRejection


type TaskRegistrationResult = (
    TaskRegistrationAccepted | TaskRegistrationDuplicate | TaskRegistrationRejected
)


@final
class TaskRegistry:
    """Own mutable task storage while exposing immutable lifecycle records."""

    def __init__(self, *, session_id: SessionId, config: SchedulerTaskConfig) -> None:
        """Create bounded task storage for one scheduler session."""
        self._session_id = session_id
        self._config = config
        self._records: dict[TaskId, TaskRecord] = {}
        self._idempotency: dict[IdempotencyKey, TaskId] = {}

    @property
    def records(self) -> tuple[TaskRecord, ...]:
        """Return retained task records in registration order."""
        return tuple(self._records.values())

    def task(self, task_id: TaskId) -> TaskRecord | None:
        """Return one immutable task record when it is retained."""
        return self._records.get(task_id)

    def register(self, request: TaskRequest) -> TaskRegistrationResult:
        """Register authorized work or return its idempotent task record."""
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
        """Terminally cancel one pending task before its result can commit."""
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
        """Undo an unqueued pending registration before it becomes observable work."""
        record = self._records.get(task_id)
        if record is None or record.state is not TaskState.PENDING:
            return None
        del self._records[task_id]
        del self._idempotency[record.request.idempotency_key]
        return record

    def cancel_pending(self, *, reason: str) -> tuple[TaskRecord, ...]:
        """Cancel every pending task as one scheduler-owned invalidation action."""
        return tuple(
            record
            for task_id in tuple(self._records)
            if (record := self.cancel(task_id, reason=reason)) is not None
        )

    def supersede(
        self, task_id: TaskId, *, replacement_task_id: TaskId
    ) -> TaskRecord | None:
        """Terminally supersede pending work with registered replacement work."""
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
        """Terminally mark pending work whose scheduler deadline elapsed."""
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
        """Record the sole reducer-approved completion of one pending task."""
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
