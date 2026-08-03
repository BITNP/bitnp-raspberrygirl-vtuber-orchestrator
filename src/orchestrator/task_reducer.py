
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import final

from orchestrator.ids import SegmentId, SessionId, TurnId
from orchestrator.sessions import SessionSnapshot, StateRevision
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.task_registry import TaskId, TaskRecord, TaskRegistry, TaskState


@dataclass(frozen=True, slots=True)
class TaskEffect:

    effect_type: str

    payload: str


@dataclass(frozen=True, slots=True)
class TaskResult:

    task_id: TaskId

    session_id: SessionId

    turn_id: TurnId

    snapshot_revision: StateRevision

    effect: TaskEffect

    cancellation_epoch: int = 0

    segment_id: SegmentId | None = None


@unique
class TaskResultRejection(StrEnum):

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

    record: TaskRecord

    effect: TaskEffect


@dataclass(frozen=True, slots=True)
class TaskResultRejected:

    reason: TaskResultRejection


type TaskReductionResult = TaskResultAccepted | TaskResultRejected


@final
class TaskResultReducer:

    def __init__(self, registry: TaskRegistry) -> None:
        self._registry = registry

    def reduce(
        self,
        result: TaskResult,
        *,
        snapshot: SessionSnapshot,
        now_ms: int,
        data_snapshot: TaskStateSnapshot | None = None,
    ) -> TaskReductionResult:
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
    match record.state:
        case TaskState.PENDING | TaskState.RUNNING:
            return None

        case TaskState.CANCELLED:
            return TaskResultRejection.CANCELLED

        case TaskState.SUPERSEDED:
            return TaskResultRejection.SUPERSEDED

        case TaskState.TIMED_OUT:
            return TaskResultRejection.DEADLINE_EXCEEDED

        case TaskState.COMPLETED:
            return TaskResultRejection.ALREADY_COMPLETED

        case TaskState.FAILED:
            return TaskResultRejection.CANCELLED


def _snapshot_rejection(
    record: TaskRecord,
    snapshot: SessionSnapshot,
    data_snapshot: TaskStateSnapshot | None,
) -> TaskResultRejection | None:
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
    if result.session_id != record.request.session_id:
        return TaskResultRejection.SESSION_MISMATCH

    if result.turn_id != record.request.turn_id:
        return TaskResultRejection.TURN_MISMATCH

    if result.snapshot_revision != record.request.snapshot_revision:
        return TaskResultRejection.STALE_REVISION

    if result.cancellation_epoch != record.request.cancellation_epoch:
        return TaskResultRejection.CANCELLED

    if result.segment_id != record.request.segment_id:
        return TaskResultRejection.CANCELLED

    return None
