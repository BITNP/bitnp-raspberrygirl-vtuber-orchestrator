"""Pure task-admission checks evaluated by the session runtime."""

from dataclasses import replace

from orchestrator.sessions import SessionSnapshot
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.task_registry import (
    TaskRegistrationRejection,
    TaskRequest,
)


def with_current_data_snapshot(
    request: TaskRequest, data_snapshot: TaskStateSnapshot
) -> TaskRequest:
    """Supply the runtime data snapshot when callers provide the initial value."""
    if request.data_snapshot == TaskStateSnapshot.initial():
        return replace(request, data_snapshot=data_snapshot)
    return request


def scheduling_rejection(
    request: TaskRequest, snapshot: SessionSnapshot
) -> TaskRegistrationRejection | None:
    """Return the current session-state reason preventing task registration."""
    if request.session_id != snapshot.session_id:
        return TaskRegistrationRejection.SESSION_MISMATCH
    if request.turn_id != snapshot.active_turn_id:
        return TaskRegistrationRejection.ACTIVE_TURN_MISMATCH
    if request.snapshot_revision != snapshot.revision:
        return TaskRegistrationRejection.STALE_SNAPSHOT
    return None
