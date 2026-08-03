"""Reducer-side admission guard for asynchronous provider results.

Provider work may finish after cancellation.  This object makes the required
identity checks explicit before a result can become an effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from orchestrator.task_registry import TaskId, TaskRecord, TaskState

if TYPE_CHECKING:
    from orchestrator.ids import SessionId, TurnId
    from orchestrator.sessions import StateRevision


@dataclass(frozen=True, slots=True)
class TaskResultFence:
    session_id: SessionId
    turn_id: TurnId
    revision: StateRevision
    cancellation_epoch: int
    now_ms: int


def result_is_current(record: TaskRecord | None, fence: TaskResultFence) -> bool:
    """Return true only for a running, in-deadline task owned by this snapshot."""
    if record is None or record.state is not TaskState.RUNNING:
        return False
    request = record.request
    return (
        request.session_id == fence.session_id
        and request.turn_id == fence.turn_id
        and request.snapshot_revision == fence.revision
        and int(request.deadline_ms) >= fence.now_ms
    )


@dataclass(frozen=True, slots=True)
class TaskHandle:
    """Cancellation identity carried by async provider adapters."""

    task_id: TaskId
    session_id: SessionId
    turn_id: TurnId
    revision: StateRevision
    cancellation_epoch: int
