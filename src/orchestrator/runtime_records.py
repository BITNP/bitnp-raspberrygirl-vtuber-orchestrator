"""Pure redacted operational-record construction."""

from orchestrator.operational_journal import OperationalRecord
from orchestrator.sessions import EventCorrelation, SessionSnapshot
from orchestrator.task_reducer import TaskResult
from orchestrator.task_registry import TaskId


def interaction_record(
    correlation: EventCorrelation,
    snapshot: SessionSnapshot,
    stage: str,
    accepted: bool,
    task_id: TaskId | None,
) -> OperationalRecord:
    """Create a redacted record for a runtime interaction decision."""
    active_turn_id = snapshot.active_turn_id
    return OperationalRecord(
        stage=stage,
        trace_id=str(correlation.trace_id),
        session_id=str(correlation.session_id),
        turn_id=None if active_turn_id is None else str(active_turn_id),
        segment_id=None,
        task_id=None if task_id is None else str(task_id),
        outcome="accepted" if accepted else "rejected",
    )


def task_result_record(
    result: TaskResult, correlation: EventCorrelation, outcome: str
) -> OperationalRecord:
    """Create a redacted record for a task reduction decision."""
    return OperationalRecord(
        stage="task_result",
        trace_id=str(correlation.trace_id),
        session_id=str(correlation.session_id),
        turn_id=str(result.turn_id),
        segment_id=None,
        task_id=str(result.task_id),
        outcome=outcome,
    )
