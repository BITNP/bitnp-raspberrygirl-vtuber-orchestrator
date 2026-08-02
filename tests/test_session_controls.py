import json

from orchestrator.control_ingress import (
    ContextResetControl,
    MemoryDeleteControl,
    parse_session_control,
)
from orchestrator.ids import SessionId, TraceId
from orchestrator.interactions import CommentProposal
from orchestrator.memory import MemoryKey
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind


def _envelope(event_type: str, data: dict[str, object], sequence: int) -> str:
    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": event_type,
            "event_id": f"event-{sequence}",
            "source": "orchestrator",
            "time": "2026-08-02T00:00:00Z",
            "trace_id": "trace-1",
            "session_id": "session-1",
            "seq": sequence,
            "data": data,
        }
    )


def test_parses_typed_memory_and_context_session_controls() -> None:
    reset = parse_session_control(_envelope("context.reset.command", {}, 1))
    delete = parse_session_control(
        _envelope("memory.delete.command", {"key": "preferred_name"}, 2)
    )

    assert isinstance(reset, ContextResetControl)
    assert isinstance(delete, MemoryDeleteControl)
    assert delete.key == MemoryKey("preferred_name")


def test_runtime_reducer_resets_context_and_deletes_memory_key() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )
    correlation = EventCorrelation(
        TraceId("comment"), SessionId("session-1"), EventSequence(1)
    )
    _ = runtime.receive_comment(CommentProposal("hello", correlation))
    data = runtime.interaction_ingress.data
    assert data.context.snapshot.entries == ()

    reset = parse_session_control(_envelope("context.reset.command", {}, 2))
    assert isinstance(reset, ContextResetControl)
    outcome = runtime.receive_session_control(reset)

    assert outcome.accepted
    assert data.context.snapshot.generation == 1
