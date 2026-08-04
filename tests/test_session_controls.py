import json
from pathlib import Path

import pytest

from orchestrator.control_ingress import (
    ContextResetControl,
    MemoryDeleteControl,
    SessionEndControl,
    parse_session_control,
)
from orchestrator.ids import SessionId, TraceId
from orchestrator.interaction_ingress import session_storage_root
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


def test_parses_empty_session_end_control_only() -> None:
    end = parse_session_control(_envelope("session.end.command", {}, 3))

    assert isinstance(end, SessionEndControl)
    assert parse_session_control(_envelope("session.end.command", {"x": 1}, 3)) is None


def test_session_end_cancels_work_erases_state_and_rejects_later_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )
    root = session_storage_root(SessionId("session-1"))
    root.mkdir(parents=True, exist_ok=True)
    _ = (root / "memory.md").write_text("session state", encoding="utf-8")
    end = parse_session_control(_envelope("session.end.command", {}, 2))
    assert isinstance(end, SessionEndControl)

    outcome = runtime.receive_session_control(end)

    assert outcome.accepted
    assert root.exists() is False
    assert runtime.interaction_ingress.data.context.snapshot.entries == ()
    later = runtime.receive_comment(
        CommentProposal(
            "later",
            EventCorrelation(
                TraceId("later"), SessionId("session-1"), EventSequence(3)
            ),
        )
    )
    assert later.accepted is False
