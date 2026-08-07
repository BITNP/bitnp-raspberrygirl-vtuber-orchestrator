from orchestrator.execution_envelope import ExecutionEnvelope
from orchestrator.ids import SegmentId, SessionId, TurnId
from orchestrator.sessions import StateRevision


def _envelope() -> ExecutionEnvelope:
    return ExecutionEnvelope(
        session_id=SessionId("session-1"),
        turn_id=TurnId("turn-1"),
        segment_id=SegmentId("agent-turn-1"),
        revision=StateRevision(3),
        cancellation_epoch=7,
        deadline_ms=100,
        allowed_actions=frozenset({"hello"}),
        allowed_expressions=frozenset({"happy"}),
    )


def test_execution_envelope_accepts_only_its_captured_result_fence() -> None:
    envelope = _envelope()

    assert envelope.is_current(
        session_id=SessionId("session-1"),
        revision=StateRevision(3),
        cancellation_epoch=7,
        now_ms=100,
        session_ended=False,
    )
    assert not envelope.is_current(
        session_id=SessionId("session-1"),
        revision=StateRevision(4),
        cancellation_epoch=7,
        now_ms=100,
        session_ended=False,
    )
    assert not envelope.is_current(
        session_id=SessionId("session-1"),
        revision=StateRevision(3),
        cancellation_epoch=7,
        now_ms=101,
        session_ended=False,
    )
