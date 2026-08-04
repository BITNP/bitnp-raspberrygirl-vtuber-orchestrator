from orchestrator.operational_journal import RedactedOperationalRecord
from orchestrator.shadow_replay import ShadowReplayEvidence, audit_shadow_replay


def test_shadow_replay_report_accepts_completed_effect_free_turn() -> None:
    report = audit_shadow_replay(
        ShadowReplayEvidence(
            records=(
                _record(
                    "response_shadow",
                    _shadow_outcome("answer", fallback=True, empty=False),
                ),
            ),
            task_records=(),
            context_revision_before=4,
            context_revision_after=4,
            memory_revision_before=2,
            memory_revision_after=2,
        )
    )

    assert report.accepted
    assert report.shadow_turns == 1
    assert report.text_fallbacks == 1
    assert report.selected_intents == frozenset({"answer"})


def test_shadow_replay_report_rejects_effect_and_context_mutation() -> None:
    report = audit_shadow_replay(
        ShadowReplayEvidence(
            records=(
                _record(
                    "response_shadow",
                    _shadow_outcome("knowledge", fallback=False, empty=True),
                ),
                _record("response_compiled", "ignored"),
            ),
            task_records=(),
            context_revision_before=4,
            context_revision_after=5,
            memory_revision_before=2,
            memory_revision_after=2,
        )
    )

    assert not report.accepted
    assert report.violations == (
        "shadow_context_mutated",
        "shadow_effect_stage_emitted",
    )


def _record(stage: str, outcome: str) -> RedactedOperationalRecord:
    return RedactedOperationalRecord(
        stage=stage,
        trace_id="trace",
        session_id="session",
        turn_id="turn",
        segment_id="segment",
        task_id="task",
        outcome=outcome,
    )


def _shadow_outcome(intent: str, *, fallback: bool, empty: bool) -> str:
    return (
        f"intent={intent};fallback={fallback};cues=0;rejected_cues=0;"
        f"empty={empty};phase=completed"
    )
