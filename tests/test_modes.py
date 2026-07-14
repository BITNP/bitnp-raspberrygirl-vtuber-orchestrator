import pytest

from orchestrator.modes import (
    AudienceInput,
    AudienceSource,
    LecturerState,
    ModePolicy,
    OrchestratorMode,
    QaWindow,
    ScriptStep,
    SlideStep,
    UnknownModeError,
    parse_orchestrator_mode,
)


def test_parse_orchestrator_mode_rejects_unknown_value() -> None:
    # Given: a malformed mode value crossing a future command boundary.
    raw_mode = "karaoke"

    # When: the boundary parser attempts to construct a typed mode.
    with pytest.raises(UnknownModeError) as error:
        _ = parse_orchestrator_mode(raw_mode)

    # Then: the malformed value is rejected before policy selection.
    assert str(error.value) == "unknown orchestrator mode: karaoke"


def test_lecturer_allows_immediate_interruption_when_enabled() -> None:
    # Given: a lecturer session on a known script and slide step.
    policy = ModePolicy.lecturer(
        LecturerState(
            script_step=ScriptStep(2),
            slide_step=SlideStep(5),
            immediate_interruption_enabled=True,
            qa_window=None,
        ),
    )
    audience_input = AudienceInput(
        source=AudienceSource.ASR,
        text="Could you explain this slide again?",
        received_at_ms=1_000,
    )

    # When: the lecturer policy evaluates the audience input.
    candidate = policy.select_answer_candidate((audience_input,))

    # Then: the voice interruption becomes an answer candidate with step context.
    assert candidate is not None
    assert candidate.input == audience_input
    assert candidate.reason == "lecturer_immediate_interruption"
    assert candidate.script_step == ScriptStep(2)
    assert candidate.slide_step == SlideStep(5)


def test_lecturer_denies_immediate_interruption_when_disabled_outside_qa() -> None:
    # Given: a lecturer session outside Q&A with immediate interruption disabled.
    policy = ModePolicy.lecturer(
        LecturerState(
            script_step=ScriptStep(4),
            slide_step=SlideStep(9),
            immediate_interruption_enabled=False,
            qa_window=None,
        ),
    )
    audience_input = AudienceInput(
        source=AudienceSource.ASR,
        text="Can I ask a question now?",
        received_at_ms=2_000,
    )

    # When: the policy evaluates the interruption attempt.
    candidate = policy.select_answer_candidate((audience_input,))

    # Then: no answer candidate is produced during the scripted step.
    assert candidate is None


def test_lecturer_allows_scheduled_qa_window_when_interruption_disabled() -> None:
    # Given: a lecturer session with a scheduled Q&A window open.
    policy = ModePolicy.lecturer(
        LecturerState(
            script_step=ScriptStep(6),
            slide_step=SlideStep(12),
            immediate_interruption_enabled=False,
            qa_window=QaWindow(start_ms=10_000, end_ms=20_000),
        ),
    )
    audience_input = AudienceInput(
        source=AudienceSource.COMMENT,
        text="What is the takeaway from this section?",
        received_at_ms=15_000,
    )

    # When: an audience question arrives during the window.
    candidate = policy.select_answer_candidate((audience_input,))

    # Then: the scheduled Q&A window admits the answer candidate.
    assert candidate is not None
    assert candidate.input == audience_input
    assert candidate.reason == "lecturer_scheduled_qa"
    assert candidate.script_step == ScriptStep(6)
    assert candidate.slide_step == SlideStep(12)


def test_virtual_streamer_prioritizes_comment_answer_candidate() -> None:
    # Given: a virtual streamer session with simultaneous voice and comment input.
    policy = ModePolicy.virtual_streamer(topic="retro games")
    asr_input = AudienceInput(
        source=AudienceSource.ASR,
        text="voice question",
        received_at_ms=100,
    )
    comment_input = AudienceInput(
        source=AudienceSource.COMMENT,
        text="comment question",
        received_at_ms=200,
    )

    # When: the policy chooses the next answer candidate.
    candidate = policy.select_answer_candidate((asr_input, comment_input))

    # Then: live comment input wins over ASR input for streamer mode.
    assert candidate is not None
    assert candidate.input == comment_input
    assert candidate.reason == "virtual_streamer_comment_priority"
    assert candidate.topic == "retro games"


def test_onsite_explainer_prioritizes_asr_answer_candidate() -> None:
    # Given: an onsite explainer session with voice and comment input.
    policy = ModePolicy.onsite_explainer()
    comment_input = AudienceInput(
        source=AudienceSource.COMMENT,
        text="remote comment",
        received_at_ms=100,
    )
    asr_input = AudienceInput(
        source=AudienceSource.ASR,
        text="onsite voice question",
        received_at_ms=200,
    )

    # When: the policy chooses the next answer candidate.
    candidate = policy.select_answer_candidate((comment_input, asr_input))

    # Then: ASR voice input wins for onsite explanation.
    assert candidate is not None
    assert candidate.input == asr_input
    assert candidate.reason == "onsite_explainer_asr_priority"
    assert candidate.mode is OrchestratorMode.ONSITE_EXPLAINER
