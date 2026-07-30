from dataclasses import fields

from orchestrator.modes import (
    AdaptiveAgentPolicy,
    AnswerCandidate,
    AudienceInput,
    AudienceSource,
)


def test_select_answer_candidate_returns_none_when_inputs_are_empty() -> None:
    # Given: an adaptive policy without audience input.
    policy = AdaptiveAgentPolicy()

    # When: it selects the next answer candidate.
    candidate = policy.select_answer_candidate(())

    # Then: no candidate is available.
    assert candidate is None


def test_select_answer_candidate_chooses_chronologically_oldest_input() -> None:
    # Given: mixed-source input whose comment predates the ASR input.
    policy = AdaptiveAgentPolicy()
    asr_input = AudienceInput(AudienceSource.ASR, "voice question", 200)
    comment_input = AudienceInput(AudienceSource.COMMENT, "comment question", 100)

    # When: the policy selects the next answer candidate.
    candidate = policy.select_answer_candidate((asr_input, comment_input))

    # Then: chronological order takes precedence over source.
    assert candidate == AnswerCandidate(input=comment_input)


def test_answer_candidate_contains_only_audience_input() -> None:
    # Given: a candidate created for one normalized audience input.
    audience_input = AudienceInput(AudienceSource.ASR, "question", 100)

    # When: the candidate is constructed.
    candidate = AnswerCandidate(input=audience_input)

    # Then: its public structure has no policy-mode metadata.
    assert tuple(field.name for field in fields(candidate)) == ("input",)
