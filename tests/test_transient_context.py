import pytest

from orchestrator.ids import SegmentId, SessionId, TurnId
from orchestrator.llm import LLMFinal, LLMPrompt, LLMRequest, MockLLMAdapter
from orchestrator.transient_context import (
    AcceptedOutput,
    CancelledMaterial,
    ContextProvenance,
    ContextSequence,
    ContextSessionMismatchError,
    ContextSourceId,
    FinalizedInput,
    ModelContextBudget,
    ModelId,
    PartialMaterial,
    StaleMaterial,
    StaticContextBudgetPolicy,
    TokenBudget,
    TransientContext,
)


def test_llm_final_is_the_complete_output_context_characterization() -> None:
    # Given: the existing adapter's stream lifecycle.

    request = LLMRequest(prompt=LLMPrompt(system="system", user="user"))

    adapter = MockLLMAdapter(answer_chunks=("first", "second"))

    # When: the stream completes normally.

    events = tuple(adapter.stream(request))

    # Then: completion is represented by one final event after deltas.

    assert events[-1] == LLMFinal(text="firstsecond", used_fallback=False)


def test_context_admits_only_finalized_inputs_and_accepted_outputs() -> None:
    # Given: final, partial, cancelled, stale, and accepted turn material.

    context = TransientContext(session_id=SessionId("session-1"))

    partial = PartialMaterial(_provenance("partial", 1), "partial")

    cancelled = CancelledMaterial(_provenance("cancelled", 2), "cancelled")

    stale = StaleMaterial(_provenance("stale", 3), "stale")

    finalized = FinalizedInput(_provenance("input", 4), "question")

    accepted = AcceptedOutput(_provenance("output", 5), "answer")

    # When: the context considers every lifecycle outcome in sequence.

    _ = context.consider(partial)

    _ = context.consider(cancelled)

    _ = context.consider(stale)

    _ = context.consider(finalized)

    _ = context.consider(accepted)

    # Then: only finalized input and accepted output become context entries.

    assert tuple(entry.text for entry in context.snapshot.entries) == (
        "question",
        "answer",
    )

    assert tuple(entry.provenance.source_id for entry in context.snapshot.entries) == (
        ContextSourceId("input"),
        ContextSourceId("output"),
    )

    assert context.snapshot.generation == 2


def test_compaction_is_deterministic_and_preserves_all_source_identities() -> None:
    # Given: one reproducible event sequence whose text exceeds the model budget.

    policy = StaticContextBudgetPolicy(
        model_id=ModelId("local-model"),
        budget=ModelContextBudget(input_tokens=TokenBudget(3)),
    )

    first = FinalizedInput(_provenance("input-1", 1), "old question")

    second = AcceptedOutput(_provenance("output-1", 2), "old answer")

    third = FinalizedInput(_provenance("input-2", 3), "new question")

    first_context = TransientContext(session_id=SessionId("session-1"))

    second_context = TransientContext(session_id=SessionId("session-1"))

    for context in (first_context, second_context):
        _ = context.consider(first)

        _ = context.consider(second)

        _ = context.consider(third)

    # When: each matching snapshot composes context for the same model.

    first_composition = first_context.compose(ModelId("local-model"), policy)

    second_composition = second_context.compose(ModelId("local-model"), policy)

    # Then: byte-stable output retains raw newest material and digest provenance.

    assert first_composition == second_composition

    assert first_composition.content_token_count == TokenBudget(3)

    assert tuple(entry.provenance.source_id for entry in first_composition.entries) == (
        ContextSourceId("input-2"),
    )

    assert first_composition.digests[0].source_provenances == (
        first.provenance,
        second.provenance,
    )


def test_compaction_digests_only_the_oversized_entry_between_retained_entries() -> None:
    # Given: a budget-three sequence with a middle entry too large to retain.

    policy = StaticContextBudgetPolicy(
        model_id=ModelId("local-model"),
        budget=ModelContextBudget(input_tokens=TokenBudget(3)),
    )

    old_fit = FinalizedInput(_provenance("old-fit", 1), "one")

    oversized = FinalizedInput(
        _provenance("oversized", 2),
        "two three four five",
    )

    new_fit = FinalizedInput(_provenance("new-fit", 3), "six")

    context = TransientContext(session_id=SessionId("session-1"))

    _ = context.consider(old_fit)

    _ = context.consider(oversized)

    _ = context.consider(new_fit)

    # When: deterministic composition retains both fitting raw entries.

    first_composition = context.compose(ModelId("local-model"), policy)

    second_composition = context.compose(ModelId("local-model"), policy)

    # Then: only the omitted oversized source is digested, without duplication.

    assert first_composition == second_composition

    assert tuple(entry.provenance.source_id for entry in first_composition.entries) == (
        ContextSourceId("old-fit"),
        ContextSourceId("new-fit"),
    )

    assert first_composition.digests[0].source_provenances == (oversized.provenance,)


def test_reset_clears_one_session_without_reusing_another_sessions_context() -> None:
    # Given: one populated session and a contribution from another session.

    context = TransientContext(session_id=SessionId("session-1"))

    _ = context.consider(FinalizedInput(_provenance("input-1", 1), "question"))

    # When: the owner resets and accepts a resumed turn contribution.

    reset_snapshot = context.reset()

    _ = context.consider(FinalizedInput(_provenance("input-2", 2), "resumed"))

    # Then: reset is observable, prior entries stay absent, and reuse fails.

    assert reset_snapshot.generation == 2

    assert reset_snapshot.entries == ()

    assert tuple(entry.text for entry in context.snapshot.entries) == ("resumed",)

    foreign = FinalizedInput(_provenance_for("session-2", "input-2", 2), "foreign")

    with pytest.raises(ContextSessionMismatchError) as raised:
        _ = context.consider(foreign)

    assert raised.value.actual_session_id == SessionId("session-2")


def _provenance(source_id: str, sequence: int) -> ContextProvenance:

    return _provenance_for("session-1", source_id, sequence)


def _provenance_for(
    session_id: str,
    source_id: str,
    sequence: int,
) -> ContextProvenance:

    return ContextProvenance(
        session_id=SessionId(session_id),
        turn_id=TurnId(f"turn-{sequence}"),
        segment_id=SegmentId(f"segment-{sequence}"),
        sequence=ContextSequence(sequence),
        source_id=ContextSourceId(source_id),
    )
