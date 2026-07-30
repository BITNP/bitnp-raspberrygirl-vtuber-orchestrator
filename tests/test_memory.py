import pytest

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.memory import (
    MemoryCategory,
    MemoryCommitAccepted,
    MemoryCommitRejected,
    MemoryCommitRejection,
    MemoryConfidence,
    MemoryKey,
    MemoryPolicy,
    MemoryProposal,
    MemoryProvenance,
    MemorySource,
    MutableMemory,
    ProposalRevision,
)
from orchestrator.state_snapshots import ConsentRevision, ProfileRevision


def test_ordinary_supported_preference_auto_commits_with_provenance() -> None:
    # Given: a scheduler-owned store and a confident, typed agent proposal.
    memory = MutableMemory(session_id=SessionId("session-1"), policy=MemoryPolicy())
    proposal = _proposal(value="小莓", confidence=95)

    # When: the policy reducer receives the proposal against its current revision.
    outcome = memory.reduce(proposal)

    # Then: the preference is revisioned and retains its non-conversation provenance.
    match outcome:
        case MemoryCommitAccepted(snapshot=snapshot):
            assert snapshot.revision == ProposalRevision(1)
            assert snapshot.entries[0].value == "小莓"
            assert snapshot.entries[0].provenance.source is MemorySource.AGENT_PROPOSAL
        case MemoryCommitRejected():
            pytest.fail("ordinary supported preference was not auto-committed")


@pytest.mark.parametrize(
    ("value", "confidence", "category", "reason"),
    [
        (
            "小莓",
            94,
            MemoryCategory.ORDINARY_PREFERENCE,
            MemoryCommitRejection.STALE_PROPOSAL,
        ),
        (
            "模板",
            95,
            MemoryCategory.RESTRICTED,
            MemoryCommitRejection.RESTRICTED_CATEGORY,
        ),
        (
            "小莓",
            40,
            MemoryCategory.ORDINARY_PREFERENCE,
            MemoryCommitRejection.UNSUPPORTED_ASSERTION,
        ),
    ],
)
def test_memory_policy_rejects_stale_restricted_and_unsupported_proposals(
    value: str,
    confidence: int,
    category: MemoryCategory,
    reason: MemoryCommitRejection,
) -> None:
    # Given: a store and one policy-relevant proposal type.
    memory = MutableMemory(session_id=SessionId("session-1"), policy=MemoryPolicy())
    match reason:
        case MemoryCommitRejection.STALE_PROPOSAL:
            _ = memory.reduce(_proposal(value="先前称呼", confidence=95))
        case (
            MemoryCommitRejection.RESTRICTED_CATEGORY
            | MemoryCommitRejection.UNSUPPORTED_ASSERTION
        ):
            pass
        case unexpected:
            pytest.fail(f"unexpected rejection case: {unexpected}")

    # When: a stale, restricted, or unsupported proposal reaches the policy reducer.
    outcome = memory.reduce(
        _proposal(value=value, confidence=confidence, category=category)
    )

    # Then: no untrusted proposal changes durable memory.
    match outcome:
        case MemoryCommitRejected(reason=actual):
            assert actual == reason
        case MemoryCommitAccepted():
            pytest.fail("rejected memory proposal changed durable state")


def test_profile_or_consent_revision_invalidates_task_snapshot() -> None:
    # Given: task work captured with the current memory/profile/consent revisions.
    memory = MutableMemory(session_id=SessionId("session-1"), policy=MemoryPolicy())
    captured = memory.task_snapshot(
        context_generation=3,
        corpus_revision=4,
        index_revision=5,
    )

    # When: the profile consent revision changes without voice recognition.
    _ = memory.set_profile_revisions(ProfileRevision(1), ConsentRevision(1))

    # Then: the earlier task snapshot is stale.
    assert memory.is_current(captured) is False


def test_memory_rejects_conflicts_and_prohibited_biometric_categories() -> None:
    # Given: an already accepted ordinary preference with its provenance retained.
    memory = MutableMemory(session_id=SessionId("session-1"), policy=MemoryPolicy())
    result = memory.reduce(_proposal(value="小莓", confidence=95))
    assert isinstance(result, MemoryCommitAccepted)

    # When: a conflicting value and prohibited biometric proposal are submitted.
    conflict = memory.reduce(
        MemoryProposal(
            key=MemoryKey("preferred_name"),
            value="莓莓",
            category=MemoryCategory.ORDINARY_PREFERENCE,
            confidence=MemoryConfidence(95),
            base_revision=ProposalRevision(1),
            provenance=_proposal(value="莓莓", confidence=95).provenance,
        )
    )
    biometric = memory.reduce(
        MemoryProposal(
            key=MemoryKey("voice"),
            value="ciphertext",
            category=MemoryCategory.BIOMETRIC,
            confidence=MemoryConfidence(99),
            base_revision=ProposalRevision(1),
            provenance=_proposal(value="ciphertext", confidence=99).provenance,
        )
    )

    # Then: neither proposal changes the accepted ordinary preference revision.
    assert conflict == MemoryCommitRejected(MemoryCommitRejection.CONFLICT)
    assert biometric == MemoryCommitRejected(MemoryCommitRejection.RESTRICTED_CATEGORY)
    assert memory.snapshot.entries[0].value == "小莓"


def _proposal(
    *,
    value: str,
    confidence: int,
    category: MemoryCategory = MemoryCategory.ORDINARY_PREFERENCE,
) -> MemoryProposal:
    return MemoryProposal(
        key=MemoryKey("preferred_name"),
        value=value,
        category=category,
        confidence=MemoryConfidence(confidence),
        base_revision=ProposalRevision(0),
        provenance=MemoryProvenance(
            source=MemorySource.AGENT_PROPOSAL,
            trace_id=TraceId("trace-1"),
            session_id=SessionId("session-1"),
            turn_id=TurnId("turn-1"),
            evidence_id="finalized-input-1",
        ),
    )
