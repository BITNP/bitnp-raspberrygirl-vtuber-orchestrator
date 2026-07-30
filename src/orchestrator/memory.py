"""Scheduler-owned policy reducer for small revisioned mutable preferences."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType, final

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.state_snapshots import (
    ConsentRevision,
    ContextGeneration,
    CorpusRevision,
    IndexRevision,
    MemoryRevision,
    ProfileRevision,
    TaskStateSnapshot,
)

MemoryKey = NewType("MemoryKey", str)
MemoryConfidence = NewType("MemoryConfidence", int)
ProposalRevision = NewType("ProposalRevision", int)


@unique
class MemoryCategory(StrEnum):
    """Closed durable-memory categories that the policy can classify."""

    ORDINARY_PREFERENCE = "ordinary_preference"
    RESTRICTED = "restricted"
    BIOMETRIC = "biometric"
    IDENTITY = "identity"
    AUTHORIZATION = "authorization"


@unique
class MemorySource(StrEnum):
    """The controlled origin of a proposal rather than raw source content."""

    AGENT_PROPOSAL = "agent_proposal"
    USER_REQUEST = "user_request"


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    """Correlation and durable evidence identifier for one memory proposal."""

    source: MemorySource
    trace_id: TraceId
    session_id: SessionId
    turn_id: TurnId
    evidence_id: str


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    """Typed suggestion that must pass scheduler policy before persistence."""

    key: MemoryKey
    value: str
    category: MemoryCategory
    confidence: MemoryConfidence
    base_revision: ProposalRevision
    provenance: MemoryProvenance


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """Human-readable preference retained with only typed provenance metadata."""

    key: MemoryKey
    value: str
    provenance: MemoryProvenance


@dataclass(frozen=True, slots=True)
class MutableMemorySnapshot:
    """Current durable-memory value and all revisions that influence task work."""

    revision: MemoryRevision
    entries: tuple[MemoryEntry, ...]
    profile_revision: ProfileRevision
    consent_revision: ConsentRevision


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    """Explicit autonomous-commit policy for ordinary non-sensitive preferences."""

    minimum_confidence: MemoryConfidence = MemoryConfidence(90)


@unique
class MemoryCommitRejection(StrEnum):
    """Closed reasons a proposal cannot mutate durable state."""

    STALE_PROPOSAL = "stale_proposal"
    SESSION_MISMATCH = "session_mismatch"
    RESTRICTED_CATEGORY = "restricted_category"
    UNSUPPORTED_ASSERTION = "unsupported_assertion"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class MemoryCommitAccepted:
    """A policy-approved mutation with its newly captured immutable snapshot."""

    snapshot: MutableMemorySnapshot


@dataclass(frozen=True, slots=True)
class MemoryCommitRejected:
    """A refused proposal that left durable memory unchanged."""

    reason: MemoryCommitRejection


type MemoryCommitResult = MemoryCommitAccepted | MemoryCommitRejected


@final
class MutableMemory:
    """Mutable scheduler state whose only mutation path is the policy reducer."""

    def __init__(self, *, session_id: SessionId, policy: MemoryPolicy) -> None:
        """Create empty session-owned memory and identity revision state."""
        self._session_id = session_id
        self._policy = policy
        self._revision = MemoryRevision(0)
        self._entries: dict[MemoryKey, MemoryEntry] = {}
        self._profile_revision = ProfileRevision(0)
        self._consent_revision = ConsentRevision(0)

    @classmethod
    def restore(
        cls,
        *,
        session_id: SessionId,
        policy: MemoryPolicy,
        snapshot: MutableMemorySnapshot,
    ) -> "MutableMemory":
        """Recreate policy-owned mutable state from one durable accepted snapshot."""
        memory = cls(session_id=session_id, policy=policy)
        memory._revision = snapshot.revision
        memory._entries = {entry.key: entry for entry in snapshot.entries}
        memory._profile_revision = snapshot.profile_revision
        memory._consent_revision = snapshot.consent_revision
        return memory

    @property
    def snapshot(self) -> MutableMemorySnapshot:
        """Return a deterministic immutable view of accepted preferences."""
        return MutableMemorySnapshot(
            revision=self._revision,
            entries=tuple(self._entries.values()),
            profile_revision=self._profile_revision,
            consent_revision=self._consent_revision,
        )

    def reduce(self, proposal: MemoryProposal) -> MemoryCommitResult:
        """Commit only a current, ordinary, supported agent preference proposal."""
        rejection = self._rejection(proposal)
        if rejection is not None:
            return MemoryCommitRejected(rejection)
        self._revision = MemoryRevision(self._revision + 1)
        self._entries[proposal.key] = MemoryEntry(
            key=proposal.key,
            value=proposal.value,
            provenance=proposal.provenance,
        )
        return MemoryCommitAccepted(self.snapshot)

    def delete(self, key: MemoryKey) -> MutableMemorySnapshot:
        """Remove one preference and advance revision even when it is already absent."""
        _ = self._entries.pop(key, None)
        self._revision = MemoryRevision(self._revision + 1)
        return self.snapshot

    def set_profile_revisions(
        self,
        profile_revision: ProfileRevision,
        consent_revision: ConsentRevision,
    ) -> MutableMemorySnapshot:
        """Advance identity-consent revisions without handling biometric data."""
        self._profile_revision = profile_revision
        self._consent_revision = consent_revision
        return self.snapshot

    def task_snapshot(
        self,
        *,
        context_generation: int,
        corpus_revision: int,
        index_revision: int,
    ) -> TaskStateSnapshot:
        """Capture all revisioned inputs needed by dependent scheduler tasks."""
        return TaskStateSnapshot(
            memory_revision=self._revision,
            context_generation=ContextGeneration(context_generation),
            profile_revision=self._profile_revision,
            consent_revision=self._consent_revision,
            corpus_revision=CorpusRevision(corpus_revision),
            index_revision=IndexRevision(index_revision),
        )

    def is_current(self, snapshot: TaskStateSnapshot) -> bool:
        """Report whether a task snapshot still matches memory and identity state."""
        return (
            snapshot.memory_revision == self._revision
            and snapshot.profile_revision == self._profile_revision
            and snapshot.consent_revision == self._consent_revision
        )

    def _rejection(self, proposal: MemoryProposal) -> MemoryCommitRejection | None:
        if proposal.base_revision != ProposalRevision(self._revision):
            return MemoryCommitRejection.STALE_PROPOSAL
        if proposal.provenance.session_id != self._session_id:
            return MemoryCommitRejection.SESSION_MISMATCH
        if proposal.category is not MemoryCategory.ORDINARY_PREFERENCE:
            return MemoryCommitRejection.RESTRICTED_CATEGORY
        if proposal.confidence < self._policy.minimum_confidence:
            return MemoryCommitRejection.UNSUPPORTED_ASSERTION
        existing = self._entries.get(proposal.key)
        if existing is not None and existing.value != proposal.value:
            return MemoryCommitRejection.CONFLICT
        return None
