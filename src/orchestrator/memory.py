from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from time import time_ns
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


def _now_ms() -> int:
    return time_ns() // 1_000_000


MemoryKey = NewType("MemoryKey", str)

MemoryConfidence = NewType("MemoryConfidence", int)

ProposalRevision = NewType("ProposalRevision", int)


@unique
class MemoryCategory(StrEnum):
    ORDINARY_PREFERENCE = "ordinary_preference"

    RESTRICTED = "restricted"

    BIOMETRIC = "biometric"

    IDENTITY = "identity"

    AUTHORIZATION = "authorization"


@unique
class MemorySource(StrEnum):
    AGENT_PROPOSAL = "agent_proposal"

    USER_REQUEST = "user_request"


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    source: MemorySource

    trace_id: TraceId

    session_id: SessionId

    turn_id: TurnId

    evidence_id: str


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    key: MemoryKey

    value: str

    category: MemoryCategory

    confidence: MemoryConfidence

    base_revision: ProposalRevision

    provenance: MemoryProvenance


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    key: MemoryKey

    value: str

    provenance: MemoryProvenance

    category: MemoryCategory = MemoryCategory.ORDINARY_PREFERENCE

    confidence: MemoryConfidence = MemoryConfidence(100)

    updated_at_ms: int = 0


@dataclass(frozen=True, slots=True)
class MemoryConflictAudit:
    """Retained in-memory evidence that a higher-confidence fact replaced one."""

    key: MemoryKey

    replaced_value: str

    replacement_value: str

    replaced_confidence: MemoryConfidence

    replacement_confidence: MemoryConfidence


@dataclass(frozen=True, slots=True)
class MutableMemorySnapshot:
    revision: MemoryRevision

    entries: tuple[MemoryEntry, ...]

    profile_revision: ProfileRevision

    consent_revision: ConsentRevision


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    minimum_confidence: MemoryConfidence = MemoryConfidence(90)


@unique
class MemoryCommitRejection(StrEnum):
    STALE_PROPOSAL = "stale_proposal"

    SESSION_MISMATCH = "session_mismatch"

    RESTRICTED_CATEGORY = "restricted_category"

    UNSUPPORTED_ASSERTION = "unsupported_assertion"

    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class MemoryCommitAccepted:
    snapshot: MutableMemorySnapshot


@dataclass(frozen=True, slots=True)
class MemoryCommitRejected:
    reason: MemoryCommitRejection


type MemoryCommitResult = MemoryCommitAccepted | MemoryCommitRejected


@final
class MutableMemory:
    def __init__(
        self,
        *,
        session_id: SessionId,
        policy: MemoryPolicy,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        self._session_id = session_id

        self._policy = policy

        self._revision = MemoryRevision(0)

        self._entries: dict[MemoryKey, MemoryEntry] = {}

        self._profile_revision = ProfileRevision(0)

        self._consent_revision = ConsentRevision(0)

        self._clock = clock

        self._conflict_audit: list[MemoryConflictAudit] = []

    @classmethod
    def restore(
        cls,
        *,
        session_id: SessionId,
        policy: MemoryPolicy,
        snapshot: MutableMemorySnapshot,
    ) -> "MutableMemory":
        memory = cls(session_id=session_id, policy=policy)

        memory._revision = snapshot.revision

        memory._entries = {entry.key: entry for entry in snapshot.entries}

        memory._profile_revision = snapshot.profile_revision

        memory._consent_revision = snapshot.consent_revision

        return memory

    @property
    def snapshot(self) -> MutableMemorySnapshot:
        return MutableMemorySnapshot(
            revision=self._revision,
            entries=tuple(self._entries.values()),
            profile_revision=self._profile_revision,
            consent_revision=self._consent_revision,
        )

    @property
    def conflict_audit(self) -> tuple[MemoryConflictAudit, ...]:
        return tuple(self._conflict_audit)

    def reduce(self, proposal: MemoryProposal) -> MemoryCommitResult:
        rejection = self.validate(proposal)

        if rejection is not None:
            return MemoryCommitRejected(rejection)

        self._revision = MemoryRevision(self._revision + 1)

        existing = self._entries.get(proposal.key)
        if existing is not None and existing.value != proposal.value:
            self._conflict_audit.append(
                MemoryConflictAudit(
                    key=proposal.key,
                    replaced_value=existing.value,
                    replacement_value=proposal.value,
                    replaced_confidence=existing.confidence,
                    replacement_confidence=proposal.confidence,
                )
            )

        self._entries[proposal.key] = MemoryEntry(
            key=proposal.key,
            value=proposal.value,
            provenance=proposal.provenance,
            category=proposal.category,
            confidence=proposal.confidence,
            updated_at_ms=self._clock(),
        )

        return MemoryCommitAccepted(self.snapshot)

    def validate(self, proposal: MemoryProposal) -> MemoryCommitRejection | None:
        """Validate a proposal without mutating the session memory revision."""
        return self._rejection(proposal)

    def delete(self, key: MemoryKey) -> MutableMemorySnapshot:
        _ = self._entries.pop(key, None)

        self._revision = MemoryRevision(self._revision + 1)

        return self.snapshot

    def clear(self) -> MutableMemorySnapshot:
        self._entries.clear()
        self._conflict_audit.clear()
        self._revision = MemoryRevision(self._revision + 1)
        return self.snapshot

    def set_profile_revisions(
        self,
        profile_revision: ProfileRevision,
        consent_revision: ConsentRevision,
    ) -> MutableMemorySnapshot:
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
        return TaskStateSnapshot(
            memory_revision=self._revision,
            context_generation=ContextGeneration(context_generation),
            profile_revision=self._profile_revision,
            consent_revision=self._consent_revision,
            corpus_revision=CorpusRevision(corpus_revision),
            index_revision=IndexRevision(index_revision),
        )

    def is_current(self, snapshot: TaskStateSnapshot) -> bool:
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

        if (
            existing is not None
            and existing.value != proposal.value
            and proposal.confidence <= existing.confidence
        ):
            return MemoryCommitRejection.CONFLICT

        return None
