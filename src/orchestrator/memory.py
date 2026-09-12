from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from hashlib import sha256
from time import time_ns
from typing import NewType, final

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.memory_policy import (
    MAX_MEMORY_AUDIT_RECORDS,
    MAX_MEMORY_CONFIDENCE,
    MAX_MEMORY_CONFLICT_RECORDS,
    MAX_MEMORY_ENTRIES,
    MAX_MEMORY_PROVENANCE_BYTES,
    MAX_MEMORY_TEXT_BYTES,
    contains_sensitive_memory,
    valid_memory_text,
)
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
class MemoryAudit:
    revision: int
    action: str
    key: str
    previous_digest: str
    value_digest: str
    trace_id: str
    turn_id: str
    evidence_id: str
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class MutableMemorySnapshot:
    revision: MemoryRevision

    entries: tuple[MemoryEntry, ...]

    profile_revision: ProfileRevision

    consent_revision: ConsentRevision

    conflict_audit: tuple[MemoryConflictAudit, ...] = ()

    audit: tuple[MemoryAudit, ...] = ()


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

    INVALID_FORMAT = "invalid_format"

    CAPACITY_EXCEEDED = "capacity_exceeded"


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
        self._audit: list[MemoryAudit] = []

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
        memory._conflict_audit = list(snapshot.conflict_audit)
        memory._audit = list(snapshot.audit)

        return memory

    @property
    def snapshot(self) -> MutableMemorySnapshot:
        return MutableMemorySnapshot(
            revision=self._revision,
            entries=tuple(self._entries.values()),
            profile_revision=self._profile_revision,
            consent_revision=self._consent_revision,
            conflict_audit=tuple(self._conflict_audit),
            audit=tuple(self._audit),
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

            self._conflict_audit = self._conflict_audit[-MAX_MEMORY_CONFLICT_RECORDS:]

        self._entries[proposal.key] = MemoryEntry(
            key=proposal.key,
            value=proposal.value,
            provenance=proposal.provenance,
            category=proposal.category,
            confidence=proposal.confidence,
            updated_at_ms=self._clock(),
        )
        self._append_audit(
            key=proposal.key,
            previous=existing,
            current=self._entries[proposal.key],
            provenance=proposal.provenance,
        )

        return MemoryCommitAccepted(self.snapshot)

    def validate(self, proposal: MemoryProposal) -> MemoryCommitRejection | None:
        """Validate a proposal without mutating the session memory revision."""
        return self._rejection(proposal)

    def delete(
        self, key: MemoryKey, *, provenance: MemoryProvenance | None = None
    ) -> MutableMemorySnapshot:
        previous = self._entries.pop(key, None)

        self._revision = MemoryRevision(self._revision + 1)
        self._conflict_audit = [
            item for item in self._conflict_audit if item.key != key
        ]
        self._append_audit(
            key=key, previous=previous, current=None, provenance=provenance
        )

        return self.snapshot

    def clear(self) -> MutableMemorySnapshot:
        self._entries.clear()
        self._conflict_audit.clear()
        self._audit.clear()
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

        rejection = self._content_rejection(proposal)
        if rejection is not None:
            return rejection

        existing = self._entries.get(proposal.key)

        projected = {entry.key: entry.value for entry in self._entries.values()} | {
            proposal.key: proposal.value
        }
        if (
            len(projected) > MAX_MEMORY_ENTRIES
            or sum(
                len(key.encode()) + len(value.encode())
                for key, value in projected.items()
            )
            > MAX_MEMORY_TEXT_BYTES
        ):
            return MemoryCommitRejection.CAPACITY_EXCEEDED

        if (
            existing is not None
            and existing.value != proposal.value
            and proposal.confidence <= existing.confidence
        ):
            return MemoryCommitRejection.CONFLICT

        return None

    def _content_rejection(
        self, proposal: MemoryProposal
    ) -> MemoryCommitRejection | None:
        if proposal.category is not MemoryCategory.ORDINARY_PREFERENCE:
            return MemoryCommitRejection.RESTRICTED_CATEGORY

        if contains_sensitive_memory(proposal.key, proposal.value):
            return MemoryCommitRejection.RESTRICTED_CATEGORY

        if not valid_memory_text(proposal.key, proposal.value):
            return MemoryCommitRejection.INVALID_FORMAT

        if any(
            not value or len(value.encode()) > MAX_MEMORY_PROVENANCE_BYTES
            for value in (
                proposal.provenance.trace_id,
                proposal.provenance.turn_id,
                proposal.provenance.evidence_id,
            )
        ):
            return MemoryCommitRejection.INVALID_FORMAT

        if (
            not self._policy.minimum_confidence
            <= proposal.confidence
            <= MAX_MEMORY_CONFIDENCE
        ):
            return MemoryCommitRejection.UNSUPPORTED_ASSERTION

        return None

    def _append_audit(
        self,
        *,
        key: MemoryKey,
        previous: MemoryEntry | None,
        current: MemoryEntry | None,
        provenance: MemoryProvenance | None,
    ) -> None:
        self._audit.append(
            MemoryAudit(
                revision=int(self._revision),
                action="delete"
                if current is None
                else "add"
                if previous is None
                else "replace",
                key=str(key),
                previous_digest=""
                if previous is None
                else sha256(previous.value.encode()).hexdigest(),
                value_digest=""
                if current is None
                else sha256(current.value.encode()).hexdigest(),
                trace_id="" if provenance is None else str(provenance.trace_id),
                turn_id="" if provenance is None else str(provenance.turn_id),
                evidence_id="" if provenance is None else provenance.evidence_id,
                updated_at_ms=self._clock(),
            )
        )
        self._audit = self._audit[-MAX_MEMORY_AUDIT_RECORDS:]
