"""Scheduler-owned composition of revisioned session data dependencies."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from orchestrator.identity import (
    InMemoryVoiceProfileVault,
    ProfileCorrection,
    ProfileEnrollment,
    ProfileRecognition,
    ProfileRecognitionResult,
    VoiceProfileId,
)
from orchestrator.ids import SessionId
from orchestrator.memory import (
    MemoryCommitAccepted,
    MemoryCommitResult,
    MemoryKey,
    MemoryPolicy,
    MemoryProposal,
    MutableMemory,
)
from orchestrator.memory_store import MemoryStore
from orchestrator.profile_store import VoiceProfileStore
from orchestrator.profile_vault import FileVoiceProfileVault
from orchestrator.prompt_composition import PromptSnapshot
from orchestrator.retrieval import VersionedRetrievalProvider
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.transient_context import ContextMaterial, TransientContext
from orchestrator.voice_profile_service import VoiceProfileService


def _monotonic_ms() -> int:
    """Return the real monotonic clock used for startup retention enforcement."""
    return int(monotonic() * 1000)


@dataclass(frozen=True, slots=True)
class ProfilePersistence:
    """Durable profile dependencies owned by one session data state."""

    store: VoiceProfileStore | None = None
    vault_directory: Path | None = None
    clock: Callable[[], int] = _monotonic_ms


@dataclass(slots=True)
class SessionDataState:
    """Own mutable data whose revisions fence scheduler-owned background work."""

    memory: MutableMemory
    context: TransientContext
    profiles: VoiceProfileService
    retrieval: VersionedRetrievalProvider
    memory_store: MemoryStore | None = None
    invalidate_pending: Callable[[str], None] = field(default=lambda _reason: None)

    @classmethod
    def create(
        cls,
        *,
        session_id: SessionId,
        retrieval: VersionedRetrievalProvider,
        memory_store: MemoryStore | None = None,
        profile_persistence: ProfilePersistence | None = None,
    ) -> "SessionDataState":
        """Create the one data authority for a session and its known corpus snapshot."""
        policy = MemoryPolicy()
        stored = memory_store.load(session_id) if memory_store is not None else None
        memory = (
            MutableMemory(session_id=session_id, policy=policy)
            if stored is None
            else MutableMemory.restore(
                session_id=session_id,
                policy=policy,
                snapshot=stored,
            )
        )
        persistence = profile_persistence or ProfilePersistence()
        profiles = VoiceProfileService(
            session_id=session_id,
            vault=FileVoiceProfileVault(
                persistence.vault_directory, str(session_id)
            )
            if persistence.vault_directory is not None
            else InMemoryVoiceProfileVault(),
            minimum_confidence=90,
            store=persistence.store,
        )
        _ = profiles.expire(now_ms=persistence.clock())
        _ = profiles.bind_memory(memory)
        return cls(
            memory=memory,
            context=TransientContext(session_id=session_id),
            profiles=profiles,
            retrieval=retrieval,
            memory_store=memory_store,
        )

    @property
    def task_snapshot(self) -> TaskStateSnapshot:
        """Capture every versioned data dependency for scheduler task admission."""
        retrieval = self.retrieval.snapshot
        return self.memory.task_snapshot(
            context_generation=self.context.snapshot.generation,
            corpus_revision=int(retrieval.corpus_revision),
            index_revision=int(retrieval.index_revision),
        )

    def is_current(self, snapshot: TaskStateSnapshot) -> bool:
        """Report whether all mutable and immutable task inputs still match."""
        return snapshot == self.task_snapshot

    def reduce_memory(self, proposal: MemoryProposal) -> MemoryCommitResult:
        """Apply a policy-approved ordinary preference proposal once."""
        result = self.memory.reduce(proposal)
        match result:
            case MemoryCommitAccepted(snapshot=snapshot):
                if self.memory_store is not None:
                    self.memory_store.save(snapshot)
            case _:
                pass
        return result

    def delete_memory(self, key: MemoryKey) -> None:
        """Delete one preference, persist the new revision, and stale queued work."""
        snapshot = self.memory.delete(key)
        if self.memory_store is not None:
            self.memory_store.save(snapshot)

    def consider_context(self, material: ContextMaterial) -> None:
        """Admit only lifecycle-approved material into session-local context."""
        _ = self.context.consider(material)

    def reset_context(self) -> None:
        """Clear session-local context and invalidate work captured before the reset."""
        _ = self.context.reset()

    def enroll_profile(self, enrollment: ProfileEnrollment) -> VoiceProfileId:
        """Enroll an explicitly consented profile and refresh task revisions."""
        profile_id = self.profiles.enroll(enrollment)
        _ = self.profiles.bind_memory(self.memory)
        return profile_id

    def correct_profile(
        self, correction: ProfileCorrection
    ) -> ProfileRecognitionResult:
        """Apply a non-voice recovery correction and refresh task revisions."""
        result = self.profiles.correct(correction)
        _ = self.profiles.bind_memory(self.memory)
        return result

    def confirm_profile(self, profile_id: VoiceProfileId) -> ProfileRecognitionResult:
        """Confirm an explicitly consented profile through a non-voice control path."""
        result = self.profiles.confirm(profile_id)
        _ = self.profiles.bind_memory(self.memory)
        return result

    def recognize_profile(
        self, recognition: ProfileRecognition
    ) -> ProfileRecognitionResult:
        """Return non-authorizing personalization only for a live consented profile."""
        return self.profiles.recognize(recognition)

    def revoke_profile_consent(self, profile_id: VoiceProfileId) -> None:
        """Atomically erase the template, revoke recognition, and stale queued work."""
        self.profiles.revoke_consent(profile_id)
        _ = self.profiles.bind_memory(self.memory)
        self.invalidate_pending(f"identity_revoked:{profile_id}")

    def expire_profiles(self, *, now_ms: int) -> bool:
        """Apply retention expiry and invalidate queued personalization work."""
        expired = self.profiles.expire(now_ms=now_ms)
        if expired:
            _ = self.profiles.bind_memory(self.memory)
            self.invalidate_pending("identity_expired")
        return expired

    def delete_profile(self, profile_id: VoiceProfileId) -> None:
        """Atomically erase profile state and invalidate all profile-dependent work."""
        self.profiles.delete(profile_id)
        _ = self.profiles.bind_memory(self.memory)
        self.invalidate_pending(f"identity_deleted:{profile_id}")

    def prompt_snapshot(self, *, max_context_chars: int) -> PromptSnapshot:
        """Return approved human-readable memory and finalized context for a prompt."""
        return PromptSnapshot(
            task_state=self.task_snapshot,
            context_entries=tuple(
                entry.text for entry in self.context.snapshot.entries
            ),
            max_context_chars=max_context_chars,
            memory_entries=tuple(
                f"{entry.key}={entry.value}" for entry in self.memory.snapshot.entries
            ),
        )
