
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
    return int(monotonic() * 1000)


@dataclass(frozen=True, slots=True)
class ProfilePersistence:

    store: VoiceProfileStore | None = None

    vault_directory: Path | None = None

    clock: Callable[[], int] = _monotonic_ms


@dataclass(slots=True)
class SessionDataState:

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
            vault=FileVoiceProfileVault(persistence.vault_directory, str(session_id))
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
        retrieval = self.retrieval.snapshot

        return self.memory.task_snapshot(
            context_generation=self.context.snapshot.generation,
            corpus_revision=int(retrieval.corpus_revision),
            index_revision=int(retrieval.index_revision),
        )

    def is_current(self, snapshot: TaskStateSnapshot) -> bool:
        return snapshot == self.task_snapshot

    def reduce_memory(self, proposal: MemoryProposal) -> MemoryCommitResult:
        result = self.memory.reduce(proposal)

        match result:
            case MemoryCommitAccepted(snapshot=snapshot):
                if self.memory_store is not None:
                    self.memory_store.save(snapshot)

            case _:
                pass

        return result

    def delete_memory(self, key: MemoryKey) -> None:
        snapshot = self.memory.delete(key)

        if self.memory_store is not None:
            self.memory_store.save(snapshot)

    def consider_context(self, material: ContextMaterial) -> None:
        _ = self.context.consider(material)

    def reset_context(self) -> None:
        _ = self.context.reset()

    def enroll_profile(self, enrollment: ProfileEnrollment) -> VoiceProfileId:
        profile_id = self.profiles.enroll(enrollment)

        _ = self.profiles.bind_memory(self.memory)

        return profile_id

    def correct_profile(
        self, correction: ProfileCorrection
    ) -> ProfileRecognitionResult:
        result = self.profiles.correct(correction)

        _ = self.profiles.bind_memory(self.memory)

        return result

    def confirm_profile(self, profile_id: VoiceProfileId) -> ProfileRecognitionResult:
        result = self.profiles.confirm(profile_id)

        _ = self.profiles.bind_memory(self.memory)

        return result

    def recognize_profile(
        self, recognition: ProfileRecognition
    ) -> ProfileRecognitionResult:
        return self.profiles.recognize(recognition)

    def revoke_profile_consent(self, profile_id: VoiceProfileId) -> None:
        self.profiles.revoke_consent(profile_id)

        _ = self.profiles.bind_memory(self.memory)

        self.invalidate_pending(f"identity_revoked:{profile_id}")

    def expire_profiles(self, *, now_ms: int) -> bool:
        expired = self.profiles.expire(now_ms=now_ms)

        if expired:
            _ = self.profiles.bind_memory(self.memory)

            self.invalidate_pending("identity_expired")

        return expired

    def delete_profile(self, profile_id: VoiceProfileId) -> None:
        self.profiles.delete(profile_id)

        _ = self.profiles.bind_memory(self.memory)

        self.invalidate_pending(f"identity_deleted:{profile_id}")

    def prompt_snapshot(self, *, max_context_chars: int) -> PromptSnapshot:
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
