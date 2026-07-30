"""模块契约说明.

职责: 提供 orchestrator.session_data
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """函数契约说明.

    功能: 执行 _monotonic_ms 的同步逻辑,并协调 int,
    monotonic。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `int`。
    """
    return int(monotonic() * 1000)


@dataclass(frozen=True, slots=True)
class ProfilePersistence:
    """类契约说明.

    职责: 保存 ProfilePersistence
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: store、vault_directory、clock。
    """

    store: VoiceProfileStore | None = None

    vault_directory: Path | None = None

    clock: Callable[[], int] = _monotonic_ms


@dataclass(slots=True)
class SessionDataState:
    """类契约说明.

    职责: 保存 SessionDataState
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: memory、context、profiles、retr
    ieval、memory_store、invalidate_pendin
    g。 方法: create、task_snapshot、is_curre
    nt、reduce_memory、delete_memory、consi
    der_context。
    """

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
        """函数契约说明.

        功能: 执行 create 的同步逻辑,并协调
        MemoryPolicy,
        VoiceProfileService, expire,
        bind_memory。
        参数: cls 表示当前类。 session_id:
        SessionId。 必填。 retrieval:
        VersionedRetrievalProvider。 必填。
        memory_store: MemoryStore |
        None。 可省略。 profile_persistence:
        ProfilePersistence | None。 可省略。
        契约: 同步调用。 返回
        `'SessionDataState'`。
        """
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
        """函数契约说明.

        功能: 执行 task_snapshot 的同步逻辑,并协调
        task_snapshot, int。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `TaskStateSnapshot`。
        """
        retrieval = self.retrieval.snapshot

        return self.memory.task_snapshot(
            context_generation=self.context.snapshot.generation,
            corpus_revision=int(retrieval.corpus_revision),
            index_revision=int(retrieval.index_revision),
        )

    def is_current(self, snapshot: TaskStateSnapshot) -> bool:
        """函数契约说明.

        功能: 执行 is_current 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 snapshot:
        TaskStateSnapshot。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        return snapshot == self.task_snapshot

    def reduce_memory(self, proposal: MemoryProposal) -> MemoryCommitResult:
        """函数契约说明.

        功能: 执行 reduce_memory 的同步逻辑,并协调
        reduce, save。
        参数: self 表示当前实例。 proposal:
        MemoryProposal。 必填。
        契约: 同步调用。 返回
        `MemoryCommitResult`。
        """
        result = self.memory.reduce(proposal)

        match result:
            case MemoryCommitAccepted(snapshot=snapshot):
                if self.memory_store is not None:
                    self.memory_store.save(snapshot)

            case _:
                pass

        return result

    def delete_memory(self, key: MemoryKey) -> None:
        """函数契约说明.

        功能: 执行 delete_memory 的同步逻辑,并协调
        delete, save。
        参数: self 表示当前实例。 key: MemoryKey。
        必填。
        契约: 同步调用。 返回 `None`。
        """
        snapshot = self.memory.delete(key)

        if self.memory_store is not None:
            self.memory_store.save(snapshot)

    def consider_context(self, material: ContextMaterial) -> None:
        """函数契约说明.

        功能: 执行 consider_context
        的同步逻辑,并协调 consider。
        参数: self 表示当前实例。 material:
        ContextMaterial。 必填。
        契约: 同步调用。 返回 `None`。
        """
        _ = self.context.consider(material)

    def reset_context(self) -> None:
        """函数契约说明.

        功能: 执行 reset_context 的同步逻辑,并协调
        reset。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        _ = self.context.reset()

    def enroll_profile(self, enrollment: ProfileEnrollment) -> VoiceProfileId:
        """函数契约说明.

        功能: 执行 enroll_profile 的同步逻辑,并协调
        enroll, bind_memory。
        参数: self 表示当前实例。 enrollment:
        ProfileEnrollment。 必填。
        契约: 同步调用。 返回 `VoiceProfileId`。
        """
        profile_id = self.profiles.enroll(enrollment)

        _ = self.profiles.bind_memory(self.memory)

        return profile_id

    def correct_profile(
        self, correction: ProfileCorrection
    ) -> ProfileRecognitionResult:
        """函数契约说明.

        功能: 执行 correct_profile 的同步逻辑,并协调
        correct, bind_memory。
        参数: self 表示当前实例。 correction:
        ProfileCorrection。 必填。
        契约: 同步调用。 返回
        `ProfileRecognitionResult`。
        """
        result = self.profiles.correct(correction)

        _ = self.profiles.bind_memory(self.memory)

        return result

    def confirm_profile(self, profile_id: VoiceProfileId) -> ProfileRecognitionResult:
        """函数契约说明.

        功能: 执行 confirm_profile 的同步逻辑,并协调
        confirm, bind_memory。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回
        `ProfileRecognitionResult`。
        """
        result = self.profiles.confirm(profile_id)

        _ = self.profiles.bind_memory(self.memory)

        return result

    def recognize_profile(
        self, recognition: ProfileRecognition
    ) -> ProfileRecognitionResult:
        """函数契约说明.

        功能: 执行 recognize_profile
        的同步逻辑,并协调 recognize。
        参数: self 表示当前实例。 recognition:
        ProfileRecognition。 必填。
        契约: 同步调用。 返回
        `ProfileRecognitionResult`。
        """
        return self.profiles.recognize(recognition)

    def revoke_profile_consent(self, profile_id: VoiceProfileId) -> None:
        """函数契约说明.

        功能: 执行 revoke_profile_consent
        的同步逻辑,并协调 revoke_consent,
        bind_memory, invalidate_pending。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self.profiles.revoke_consent(profile_id)

        _ = self.profiles.bind_memory(self.memory)

        self.invalidate_pending(f"identity_revoked:{profile_id}")

    def expire_profiles(self, *, now_ms: int) -> bool:
        """函数契约说明.

        功能: 执行 expire_profiles 的同步逻辑,并协调
        expire, bind_memory,
        invalidate_pending。
        参数: self 表示当前实例。 now_ms: int。
        必填。
        契约: 同步调用。 返回 `bool`。
        """
        expired = self.profiles.expire(now_ms=now_ms)

        if expired:
            _ = self.profiles.bind_memory(self.memory)

            self.invalidate_pending("identity_expired")

        return expired

    def delete_profile(self, profile_id: VoiceProfileId) -> None:
        """函数契约说明.

        功能: 执行 delete_profile 的同步逻辑,并协调
        delete, bind_memory,
        invalidate_pending。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self.profiles.delete(profile_id)

        _ = self.profiles.bind_memory(self.memory)

        self.invalidate_pending(f"identity_deleted:{profile_id}")

    def prompt_snapshot(self, *, max_context_chars: int) -> PromptSnapshot:
        """函数契约说明.

        功能: 执行 prompt_snapshot 的同步逻辑,并协调
        PromptSnapshot, tuple。
        参数: self 表示当前实例。
        max_context_chars: int。 必填。
        契约: 同步调用。 返回 `PromptSnapshot`。
        """
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
