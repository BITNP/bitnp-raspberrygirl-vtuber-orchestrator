"""模块契约说明.

职责: 提供 orchestrator.memory
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 定义 MemoryCategory 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    ORDINARY_PREFERENCE = "ordinary_preference"

    RESTRICTED = "restricted"

    BIOMETRIC = "biometric"

    IDENTITY = "identity"

    AUTHORIZATION = "authorization"


@unique
class MemorySource(StrEnum):
    """类契约说明.

    职责: 定义 MemorySource 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    AGENT_PROPOSAL = "agent_proposal"

    USER_REQUEST = "user_request"


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    """类契约说明.

    职责: 保存 MemoryProvenance
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: source、trace_id、session_id、t
    urn_id、evidence_id。
    """

    source: MemorySource

    trace_id: TraceId

    session_id: SessionId

    turn_id: TurnId

    evidence_id: str


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    """类契约说明.

    职责: 保存 MemoryProposal
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: key、value、category、confidenc
    e、base_revision、provenance。
    """

    key: MemoryKey

    value: str

    category: MemoryCategory

    confidence: MemoryConfidence

    base_revision: ProposalRevision

    provenance: MemoryProvenance


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """类契约说明.

    职责: 保存 MemoryEntry
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: key、value、provenance。
    """

    key: MemoryKey

    value: str

    provenance: MemoryProvenance


@dataclass(frozen=True, slots=True)
class MutableMemorySnapshot:
    """类契约说明.

    职责: 保存 MutableMemorySnapshot
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: revision、entries、profile_rev
    ision、consent_revision。
    """

    revision: MemoryRevision

    entries: tuple[MemoryEntry, ...]

    profile_revision: ProfileRevision

    consent_revision: ConsentRevision


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    """类契约说明.

    职责: 保存 MemoryPolicy
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: minimum_confidence。
    """

    minimum_confidence: MemoryConfidence = MemoryConfidence(90)


@unique
class MemoryCommitRejection(StrEnum):
    """类契约说明.

    职责: 定义 MemoryCommitRejection
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    STALE_PROPOSAL = "stale_proposal"

    SESSION_MISMATCH = "session_mismatch"

    RESTRICTED_CATEGORY = "restricted_category"

    UNSUPPORTED_ASSERTION = "unsupported_assertion"

    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class MemoryCommitAccepted:
    """类契约说明.

    职责: 保存 MemoryCommitAccepted
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: snapshot。
    """

    snapshot: MutableMemorySnapshot


@dataclass(frozen=True, slots=True)
class MemoryCommitRejected:
    """类契约说明.

    职责: 保存 MemoryCommitRejected
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。
    """

    reason: MemoryCommitRejection


type MemoryCommitResult = MemoryCommitAccepted | MemoryCommitRejected


@final
class MutableMemory:
    """类契约说明.

    职责: 定义 MutableMemory 的状态、行为和对外协作边界。
    契约: 方法: __init__、restore、snapshot、re
    duce、delete、set_profile_revisions。
    """

    def __init__(self, *, session_id: SessionId, policy: MemoryPolicy) -> None:
        """函数契约说明.

        功能: 初始化 MutableMemory
        的字段并建立实例不变式。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。 policy:
        MemoryPolicy。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 restore 的同步逻辑,并协调 cls。
        参数: cls 表示当前类。 session_id:
        SessionId。 必填。 policy:
        MemoryPolicy。 必填。 snapshot:
        MutableMemorySnapshot。 必填。
        契约: 同步调用。 返回 `'MutableMemory'`。
        """
        memory = cls(session_id=session_id, policy=policy)

        memory._revision = snapshot.revision

        memory._entries = {entry.key: entry for entry in snapshot.entries}

        memory._profile_revision = snapshot.profile_revision

        memory._consent_revision = snapshot.consent_revision

        return memory

    @property
    def snapshot(self) -> MutableMemorySnapshot:
        """函数契约说明.

        功能: 执行 snapshot 的同步逻辑,并协调
        MutableMemorySnapshot, tuple,
        values。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `MutableMemorySnapshot`。
        """
        return MutableMemorySnapshot(
            revision=self._revision,
            entries=tuple(self._entries.values()),
            profile_revision=self._profile_revision,
            consent_revision=self._consent_revision,
        )

    def reduce(self, proposal: MemoryProposal) -> MemoryCommitResult:
        """函数契约说明.

        功能: 执行 reduce 的同步逻辑,并协调
        _rejection, MemoryRevision,
        MemoryEntry,
        MemoryCommitAccepted。
        参数: self 表示当前实例。 proposal:
        MemoryProposal。 必填。
        契约: 同步调用。 返回
        `MemoryCommitResult`。
        """
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
        """函数契约说明.

        功能: 执行 delete 的同步逻辑,并协调 pop,
        MemoryRevision。
        参数: self 表示当前实例。 key: MemoryKey。
        必填。
        契约: 同步调用。 返回
        `MutableMemorySnapshot`。
        """
        _ = self._entries.pop(key, None)

        self._revision = MemoryRevision(self._revision + 1)

        return self.snapshot

    def set_profile_revisions(
        self,
        profile_revision: ProfileRevision,
        consent_revision: ConsentRevision,
    ) -> MutableMemorySnapshot:
        """函数契约说明.

        功能: 执行 set_profile_revisions
        的同步逻辑,并产出 _profile_revision,
        _consent_revision。
        参数: self 表示当前实例。
        profile_revision:
        ProfileRevision。 必填。
        consent_revision:
        ConsentRevision。 必填。
        契约: 同步调用。 返回
        `MutableMemorySnapshot`。
        """
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
        """函数契约说明.

        功能: 执行 task_snapshot 的同步逻辑,并协调
        TaskStateSnapshot,
        ContextGeneration,
        CorpusRevision, IndexRevision。
        参数: self 表示当前实例。
        context_generation: int。 必填。
        corpus_revision: int。 必填。
        index_revision: int。 必填。
        契约: 同步调用。 返回
        `TaskStateSnapshot`。
        """
        return TaskStateSnapshot(
            memory_revision=self._revision,
            context_generation=ContextGeneration(context_generation),
            profile_revision=self._profile_revision,
            consent_revision=self._consent_revision,
            corpus_revision=CorpusRevision(corpus_revision),
            index_revision=IndexRevision(index_revision),
        )

    def is_current(self, snapshot: TaskStateSnapshot) -> bool:
        """函数契约说明.

        功能: 执行 is_current 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 snapshot:
        TaskStateSnapshot。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        return (
            snapshot.memory_revision == self._revision
            and snapshot.profile_revision == self._profile_revision
            and snapshot.consent_revision == self._consent_revision
        )

    def _rejection(self, proposal: MemoryProposal) -> MemoryCommitRejection | None:
        """函数契约说明.

        功能: 执行 _rejection 的同步逻辑,并协调 get,
        ProposalRevision。
        参数: self 表示当前实例。 proposal:
        MemoryProposal。 必填。
        契约: 同步调用。 返回
        `MemoryCommitRejection | None`。
        """
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
