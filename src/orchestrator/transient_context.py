"""模块契约说明.

职责: 提供 orchestrator.transient_context
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from hashlib import sha256
from typing import TYPE_CHECKING, NewType, Protocol, final, override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from orchestrator.ids import SegmentId, SessionId, TurnId


ContextSequence = NewType("ContextSequence", int)

ContextSourceId = NewType("ContextSourceId", str)

ModelId = NewType("ModelId", str)

TokenBudget = NewType("TokenBudget", int)


@dataclass(frozen=True, slots=True)
class ContextProvenance:
    """类契约说明.

    职责: 保存 ContextProvenance
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: session_id、turn_id、segment_i
    d、sequence、source_id。
    """

    session_id: SessionId

    turn_id: TurnId

    segment_id: SegmentId

    sequence: ContextSequence

    source_id: ContextSourceId


@dataclass(frozen=True, slots=True)
class FinalizedInput:
    """类契约说明.

    职责: 保存 FinalizedInput
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: provenance、text。
    """

    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class AcceptedOutput:
    """类契约说明.

    职责: 保存 AcceptedOutput
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: provenance、text。
    """

    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class PartialMaterial:
    """类契约说明.

    职责: 保存 PartialMaterial
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: provenance、text。
    """

    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class CancelledMaterial:
    """类契约说明.

    职责: 保存 CancelledMaterial
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: provenance、text。
    """

    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class StaleMaterial:
    """类契约说明.

    职责: 保存 StaleMaterial
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: provenance、text。
    """

    provenance: ContextProvenance

    text: str


type ContextMaterial = (
    FinalizedInput
    | AcceptedOutput
    | PartialMaterial
    | CancelledMaterial
    | StaleMaterial
)


@unique
class ContextEntryKind(StrEnum):
    """类契约说明.

    职责: 定义 ContextEntryKind
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    INPUT = "input"

    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class ContextEntry:
    """类契约说明.

    职责: 保存 ContextEntry
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: kind、provenance、text。
    """

    kind: ContextEntryKind

    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class TransientContextSnapshot:
    """类契约说明.

    职责: 保存 TransientContextSnapshot
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    session_id、generation、entries。
    """

    session_id: SessionId

    generation: int

    entries: tuple[ContextEntry, ...]


@dataclass(frozen=True, slots=True)
class ModelContextBudget:
    """类契约说明.

    职责: 保存 ModelContextBudget
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: input_tokens。 方法:
    __post_init__。
    """

    input_tokens: TokenBudget

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化 ModelContextBudget
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。 可能抛出
        InvalidContextBudgetError。
        """
        if self.input_tokens <= 0:
            raise InvalidContextBudgetError(input_tokens=self.input_tokens)


class ContextBudgetPolicy(Protocol):
    """类契约说明.

    职责: 声明 ContextBudgetPolicy
    协议接口,约束实现方必须提供的行为。
    契约: 方法: budget_for。
    """

    def budget_for(self, model_id: ModelId) -> ModelContextBudget:
        """函数契约说明.

        功能: 执行 budget_for 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 model_id:
        ModelId。 必填。
        契约: 同步调用。 返回
        `ModelContextBudget`。
        """
        ...


@dataclass(frozen=True, slots=True)
class StaticContextBudgetPolicy:
    """类契约说明.

    职责: 保存 StaticContextBudgetPolicy
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: model_id、budget。 方法:
    budget_for。
    """

    model_id: ModelId

    budget: ModelContextBudget

    def budget_for(self, model_id: ModelId) -> ModelContextBudget:
        """函数契约说明.

        功能: 执行 budget_for 的同步逻辑,并协调
        ModelBudgetUnavailableError。
        参数: self 表示当前实例。 model_id:
        ModelId。 必填。
        契约: 同步调用。 返回
        `ModelContextBudget`。 可能抛出
        ModelBudgetUnavailableError。
        """
        if model_id != self.model_id:
            raise ModelBudgetUnavailableError(model_id=model_id)

        return self.budget


@dataclass(frozen=True, slots=True)
class ContextDigest:
    """类契约说明.

    职责: 保存 ContextDigest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    source_provenances、content_hash。
    """

    source_provenances: tuple[ContextProvenance, ...]

    content_hash: str


@dataclass(frozen=True, slots=True)
class ContextComposition:
    """类契约说明.

    职责: 保存 ContextComposition
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: snapshot、entries、digests、con
    tent_token_count。
    """

    snapshot: TransientContextSnapshot

    entries: tuple[ContextEntry, ...]

    digests: tuple[ContextDigest, ...]

    content_token_count: TokenBudget


@dataclass(frozen=True, slots=True)
class ContextSessionMismatchError(Exception):
    """类契约说明.

    职责: 保存 ContextSessionMismatchError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: expected_session_id、actual_s
    ession_id。 方法: __str__。
    """

    expected_session_id: SessionId

    actual_session_id: SessionId

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return "transient context session mismatch"


@dataclass(frozen=True, slots=True)
class InvalidContextBudgetError(Exception):
    """类契约说明.

    职责: 保存 InvalidContextBudgetError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: input_tokens。 方法: __str__。
    """

    input_tokens: TokenBudget

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return "transient context budget must be positive"


@dataclass(frozen=True, slots=True)
class ModelBudgetUnavailableError(Exception):
    """类契约说明.

    职责: 保存 ModelBudgetUnavailableError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: model_id。 方法: __str__。
    """

    model_id: ModelId

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return "transient context model budget is unavailable"


@final
class TransientContext:
    """类契约说明.

    职责: 定义 TransientContext
    的状态、行为和对外协作边界。
    契约: 方法: __init__、snapshot、consider、r
    eset、compose、_ensure_session。
    """

    def __init__(self, *, session_id: SessionId) -> None:
        """函数契约说明.

        功能: 初始化 TransientContext
        的字段并建立实例不变式。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._session_id = session_id

        self._generation = 0

        self._entries: list[ContextEntry] = []

    @property
    def snapshot(self) -> TransientContextSnapshot:
        """函数契约说明.

        功能: 执行 snapshot 的同步逻辑,并协调
        TransientContextSnapshot, tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `TransientContextSnapshot`。
        """
        return TransientContextSnapshot(
            session_id=self._session_id,
            generation=self._generation,
            entries=tuple(self._entries),
        )

    def consider(self, material: ContextMaterial) -> TransientContextSnapshot:
        """函数契约说明.

        功能: 执行 consider 的同步逻辑,并协调
        _ensure_session, append,
        ContextEntry。
        参数: self 表示当前实例。 material:
        ContextMaterial。 必填。
        契约: 同步调用。 返回
        `TransientContextSnapshot`。
        """
        self._ensure_session(material.provenance)

        match material:
            case FinalizedInput(provenance=provenance, text=text):
                self._entries.append(
                    ContextEntry(ContextEntryKind.INPUT, provenance, text)
                )

            case AcceptedOutput(provenance=provenance, text=text):
                self._entries.append(
                    ContextEntry(ContextEntryKind.OUTPUT, provenance, text)
                )

            case PartialMaterial() | CancelledMaterial() | StaleMaterial():
                pass

        return self.snapshot

    def reset(self) -> TransientContextSnapshot:
        """函数契约说明.

        功能: 执行 reset 的同步逻辑,并协调 clear。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `TransientContextSnapshot`。
        """
        self._entries.clear()

        self._generation += 1

        return self.snapshot

    def compose(
        self,
        model_id: ModelId,
        policy: ContextBudgetPolicy,
    ) -> ContextComposition:
        """函数契约说明.

        功能: 执行 compose 的同步逻辑,并协调
        compose_context, budget_for。
        参数: self 表示当前实例。 model_id:
        ModelId。 必填。 policy:
        ContextBudgetPolicy。 必填。
        契约: 同步调用。 返回
        `ContextComposition`。
        """
        return compose_context(self.snapshot, policy.budget_for(model_id))

    def _ensure_session(self, provenance: ContextProvenance) -> None:
        """函数契约说明.

        功能: 执行 _ensure_session 的同步逻辑,并协调
        ContextSessionMismatchError。
        参数: self 表示当前实例。 provenance:
        ContextProvenance。 必填。
        契约: 同步调用。 返回 `None`。 可能抛出
        ContextSessionMismatchError。
        """
        if provenance.session_id != self._session_id:
            raise ContextSessionMismatchError(
                expected_session_id=self._session_id,
                actual_session_id=provenance.session_id,
            )


def compose_context(
    snapshot: TransientContextSnapshot,
    budget: ModelContextBudget,
) -> ContextComposition:
    """函数契约说明.

    功能: 执行 compose_context 的同步逻辑,并协调
    _content_tokens,
    _retain_newest_entry_indexes, tuple,
    ContextDigest。
    参数: snapshot:
    TransientContextSnapshot。 必填。
    budget: ModelContextBudget。 必填。
    契约: 同步调用。 返回 `ContextComposition`。
    """
    total_tokens = _content_tokens(snapshot.entries)

    if total_tokens <= budget.input_tokens:
        return ContextComposition(snapshot, snapshot.entries, (), total_tokens)

    retained_indexes = _retain_newest_entry_indexes(
        snapshot.entries,
        budget.input_tokens,
    )

    retained_entries = tuple(
        entry
        for index, entry in enumerate(snapshot.entries)
        if index in retained_indexes
    )

    compacted_entries = tuple(
        entry
        for index, entry in enumerate(snapshot.entries)
        if index not in retained_indexes
    )

    digest = ContextDigest(
        source_provenances=tuple(entry.provenance for entry in compacted_entries),
        content_hash=_content_hash(compacted_entries),
    )

    return ContextComposition(
        snapshot,
        retained_entries,
        (digest,),
        TokenBudget(_content_tokens(retained_entries) + 1),
    )


def _retain_newest_entry_indexes(
    entries: Sequence[ContextEntry],
    budget: TokenBudget,
) -> frozenset[int]:
    """函数契约说明.

    功能: 执行 _retain_newest_entry_indexes
    的同步逻辑,并协调 set, range, frozenset,
    int。
    参数: entries: Sequence[ContextEntry]。
    必填。 budget: TokenBudget。 必填。
    契约: 同步调用。 返回 `frozenset[int]`。
    """
    remaining_tokens = int(budget) - 1

    retained_indexes: set[int] = set()

    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]

        entry_tokens = _content_tokens((entry,))

        if entry_tokens <= remaining_tokens:
            retained_indexes.add(index)

            remaining_tokens -= entry_tokens

    return frozenset(retained_indexes)


def _content_tokens(entries: Sequence[ContextEntry]) -> TokenBudget:
    """函数契约说明.

    功能: 执行 _content_tokens 的同步逻辑,并协调
    TokenBudget, sum, len, split。
    参数: entries: Sequence[ContextEntry]。
    必填。
    契约: 同步调用。 返回 `TokenBudget`。
    """
    return TokenBudget(sum(len(entry.text.split()) for entry in entries))


def _content_hash(entries: Sequence[ContextEntry]) -> str:
    """函数契约说明.

    功能: 执行 _content_hash 的同步逻辑,并协调 join,
    hexdigest, sha256, encode。
    参数: entries: Sequence[ContextEntry]。
    必填。
    契约: 同步调用。 返回 `str`。
    """
    canonical = "\x1e".join(
        "\x1f".join(
            (
                entry.kind.value,
                entry.provenance.session_id,
                entry.provenance.turn_id,
                entry.provenance.segment_id,
                str(entry.provenance.sequence),
                entry.provenance.source_id,
                entry.text,
            )
        )
        for entry in entries
    )

    return sha256(canonical.encode()).hexdigest()
