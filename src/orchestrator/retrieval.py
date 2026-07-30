"""模块契约说明.

职责: 提供 orchestrator.retrieval
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from typing import Final, Protocol

from orchestrator.modes import AnswerCandidate
from orchestrator.state_snapshots import CorpusRevision, IndexRevision

_FIXTURE_CORPUS_REVISION: Final = CorpusRevision(1)

_FIXTURE_INDEX_REVISION: Final = IndexRevision(1)

_ATTRIBUTION_MISMATCH: Final = "knowledge_attribution_mismatch"


@dataclass(frozen=True, slots=True)
class KnowledgeRef:
    """类契约说明.

    职责: 保存 KnowledgeRef
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: ref_id、title、text、corpus_id、
    corpus_revision、index_id。
    """

    ref_id: str

    title: str

    text: str

    corpus_id: str = "fixture-corpus"

    corpus_revision: CorpusRevision = _FIXTURE_CORPUS_REVISION

    index_id: str = "fixture-index"

    index_revision: IndexRevision = _FIXTURE_INDEX_REVISION


@dataclass(frozen=True, slots=True)
class RetrievalSnapshot:
    """类契约说明.

    职责: 保存 RetrievalSnapshot
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: corpus_id、corpus_revision、in
    dex_id、index_revision。
    """

    corpus_id: str

    corpus_revision: CorpusRevision

    index_id: str

    index_revision: IndexRevision


class KnowledgeAttributionError(ValueError):
    """类契约说明.

    职责: 表示 KnowledgeAttributionError
    错误类别,并携带调用方可处理的失败信息。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """类契约说明.

    职责: 保存 RetrievalResult
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: snapshot、refs。 方法:
    __post_init__。
    """

    snapshot: RetrievalSnapshot

    refs: tuple[KnowledgeRef, ...]

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化 RetrievalResult
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。 可能抛出
        KnowledgeAttributionError。
        """
        for ref in self.refs:
            if _attribution(ref) != self.snapshot:
                raise KnowledgeAttributionError(_ATTRIBUTION_MISMATCH)


class RetrievalProvider(Protocol):
    """类契约说明.

    职责: 声明 RetrievalProvider
    协议接口,约束实现方必须提供的行为。
    契约: 方法: retrieve。
    """

    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult:
        """函数契约说明.

        功能: 执行 retrieve 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 candidate:
        AnswerCandidate。 必填。
        契约: 同步调用。 返回 `RetrievalResult`。
        """
        ...


class VersionedRetrievalProvider(RetrievalProvider, Protocol):
    """类契约说明.

    职责: 声明 VersionedRetrievalProvider
    协议接口,约束实现方必须提供的行为。
    契约: 方法: snapshot。
    """

    @property
    def snapshot(self) -> RetrievalSnapshot:
        """函数契约说明.

        功能: 执行 snapshot 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `RetrievalSnapshot`。
        """
        ...


@dataclass(frozen=True, slots=True)
class RetrievalFixtureProvider:
    """类契约说明.

    职责: 保存 RetrievalFixtureProvider
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: refs。 方法: snapshot、retrieve。
    """

    refs: tuple[KnowledgeRef, ...]

    @property
    def snapshot(self) -> RetrievalSnapshot:
        """函数契约说明.

        功能: 执行 snapshot 的同步逻辑,并协调
        _attribution, len,
        RetrievalSnapshot。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `RetrievalSnapshot`。
        """
        if len(self.refs) == 0:
            return RetrievalSnapshot(
                "fixture-corpus",
                _FIXTURE_CORPUS_REVISION,
                "fixture-index",
                _FIXTURE_INDEX_REVISION,
            )

        return _attribution(self.refs[0])

    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult:
        """函数契约说明.

        功能: 执行 retrieve 的同步逻辑,并协调
        RetrievalResult。
        参数: self 表示当前实例。 candidate:
        AnswerCandidate。 必填。
        契约: 同步调用。 返回 `RetrievalResult`。
        """
        _ = candidate

        return RetrievalResult(snapshot=self.snapshot, refs=self.refs)


def _attribution(ref: KnowledgeRef) -> RetrievalSnapshot:
    """函数契约说明.

    功能: 执行 _attribution 的同步逻辑,并协调
    RetrievalSnapshot。
    参数: ref: KnowledgeRef。 必填。
    契约: 同步调用。 返回 `RetrievalSnapshot`。
    """
    return RetrievalSnapshot(
        ref.corpus_id,
        ref.corpus_revision,
        ref.index_id,
        ref.index_revision,
    )
