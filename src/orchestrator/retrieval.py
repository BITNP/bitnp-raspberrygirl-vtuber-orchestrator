
from dataclasses import dataclass
from typing import Final, Protocol

from orchestrator.modes import AnswerCandidate
from orchestrator.state_snapshots import CorpusRevision, IndexRevision

_FIXTURE_CORPUS_REVISION: Final = CorpusRevision(1)

_FIXTURE_INDEX_REVISION: Final = IndexRevision(1)

_ATTRIBUTION_MISMATCH: Final = "knowledge_attribution_mismatch"


@dataclass(frozen=True, slots=True)
class KnowledgeRef:

    ref_id: str

    title: str

    text: str

    corpus_id: str = "fixture-corpus"

    corpus_revision: CorpusRevision = _FIXTURE_CORPUS_REVISION

    index_id: str = "fixture-index"

    index_revision: IndexRevision = _FIXTURE_INDEX_REVISION


@dataclass(frozen=True, slots=True)
class RetrievalSnapshot:

    corpus_id: str

    corpus_revision: CorpusRevision

    index_id: str

    index_revision: IndexRevision


class KnowledgeAttributionError(ValueError):
    ...


@dataclass(frozen=True, slots=True)
class RetrievalResult:

    snapshot: RetrievalSnapshot

    refs: tuple[KnowledgeRef, ...]

    def __post_init__(self) -> None:
        for ref in self.refs:
            if _attribution(ref) != self.snapshot:
                raise KnowledgeAttributionError(_ATTRIBUTION_MISMATCH)


class RetrievalProvider(Protocol):

    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult:
        ...


class VersionedRetrievalProvider(RetrievalProvider, Protocol):

    @property
    def snapshot(self) -> RetrievalSnapshot:
        ...


@dataclass(frozen=True, slots=True)
class RetrievalFixtureProvider:

    refs: tuple[KnowledgeRef, ...]

    @property
    def snapshot(self) -> RetrievalSnapshot:
        if len(self.refs) == 0:
            return RetrievalSnapshot(
                "fixture-corpus",
                _FIXTURE_CORPUS_REVISION,
                "fixture-index",
                _FIXTURE_INDEX_REVISION,
            )

        return _attribution(self.refs[0])

    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult:
        _ = candidate

        return RetrievalResult(snapshot=self.snapshot, refs=self.refs)


def _attribution(ref: KnowledgeRef) -> RetrievalSnapshot:
    return RetrievalSnapshot(
        ref.corpus_id,
        ref.corpus_revision,
        ref.index_id,
        ref.index_revision,
    )
