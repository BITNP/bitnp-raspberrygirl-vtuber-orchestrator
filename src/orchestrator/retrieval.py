"""Immutable, attributed retrieval results for Orchestrator prompt composition."""

from dataclasses import dataclass
from typing import Final, Protocol

from orchestrator.modes import AnswerCandidate
from orchestrator.state_snapshots import CorpusRevision, IndexRevision

_FIXTURE_CORPUS_REVISION: Final = CorpusRevision(1)
_FIXTURE_INDEX_REVISION: Final = IndexRevision(1)
_ATTRIBUTION_MISMATCH: Final = "knowledge_attribution_mismatch"


@dataclass(frozen=True, slots=True)
class KnowledgeRef:
    """Untrusted knowledge-base context returned to prompt construction."""

    ref_id: str
    title: str
    text: str
    corpus_id: str = "fixture-corpus"
    corpus_revision: CorpusRevision = _FIXTURE_CORPUS_REVISION
    index_id: str = "fixture-index"
    index_revision: IndexRevision = _FIXTURE_INDEX_REVISION


@dataclass(frozen=True, slots=True)
class RetrievalSnapshot:
    """Immutable corpus and index identity captured for one retrieval result."""

    corpus_id: str
    corpus_revision: CorpusRevision
    index_id: str
    index_revision: IndexRevision


class KnowledgeAttributionError(ValueError):
    """Raised when a result cannot be replayed against one immutable snapshot."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Attributed untrusted references from one immutable corpus/index snapshot."""

    snapshot: RetrievalSnapshot
    refs: tuple[KnowledgeRef, ...]

    def __post_init__(self) -> None:
        """Reject references that cannot replay against the captured snapshot."""
        for ref in self.refs:
            if _attribution(ref) != self.snapshot:
                raise KnowledgeAttributionError(_ATTRIBUTION_MISMATCH)


class RetrievalProvider(Protocol):
    """Optional Orchestrator retrieval boundary."""

    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult:
        """Return context refs for an answer candidate."""
        ...


class VersionedRetrievalProvider(RetrievalProvider, Protocol):
    """Retrieval boundary that exposes the immutable snapshot it will return."""

    @property
    def snapshot(self) -> RetrievalSnapshot:
        """Return the current immutable corpus/index attribution."""
        ...


@dataclass(frozen=True, slots=True)
class RetrievalFixtureProvider:
    """Deterministic fixture provider; no vector DB or ingestion pipeline."""

    refs: tuple[KnowledgeRef, ...]

    @property
    def snapshot(self) -> RetrievalSnapshot:
        """Return the single immutable fixture attribution shared by all refs."""
        if len(self.refs) == 0:
            return RetrievalSnapshot(
                "fixture-corpus",
                _FIXTURE_CORPUS_REVISION,
                "fixture-index",
                _FIXTURE_INDEX_REVISION,
            )
        return _attribution(self.refs[0])

    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult:
        """Return fixture refs unchanged for deterministic tests."""
        _ = candidate
        return RetrievalResult(snapshot=self.snapshot, refs=self.refs)


def _attribution(ref: KnowledgeRef) -> RetrievalSnapshot:
    """Extract the immutable corpus/index identity carried by one reference."""
    return RetrievalSnapshot(
        ref.corpus_id,
        ref.corpus_revision,
        ref.index_id,
        ref.index_revision,
    )
