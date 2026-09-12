from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, final

from orchestrator.knowledge_corpus import KeywordCorpus
from orchestrator.state_snapshots import CorpusRevision, IndexRevision

if TYPE_CHECKING:
    from collections.abc import Mapping

    from orchestrator.modes import AnswerCandidate

_FIXTURE_CORPUS_REVISION: Final = CorpusRevision(1)

_FIXTURE_INDEX_REVISION: Final = IndexRevision(1)

_MAX_TOP_K = 8

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


class KnowledgeAttributionError(ValueError): ...


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    snapshot: RetrievalSnapshot

    refs: tuple[KnowledgeRef, ...]

    def __post_init__(self) -> None:
        for ref in self.refs:
            if _attribution(ref) != self.snapshot:
                raise KnowledgeAttributionError(_ATTRIBUTION_MISMATCH)


class RetrievalProvider(Protocol):
    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult: ...


class VersionedRetrievalProvider(RetrievalProvider, Protocol):
    @property
    def snapshot(self) -> RetrievalSnapshot: ...

    async def retrieve_async(self, candidate: AnswerCandidate) -> RetrievalResult: ...


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

    async def retrieve_async(self, candidate: AnswerCandidate) -> RetrievalResult:
        return self.retrieve(candidate)


@dataclass(frozen=True, slots=True)
class ReadonlyCorpusConfig:
    directory: Path
    corpus_id: str = "local-corpus"
    index_id: str = "llama-index-bm25-zh-v1"
    top_k: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= _MAX_TOP_K or not self.corpus_id or not self.index_id:
            message = "invalid knowledge configuration"
            raise ValueError(message)


@final
class ReadonlyLlamaIndexProvider:
    """An immutable startup snapshot with offline Chinese BM25 retrieval."""

    def __init__(self, config: ReadonlyCorpusConfig) -> None:
        self._config = config
        self._corpus = KeywordCorpus(config.directory)
        self._snapshot = RetrievalSnapshot(
            config.corpus_id,
            CorpusRevision(self._corpus.revision),
            config.index_id,
            IndexRevision(self._corpus.index_revision),
        )

    @property
    def snapshot(self) -> RetrievalSnapshot:
        return self._snapshot

    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult:
        return self._result(
            self._corpus.search(candidate.input.text, self._config.top_k)
        )

    async def retrieve_async(self, candidate: AnswerCandidate) -> RetrievalResult:
        hits = await self._corpus.search_async(candidate.input.text, self._config.top_k)
        return self._result(hits)

    def _result(self, hits: tuple[int, ...]) -> RetrievalResult:
        return RetrievalResult(
            self._snapshot,
            tuple(
                KnowledgeRef(
                    ref_id=self._corpus.chunks[index].source,
                    title=self._corpus.chunks[index].title,
                    text=self._corpus.chunks[index].text,
                    corpus_id=self._snapshot.corpus_id,
                    corpus_revision=self._snapshot.corpus_revision,
                    index_id=self._snapshot.index_id,
                    index_revision=self._snapshot.index_revision,
                )
                for index in hits
            ),
        )


def load_knowledge_provider(env: Mapping[str, str]) -> VersionedRetrievalProvider:
    """Called once by the process composition root, before accepting sessions."""
    directory = env.get("ORCHESTRATOR_KNOWLEDGE_DIR", "").strip()
    if not directory:
        return RetrievalFixtureProvider(refs=())
    return ReadonlyLlamaIndexProvider(
        ReadonlyCorpusConfig(
            Path(directory),
            top_k=int(env.get("ORCHESTRATOR_KNOWLEDGE_TOP_K", "4")),
        )
    )


def _attribution(ref: KnowledgeRef) -> RetrievalSnapshot:
    return RetrievalSnapshot(
        ref.corpus_id,
        ref.corpus_revision,
        ref.index_id,
        ref.index_revision,
    )
