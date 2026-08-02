from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, Protocol, cast, final

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

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


@dataclass(frozen=True, slots=True)
class ReadonlyCorpusConfig:
    directory: Path
    corpus_id: str = "local-corpus"
    index_id: str = "llama-index"
    top_k: int = 4


@final
class ReadonlyLlamaIndexProvider:
    """Startup-built local corpus; it never writes source files or the index.

    The default embedding is deterministic only for mock/test deployments.
    Production supplies the provisioned local ONNX embedding implementation to
    ``embed_model``.  Index persistence is intentionally disabled: a corpus
    change only takes effect after service restart and yields a new revision.
    """

    def __init__(
        self,
        config: ReadonlyCorpusConfig,
        *,
        embed_model: object | None = None,
    ) -> None:
        directory = config.directory.resolve()
        if not directory.is_dir():
            message = "knowledge corpus directory does not exist"
            raise ValueError(message)
        paths = tuple(
            sorted(
                path
                for extension in ("*.md", "*.txt", "*.json")
                for path in directory.rglob(extension)
                if path.is_file()
            )
        )
        digest = sha256()
        for path in paths:
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        revision = int.from_bytes(digest.digest()[:8], "big")
        self._config: ReadonlyCorpusConfig = config
        self._directory: Path = directory
        self._snapshot: RetrievalSnapshot = RetrievalSnapshot(
            config.corpus_id,
            CorpusRevision(revision),
            config.index_id,
            IndexRevision(revision),
        )
        documents = SimpleDirectoryReader(
            input_files=list(paths),
            required_exts=[".md", ".txt", ".json"],
        ).load_data()
        model = MockEmbedding(embed_dim=384) if embed_model is None else embed_model
        self._index: VectorStoreIndex = VectorStoreIndex.from_documents(
            documents, embed_model=model
        )

    @property
    def snapshot(self) -> RetrievalSnapshot:
        return self._snapshot

    def retrieve(self, candidate: AnswerCandidate) -> RetrievalResult:
        retriever = self._index.as_retriever(similarity_top_k=self._config.top_k)
        nodes = retriever.retrieve(candidate.input.text)
        refs = tuple(
            KnowledgeRef(
                ref_id=f"{index}:{node.node.metadata.get('file_path', '')}",
                title=cast(
                    "str", node.node.metadata.get("file_name", "local knowledge")
                ),
                text=node.node.get_content(),
                corpus_id=self._snapshot.corpus_id,
                corpus_revision=self._snapshot.corpus_revision,
                index_id=self._snapshot.index_id,
                index_revision=self._snapshot.index_revision,
            )
            for index, node in enumerate(nodes)
        )
        return RetrievalResult(self._snapshot, refs)


def _attribution(ref: KnowledgeRef) -> RetrievalSnapshot:
    return RetrievalSnapshot(
        ref.corpus_id,
        ref.corpus_revision,
        ref.index_id,
        ref.index_revision,
    )
