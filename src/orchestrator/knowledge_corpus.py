"""Bounded LlamaIndex ingestion and cancellable offline BM25 ranking."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, cast, final

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from orchestrator.json_boundary import parse_json_value

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_MAX_FILE_BYTES = 1_048_576
_MAX_CORPUS_BYTES = 16_777_216
_MAX_FILES = 256
_MAX_ENTRIES = 4096
_MAX_CHUNKS = 8192
_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]+|[a-z0-9_]+")
_INDEX_VERSION = (
    b"bm25-zh-v1;unicode-char;zh-sentence;chunk=256;overlap=32;k1=1.2;b=0.75"
)


class CorpusLoadError(ValueError):
    """A local corpus violates the controlled ingestion contract."""


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    source: str
    title: str
    text: str


def _tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(unicodedata.normalize("NFKC", text).lower()):
        word = match.group()
        if "\u3400" <= word[0] <= "\u9fff":
            tokens.extend(word)
            tokens.extend(word[index : index + 2] for index in range(len(word) - 1))
        else:
            tokens.append(word)
    return tuple(tokens)


def _sentences(text: str) -> list[str]:
    """Keep punctuation while avoiding NLTK's lazy network downloads."""
    return [
        match.group()
        for match in re.finditer(
            r"[^。\uff01\uff1f.!?\n]+[。\uff01\uff1f.!?\n]*|[。\uff01\uff1f.!?\n]+",
            text,
        )
    ]


def _files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        message = "knowledge directory does not exist"
        raise CorpusLoadError(message)
    paths: list[Path] = []
    entries = 0
    for root, directories, filenames in directory.walk(follow_symlinks=False):
        for name in sorted([*directories, *filenames]):
            entries += 1
            path = root / name
            if entries > _MAX_ENTRIES or path.is_symlink():
                message = "knowledge tree exceeds limits or contains links"
                raise CorpusLoadError(message)
            if name in filenames and path.suffix in {".md", ".txt", ".json"}:
                if not path.is_file() or path.resolve().parent != root.resolve():
                    message = "knowledge path is not a controlled file"
                    raise CorpusLoadError(message)
                paths.append(path)
                if len(paths) > _MAX_FILES:
                    message = "too many knowledge files"
                    raise CorpusLoadError(message)
    return tuple(sorted(paths))


@final
class KeywordCorpus:
    """Own source bytes, LlamaIndex chunks, provenance, and the ranking index."""

    def __init__(self, directory: Path) -> None:
        directory = directory.resolve()
        digest = sha256()
        documents: list[Document] = []
        total_bytes = 0
        for path in _files(directory):
            with path.open("rb") as stream:
                payload = stream.read(_MAX_FILE_BYTES + 1)
            total_bytes += len(payload)
            if len(payload) > _MAX_FILE_BYTES or total_bytes > _MAX_CORPUS_BYTES:
                message = "knowledge bytes exceed limits"
                raise CorpusLoadError(message)
            text = payload.decode("utf-8")
            if path.suffix == ".json":
                _ = parse_json_value(text)
            source = path.relative_to(directory).as_posix()
            digest.update(source.encode() + b"\0" + payload + b"\0")
            if text.strip():
                documents.append(Document(text=text, metadata={"source": source}))
        self.revision = int.from_bytes(digest.digest()[:8], "big")
        self.index_revision = int.from_bytes(
            sha256(digest.digest() + _INDEX_VERSION).digest()[:8], "big"
        )
        splitter = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=32,
            include_metadata=False,
            tokenizer=list,
            chunking_tokenizer_fn=_sentences,
        )
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            source = cast("str", document.metadata["source"])
            for index, text in enumerate(splitter.split_text(document.text)):
                chunks.append(KnowledgeChunk(f"{source}#{index}", source, text))
                if len(chunks) > _MAX_CHUNKS:
                    message = "too many knowledge chunks"
                    raise CorpusLoadError(message)
        self.chunks = tuple(chunks)
        self._terms = tuple(Counter(_tokens(chunk.text)) for chunk in self.chunks)
        self._lengths = tuple(sum(terms.values()) for terms in self._terms)
        self._average = sum(self._lengths) / max(1, len(self._lengths)) or 1.0
        self._frequencies = Counter(term for terms in self._terms for term in terms)
        _LOGGER.debug(
            "knowledge_loaded corpus=%d index=%d files=%d chunks=%d bytes=%d digest=sha256:%s outcome=ready",  # noqa: E501
            self.revision,
            self.index_revision,
            len(documents),
            len(chunks),
            total_bytes,
            digest.hexdigest(),
        )

    def _scores(self, query: str) -> Iterator[tuple[float, int]]:
        terms = set(_tokens(query[:16_384]))
        count = len(self.chunks)
        for index, frequencies in enumerate(self._terms):
            score = 0.0
            for term in sorted(terms.intersection(frequencies)):
                inverse = math.log(
                    1
                    + (count - self._frequencies[term] + 0.5)
                    / (self._frequencies[term] + 0.5)
                )
                frequency = frequencies[term]
                score += (
                    inverse
                    * frequency
                    * 2.2
                    / (
                        frequency
                        + 1.2 * (0.25 + 0.75 * self._lengths[index] / self._average)
                    )
                )
            yield score, index

    @staticmethod
    def _best(scores: list[tuple[float, int]], top_k: int) -> tuple[int, ...]:
        return tuple(
            index
            for score, index in sorted(scores, key=lambda item: (-item[0], item[1]))[
                :top_k
            ]
            if score > 0
        )

    def search(self, query: str, top_k: int) -> tuple[int, ...]:
        return self._best(list(self._scores(query)), top_k)

    async def search_async(self, query: str, top_k: int) -> tuple[int, ...]:
        scores: list[tuple[float, int]] = []
        for score, index in self._scores(query):
            if index % 64 == 0:
                await asyncio.sleep(0)
            scores.append((score, index))
        return self._best(scores, top_k)
