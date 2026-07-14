"""Deterministic Orchestrator retrieval fixtures."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from orchestrator.modes import AnswerCandidate


@dataclass(frozen=True, slots=True)
class KnowledgeRef:
    """Untrusted knowledge-base context returned to prompt construction."""

    ref_id: str
    title: str
    text: str


class RetrievalProvider(Protocol):
    """Optional Orchestrator retrieval boundary."""

    def retrieve(self, candidate: AnswerCandidate) -> Sequence[KnowledgeRef]:
        """Return context refs for an answer candidate."""
        ...


@dataclass(frozen=True, slots=True)
class RetrievalFixtureProvider:
    """Deterministic fixture provider; no vector DB or ingestion pipeline."""

    refs: tuple[KnowledgeRef, ...]

    def retrieve(self, candidate: AnswerCandidate) -> Sequence[KnowledgeRef]:
        """Return fixture refs unchanged for deterministic tests."""
        _ = candidate
        return self.refs
