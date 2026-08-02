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

MAX_CONTEXT_SUMMARY_CHARS = 4_000


@dataclass(frozen=True, slots=True)
class ContextProvenance:
    session_id: SessionId

    turn_id: TurnId

    segment_id: SegmentId

    sequence: ContextSequence

    source_id: ContextSourceId


@dataclass(frozen=True, slots=True)
class FinalizedInput:
    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class AcceptedOutput:
    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """A successful, source-labelled tool observation from an accepted turn."""

    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class PartialMaterial:
    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class CancelledMaterial:
    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class StaleMaterial:
    provenance: ContextProvenance

    text: str


type ContextMaterial = (
    FinalizedInput
    | AcceptedOutput
    | ToolObservation
    | PartialMaterial
    | CancelledMaterial
    | StaleMaterial
)


@unique
class ContextEntryKind(StrEnum):
    INPUT = "input"

    OUTPUT = "output"

    OBSERVATION = "observation"


@dataclass(frozen=True, slots=True)
class ContextEntry:
    kind: ContextEntryKind

    provenance: ContextProvenance

    text: str


@dataclass(frozen=True, slots=True)
class TransientContextSnapshot:
    session_id: SessionId

    generation: int

    entries: tuple[ContextEntry, ...]

    summary: str = ""


@dataclass(frozen=True, slots=True)
class ModelContextBudget:
    input_tokens: TokenBudget

    def __post_init__(self) -> None:
        if self.input_tokens <= 0:
            raise InvalidContextBudgetError(input_tokens=self.input_tokens)


class ContextBudgetPolicy(Protocol):
    def budget_for(self, model_id: ModelId) -> ModelContextBudget: ...


@dataclass(frozen=True, slots=True)
class StaticContextBudgetPolicy:
    model_id: ModelId

    budget: ModelContextBudget

    def budget_for(self, model_id: ModelId) -> ModelContextBudget:
        if model_id != self.model_id:
            raise ModelBudgetUnavailableError(model_id=model_id)

        return self.budget


@dataclass(frozen=True, slots=True)
class ContextDigest:
    source_provenances: tuple[ContextProvenance, ...]

    content_hash: str


@dataclass(frozen=True, slots=True)
class ContextComposition:
    snapshot: TransientContextSnapshot

    entries: tuple[ContextEntry, ...]

    digests: tuple[ContextDigest, ...]

    content_token_count: TokenBudget


@dataclass(frozen=True, slots=True)
class ContextSessionMismatchError(Exception):
    expected_session_id: SessionId

    actual_session_id: SessionId

    @override
    def __str__(self) -> str:
        return "transient context session mismatch"


@dataclass(frozen=True, slots=True)
class InvalidContextBudgetError(Exception):
    input_tokens: TokenBudget

    @override
    def __str__(self) -> str:
        return "transient context budget must be positive"


@dataclass(frozen=True, slots=True)
class ModelBudgetUnavailableError(Exception):
    model_id: ModelId

    @override
    def __str__(self) -> str:
        return "transient context model budget is unavailable"


@dataclass(frozen=True, slots=True)
class ContextCompactionError(Exception):
    def __str__(self) -> str:
        return "transient context compaction source is stale or invalid"


@final
class TransientContext:
    def __init__(self, *, session_id: SessionId) -> None:
        self._session_id = session_id

        self._generation = 0

        self._entries: list[ContextEntry] = []

        self._summary = ""

    @property
    def snapshot(self) -> TransientContextSnapshot:
        return TransientContextSnapshot(
            session_id=self._session_id,
            generation=self._generation,
            entries=tuple(self._entries),
            summary=self._summary,
        )

    def consider(self, material: ContextMaterial) -> TransientContextSnapshot:
        self._ensure_session(material.provenance)

        match material:
            case FinalizedInput(provenance=provenance, text=text):
                self._entries.append(
                    ContextEntry(ContextEntryKind.INPUT, provenance, text)
                )
                self._generation += 1

            case AcceptedOutput(provenance=provenance, text=text):
                self._entries.append(
                    ContextEntry(ContextEntryKind.OUTPUT, provenance, text)
                )
                self._generation += 1

            case ToolObservation(provenance=provenance, text=text):
                self._entries.append(
                    ContextEntry(ContextEntryKind.OBSERVATION, provenance, text)
                )
                self._generation += 1

            case PartialMaterial() | CancelledMaterial() | StaleMaterial():
                pass

        return self.snapshot

    def reset(self) -> TransientContextSnapshot:
        self._entries.clear()

        self._summary = ""

        self._generation += 1

        return self.snapshot

    def compact(
        self,
        composition: ContextComposition,
        *,
        summary: str,
    ) -> TransientContextSnapshot:
        """Atomically retain recent raw material and replace compacted source.

        The caller must pass the exact composition used by Brain.  This makes
        late summaries harmless after another accepted context write.
        """
        if composition.snapshot != self.snapshot or not composition.digests:
            raise ContextCompactionError
        normalized = summary.strip()
        if normalized == "" or len(normalized) > MAX_CONTEXT_SUMMARY_CHARS:
            raise ContextCompactionError
        self._entries = list(composition.entries)
        self._summary = normalized
        self._generation += 1
        return self.snapshot

    def compose(
        self,
        model_id: ModelId,
        policy: ContextBudgetPolicy,
    ) -> ContextComposition:
        return compose_context(self.snapshot, policy.budget_for(model_id))

    def _ensure_session(self, provenance: ContextProvenance) -> None:
        if provenance.session_id != self._session_id:
            raise ContextSessionMismatchError(
                expected_session_id=self._session_id,
                actual_session_id=provenance.session_id,
            )


def compose_context(
    snapshot: TransientContextSnapshot,
    budget: ModelContextBudget,
) -> ContextComposition:
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
    return TokenBudget(sum(len(entry.text.split()) for entry in entries))


def _content_hash(entries: Sequence[ContextEntry]) -> str:
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
