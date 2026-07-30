"""Human-readable atomic persistence for policy-approved ordinary memory."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final, override

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.json_boundary import JsonValue, parse_json_value
from orchestrator.memory import (
    MemoryEntry,
    MemoryKey,
    MemoryProvenance,
    MemorySource,
    MutableMemorySnapshot,
)
from orchestrator.state_snapshots import (
    ConsentRevision,
    MemoryRevision,
    ProfileRevision,
)


class MemoryStore(Protocol):
    """Durable boundary for revisioned mutable-memory snapshots."""

    def save(self, snapshot: MutableMemorySnapshot) -> None:
        """Persist one complete accepted memory snapshot atomically."""
        ...

    def load(self, session_id: SessionId) -> MutableMemorySnapshot | None:
        """Load one accepted session snapshot when durable state exists."""
        ...


@dataclass(frozen=True, slots=True)
class MemoryStoreBoundaryError(ValueError):
    """Raised when persisted ordinary memory fails typed boundary parsing."""

    field: str

    @override
    def __str__(self) -> str:
        return f"invalid memory record: {self.field}"


@final
class JsonMemoryStore:
    """Persist approved preferences as an inspectable JSON document."""

    def __init__(self, path: Path) -> None:
        """Bind the store to its single scheduler-owned document path."""
        self._path: Path = path

    def save(self, snapshot: MutableMemorySnapshot) -> None:
        """Atomically replace the document with the current accepted revision."""
        document = {
            "revision": int(snapshot.revision),
            "preferences": [
                {
                    "key": entry.key,
                    "value": entry.value,
                    "source": entry.provenance.source,
                    "trace_id": entry.provenance.trace_id,
                    "session_id": entry.provenance.session_id,
                    "turn_id": entry.provenance.turn_id,
                    "evidence_id": entry.provenance.evidence_id,
                }
                for entry in snapshot.entries
            ],
        }
        _ = self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        _ = temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _ = temporary.replace(self._path)

    def load(self, session_id: SessionId) -> MutableMemorySnapshot | None:
        """Load an atomically persisted snapshot only for its originating session."""
        if not self._path.exists():
            return None
        document = _object(parse_json_value(self._path.read_text(encoding="utf-8")))
        preferences = _array(document, "preferences")
        entries = tuple(
            MemoryEntry(
                key=MemoryKey(_text(_object(item), "key")),
                value=_text(_object(item), "value"),
                provenance=MemoryProvenance(
                    source=MemorySource(_text(_object(item), "source")),
                    trace_id=TraceId(_text(_object(item), "trace_id")),
                    session_id=SessionId(_text(_object(item), "session_id")),
                    turn_id=TurnId(_text(_object(item), "turn_id")),
                    evidence_id=_text(_object(item), "evidence_id"),
                ),
            )
            for item in preferences
        )
        if any(entry.provenance.session_id != session_id for entry in entries):
            return None
        return MutableMemorySnapshot(
            revision=MemoryRevision(_integer(document, "revision")),
            entries=entries,
            profile_revision=ProfileRevision(0),
            consent_revision=ConsentRevision(0),
        )


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        field = "$"
        raise MemoryStoreBoundaryError(field)
    return value


def _array(document: dict[str, JsonValue], field: str) -> list[JsonValue]:
    value = document.get(field)
    if not isinstance(value, list):
        raise MemoryStoreBoundaryError(field)
    return value


def _text(document: dict[str, JsonValue], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str):
        raise MemoryStoreBoundaryError(field)
    return value


def _integer(document: dict[str, JsonValue], field: str) -> int:
    value = document.get(field)
    if type(value) is not int:
        raise MemoryStoreBoundaryError(field)
    return value
