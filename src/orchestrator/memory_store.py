import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, final, override

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.json_boundary import JsonBoundaryError, JsonValue, parse_json_value
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

_SESSION_ID_FIELD: Final = "session_id"

_MARKDOWN_STATE_OPEN: Final = "<!-- bitnp-memory-state\n"

_MARKDOWN_STATE_CLOSE: Final = "\n-->"


class MemoryStore(Protocol):
    def save(self, snapshot: MutableMemorySnapshot) -> None: ...

    def load(self, session_id: SessionId) -> MutableMemorySnapshot | None: ...


@dataclass(frozen=True, slots=True)
class MemoryStoreBoundaryError(ValueError):
    field: str

    @override
    def __str__(self) -> str:
        return f"invalid memory record: {self.field}"


@final
class JsonMemoryStore:
    def __init__(self, path: Path) -> None:
        self._path: Path = path

        self._session_id: SessionId | None = None

    def save(self, snapshot: MutableMemorySnapshot) -> None:
        if self._session_id is None:
            raise MemoryStoreBoundaryError(_SESSION_ID_FIELD)

        document = {
            "session_id": str(self._session_id),
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
        if self._session_id is None:
            self._session_id = session_id

        elif self._session_id != session_id:
            raise MemoryStoreBoundaryError(_SESSION_ID_FIELD)

        if not self._path.exists():
            return None

        try:
            document = _object(parse_json_value(self._path.read_text(encoding="utf-8")))

        except JsonBoundaryError as error:
            raise MemoryStoreBoundaryError(error.field_name) from error

        return _snapshot_from_document(document, session_id)


@final
class MarkdownMemoryStore:
    """Session-isolated human-readable ``memory.md`` with atomic persistence."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._session_id: SessionId | None = None

    def save(self, snapshot: MutableMemorySnapshot) -> None:
        if self._session_id is None:
            raise MemoryStoreBoundaryError(_SESSION_ID_FIELD)
        document = _document_for_snapshot(snapshot, self._session_id)
        rendered_entries = "\n".join(
            "\n".join(
                (
                    f"## {entry.key}",
                    f"- 值: {entry.value}",
                    f"- 来源轮次: {entry.provenance.turn_id}",
                    f"- 证据: {entry.provenance.evidence_id}",
                )
            )
            for entry in snapshot.entries
        )
        state = json.dumps(document, ensure_ascii=False, sort_keys=True)
        rendered = (
            "# 会话记忆\n\n"
            f"会话: {self._session_id}\n\n"
            f"版本: {snapshot.revision}\n\n"
            f"{rendered_entries}\n\n"
            f"{_MARKDOWN_STATE_OPEN}{state}{_MARKDOWN_STATE_CLOSE}\n"
        )
        _ = self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        _ = temporary.write_text(rendered, encoding="utf-8")
        _ = temporary.replace(self._path)

    def load(self, session_id: SessionId) -> MutableMemorySnapshot | None:
        if self._session_id is None:
            self._session_id = session_id
        elif self._session_id != session_id:
            raise MemoryStoreBoundaryError(_SESSION_ID_FIELD)
        if not self._path.exists():
            return None
        raw = self._path.read_text(encoding="utf-8")
        start = raw.find(_MARKDOWN_STATE_OPEN)
        end = raw.find(_MARKDOWN_STATE_CLOSE, start + len(_MARKDOWN_STATE_OPEN))
        if start < 0 or end < 0:
            field = "memory.md"
            raise MemoryStoreBoundaryError(field)
        encoded = raw[start + len(_MARKDOWN_STATE_OPEN) : end]
        try:
            document = _object(parse_json_value(encoded))
        except JsonBoundaryError as error:
            raise MemoryStoreBoundaryError(error.field_name) from error
        return _snapshot_from_document(document, session_id)


def _document_for_snapshot(
    snapshot: MutableMemorySnapshot, session_id: SessionId
) -> dict[str, object]:
    return {
        "session_id": str(session_id),
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


def _snapshot_from_document(
    document: dict[str, JsonValue], session_id: SessionId
) -> MutableMemorySnapshot | None:
    stored_session_id = SessionId(_text(document, _SESSION_ID_FIELD))
    if stored_session_id != session_id:
        return None
    preferences = _array(document, "preferences")
    entries = tuple(
        MemoryEntry(
            key=MemoryKey(_text(entry, "key")),
            value=_text(entry, "value"),
            provenance=MemoryProvenance(
                source=_source(_text(entry, "source"), index),
                trace_id=TraceId(_text(entry, "trace_id")),
                session_id=SessionId(_text(entry, "session_id")),
                turn_id=TurnId(_text(entry, "turn_id")),
                evidence_id=_text(entry, "evidence_id"),
            ),
        )
        for index, item in enumerate(preferences)
        for entry in (_object(item),)
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


def _source(value: str, index: int) -> MemorySource:
    try:
        return MemorySource(value)

    except ValueError as error:
        field = f"preferences[{index}].source"

        raise MemoryStoreBoundaryError(field) from error
