import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol, final, override

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.json_boundary import JsonBoundaryError, JsonValue, parse_json_value
from orchestrator.memory import (
    MemoryAudit,
    MemoryCategory,
    MemoryConfidence,
    MemoryConflictAudit,
    MemoryEntry,
    MemoryKey,
    MemoryPolicy,
    MemoryProposal,
    MemoryProvenance,
    MemorySource,
    MutableMemory,
    MutableMemorySnapshot,
    ProposalRevision,
)
from orchestrator.memory_policy import (
    MAX_MEMORY_AUDIT_RECORDS,
    MAX_MEMORY_CONFIDENCE,
    MAX_MEMORY_CONFLICT_RECORDS,
    MAX_MEMORY_DOCUMENT_BYTES,
    MAX_MEMORY_PROVENANCE_BYTES,
    contains_sensitive_memory,
    valid_memory_text,
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

        document = _document_for_snapshot(snapshot, self._session_id)
        rendered = (
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _check_document_size(rendered)

        _ = self._path.parent.mkdir(parents=True, exist_ok=True)

        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")

        _ = temporary.write_text(
            rendered,
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
            document = _object(parse_json_value(_read_document(self._path)))

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
        rendered = render_markdown_memory(snapshot, self._session_id)
        _check_document_size(rendered)
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
        raw = _read_document(self._path)
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


def render_markdown_memory(
    snapshot: MutableMemorySnapshot, session_id: SessionId
) -> str:
    """Render the exact session-owned memory document injected into the Brain."""
    document = _document_for_snapshot(snapshot, session_id)
    rendered_entries = "\n".join(
        "\n".join(
            (
                f"## {entry.key}",
                f"- 值: {entry.value}",
                f"- 类别: {entry.category}",
                f"- 来源轮次: {entry.provenance.turn_id}",
                f"- 置信度: {entry.confidence}",
                f"- 更新时间(毫秒): {entry.updated_at_ms}",
                f"- 证据: {entry.provenance.evidence_id}",
            )
        )
        for entry in snapshot.entries
    )
    state = json.dumps(document, ensure_ascii=False, sort_keys=True)
    return (
        "# 会话记忆\n\n"
        f"会话: {session_id}\n\n"
        f"版本: {snapshot.revision}\n\n"
        f"{rendered_entries}\n\n"
        f"{_MARKDOWN_STATE_OPEN}{state}{_MARKDOWN_STATE_CLOSE}\n"
    )


def _document_for_snapshot(
    snapshot: MutableMemorySnapshot, session_id: SessionId
) -> dict[str, object]:
    return {
        "format_version": 2,
        "session_id": str(session_id),
        "revision": int(snapshot.revision),
        "preferences": [
            {
                "key": entry.key,
                "value": entry.value,
                "category": entry.category,
                "confidence": entry.confidence,
                "updated_at_ms": entry.updated_at_ms,
                "source": entry.provenance.source,
                "trace_id": entry.provenance.trace_id,
                "session_id": entry.provenance.session_id,
                "turn_id": entry.provenance.turn_id,
                "evidence_id": entry.provenance.evidence_id,
            }
            for entry in snapshot.entries
        ],
        "conflict_audit": [asdict(item) for item in snapshot.conflict_audit],
        "audit": [asdict(item) for item in snapshot.audit],
    }


def _snapshot_from_document(
    document: dict[str, JsonValue], session_id: SessionId
) -> MutableMemorySnapshot | None:
    version = _optional_integer(document, "format_version", 1)
    if version not in {1, 2}:
        field = "format_version"
        raise MemoryStoreBoundaryError(field)
    stored_session_id = SessionId(_text(document, _SESSION_ID_FIELD))
    if stored_session_id != session_id:
        return None
    preferences = _array(document, "preferences")
    entries = tuple(
        MemoryEntry(
            key=MemoryKey(_text(entry, "key")),
            value=_text(entry, "value"),
            category=_category(entry.get("category"), index),
            confidence=MemoryConfidence(_optional_integer(entry, "confidence", 100)),
            updated_at_ms=_optional_integer(entry, "updated_at_ms", 0),
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
    # Never silently migrate unsafe legacy files into the Brain's memory.
    validator = MutableMemory(session_id=session_id, policy=MemoryPolicy())
    for entry in entries:
        proposal = MemoryProposal(
            entry.key,
            entry.value,
            entry.category,
            entry.confidence,
            ProposalRevision(validator.snapshot.revision),
            entry.provenance,
        )
        if validator.validate(proposal) is not None or any(
            item.key == entry.key for item in validator.snapshot.entries
        ):
            field = "preferences"
            raise MemoryStoreBoundaryError(field)
        _ = validator.reduce(proposal)
    revision = _integer(document, "revision")
    if revision < 0:
        field = "revision"
        raise MemoryStoreBoundaryError(field)
    return MutableMemorySnapshot(
        revision=MemoryRevision(revision),
        entries=entries,
        profile_revision=ProfileRevision(0),
        consent_revision=ConsentRevision(0),
        conflict_audit=_conflicts(document),
        audit=_audit(document, revision),
    )


def _check_document_size(text: str) -> None:
    if len(text.encode()) > MAX_MEMORY_DOCUMENT_BYTES:
        field = "document_size"
        raise MemoryStoreBoundaryError(field)


def _read_document(path: Path) -> str:
    with path.open("rb") as stream:
        raw = stream.read(MAX_MEMORY_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_MEMORY_DOCUMENT_BYTES:
        field = "document_size"
        raise MemoryStoreBoundaryError(field)
    try:
        return raw.decode("utf-8")
    except UnicodeError as error:
        field = "encoding"
        raise MemoryStoreBoundaryError(field) from error


def _audit_items(
    document: dict[str, JsonValue], field: str
) -> list[dict[str, JsonValue]]:
    items = document.get(field, [])
    if not isinstance(items, list) or len(items) > MAX_MEMORY_AUDIT_RECORDS:
        raise MemoryStoreBoundaryError(field)
    return [_object(item) for item in items]


def _conflicts(document: dict[str, JsonValue]) -> tuple[MemoryConflictAudit, ...]:
    results: list[MemoryConflictAudit] = []
    items = _audit_items(document, "conflict_audit")
    if len(items) > MAX_MEMORY_CONFLICT_RECORDS:
        field = "conflict_audit"
        raise MemoryStoreBoundaryError(field)
    for item in items:
        key = _text(item, "key")
        old = _text(item, "replaced_value")
        new = _text(item, "replacement_value")
        confidences = (
            _integer(item, "replaced_confidence"),
            _integer(item, "replacement_confidence"),
        )
        if any(
            not valid_memory_text(key, value) or contains_sensitive_memory(key, value)
            for value in (old, new)
        ) or any(not 0 <= value <= MAX_MEMORY_CONFIDENCE for value in confidences):
            field = "conflict_audit"
            raise MemoryStoreBoundaryError(field)
        results.append(
            MemoryConflictAudit(
                MemoryKey(key),
                old,
                new,
                MemoryConfidence(confidences[0]),
                MemoryConfidence(confidences[1]),
            )
        )
    return tuple(results)


def _audit(document: dict[str, JsonValue], revision: int) -> tuple[MemoryAudit, ...]:
    results: list[MemoryAudit] = []
    for item in _audit_items(document, "audit"):
        record = MemoryAudit(
            revision=_integer(item, "revision"),
            action=_text(item, "action"),
            key=_text(item, "key"),
            previous_digest=_text(item, "previous_digest"),
            value_digest=_text(item, "value_digest"),
            trace_id=_text(item, "trace_id"),
            turn_id=_text(item, "turn_id"),
            evidence_id=_text(item, "evidence_id"),
            updated_at_ms=_integer(item, "updated_at_ms"),
        )
        if (
            not 0 < record.revision <= revision
            or record.action not in {"add", "replace", "delete"}
            or any(
                len(value.encode()) > MAX_MEMORY_PROVENANCE_BYTES
                for value in (
                    record.key,
                    record.trace_id,
                    record.turn_id,
                    record.evidence_id,
                )
            )
        ):
            field = "audit"
            raise MemoryStoreBoundaryError(field)
        results.append(record)
    return tuple(results)


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


def _optional_integer(document: dict[str, JsonValue], field: str, default: int) -> int:
    value = document.get(field, default)
    if type(value) is not int:
        raise MemoryStoreBoundaryError(field)
    return value


def _category(value: JsonValue | None, index: int) -> MemoryCategory:
    field = f"preferences[{index}].category"
    if value is None:
        return MemoryCategory.ORDINARY_PREFERENCE
    if not isinstance(value, str):
        raise MemoryStoreBoundaryError(field)
    try:
        return MemoryCategory(value)
    except ValueError as error:
        raise MemoryStoreBoundaryError(field) from error


def _source(value: str, index: int) -> MemorySource:
    try:
        return MemorySource(value)

    except ValueError as error:
        field = f"preferences[{index}].source"

        raise MemoryStoreBoundaryError(field) from error
