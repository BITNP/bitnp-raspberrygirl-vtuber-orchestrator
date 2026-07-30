
import json
import os
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Protocol, final, override

from orchestrator.identity import VoiceProfileId
from orchestrator.ids import SessionId
from orchestrator.json_boundary import JsonBoundaryError, JsonValue, parse_json_value
from orchestrator.state_snapshots import ConsentRevision, ProfileRevision


@unique
class ProfileLifecycle(StrEnum):

    ACTIVE = "active"

    REVOKED = "revoked"

    EXPIRED = "expired"

    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class ProfileAuditEntry:

    action: str

    revision: ProfileRevision


@dataclass(frozen=True, slots=True)
class VoiceProfileRecord:

    profile_id: VoiceProfileId

    preferred_name: str

    purpose: str

    confirmed: bool

    expires_at_ms: int | None

    lifecycle: ProfileLifecycle

    revision: ProfileRevision

    audit: tuple[ProfileAuditEntry, ...]


@dataclass(frozen=True, slots=True)
class VoiceProfileSnapshot:

    session_id: SessionId

    profile_revision: ProfileRevision

    consent_revision: ConsentRevision

    records: tuple[VoiceProfileRecord, ...]


class VoiceProfileStore(Protocol):

    def save(self, snapshot: VoiceProfileSnapshot) -> None:
        ...

    def load(self, session_id: SessionId) -> VoiceProfileSnapshot | None:
        ...


@dataclass(frozen=True, slots=True)
class ProfileStoreBoundaryError(ValueError):

    field_name: str

    @override
    def __str__(self) -> str:
        return f"invalid voice profile record: {self.field_name}"


@dataclass(frozen=True, slots=True)
class ProfileStoreAmbiguousCommitError(OSError):
    ...


@final
class JsonVoiceProfileStore:

    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, snapshot: VoiceProfileSnapshot) -> None:
        document = {
            "session_id": str(snapshot.session_id),
            "profile_revision": int(snapshot.profile_revision),
            "consent_revision": int(snapshot.consent_revision),
            "profiles": [
                {
                    "profile_id": str(record.profile_id),
                    "preferred_name": record.preferred_name,
                    "purpose": record.purpose,
                    "confirmed": record.confirmed,
                    "expires_at_ms": record.expires_at_ms,
                    "lifecycle": record.lifecycle.value,
                    "revision": int(record.revision),
                    "audit": [
                        {"action": entry.action, "revision": int(entry.revision)}
                        for entry in record.audit
                    ],
                }
                for record in snapshot.records
            ],
        }

        _ = self._path.parent.mkdir(parents=True, exist_ok=True)

        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")

        with temporary.open("w", encoding="utf-8") as file:
            _ = file.write(
                json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )

            _ = file.flush()

            os.fsync(file.fileno())

        _ = temporary.replace(self._path)

        try:
            _fsync_directory(self._path.parent)

        except OSError as error:
            loaded = self.load(snapshot.session_id)

            if loaded == snapshot:
                return

            raise ProfileStoreAmbiguousCommitError from error

    def load(self, session_id: SessionId) -> VoiceProfileSnapshot | None:
        if not self._path.exists():
            return None

        try:
            value = parse_json_value(self._path.read_text(encoding="utf-8"))

        except JsonBoundaryError as error:
            raise ProfileStoreBoundaryError(error.field_name) from error

        document = _object(value, "$")

        stored_session_id = SessionId(_text(document, "session_id"))

        if stored_session_id != session_id:
            return None

        records = tuple(
            _record(item, index)
            for index, item in enumerate(_array(document, "profiles"))
        )

        return VoiceProfileSnapshot(
            session_id=stored_session_id,
            profile_revision=ProfileRevision(_integer(document, "profile_revision")),
            consent_revision=ConsentRevision(_integer(document, "consent_revision")),
            records=records,
        )


def _record(value: JsonValue, index: int) -> VoiceProfileRecord:
    document = _object(value, f"profiles[{index}]")

    lifecycle_text = _text(document, "lifecycle")

    try:
        lifecycle = ProfileLifecycle(lifecycle_text)

    except ValueError as error:
        field_name = f"profiles[{index}].lifecycle"

        raise ProfileStoreBoundaryError(field_name) from error

    audit = tuple(
        ProfileAuditEntry(
            action=_text(_object(item, f"profiles[{index}].audit"), "action"),
            revision=ProfileRevision(
                _integer(_object(item, f"profiles[{index}].audit"), "revision")
            ),
        )
        for item in _array(document, "audit")
    )

    return VoiceProfileRecord(
        profile_id=VoiceProfileId(_text(document, "profile_id")),
        preferred_name=_text(document, "preferred_name"),
        purpose=_text(document, "purpose"),
        confirmed=_boolean(document, "confirmed"),
        expires_at_ms=_optional_integer(document, "expires_at_ms"),
        lifecycle=lifecycle,
        revision=ProfileRevision(_integer(document, "revision")),
        audit=audit,
    )


def _object(value: JsonValue, field_name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ProfileStoreBoundaryError(field_name)

    return value


def _array(document: dict[str, JsonValue], field_name: str) -> list[JsonValue]:
    value = document.get(field_name)

    if not isinstance(value, list):
        raise ProfileStoreBoundaryError(field_name)

    return value


def _text(document: dict[str, JsonValue], field_name: str) -> str:
    value = document.get(field_name)

    if not isinstance(value, str) or value.strip() == "":
        raise ProfileStoreBoundaryError(field_name)

    return value


def _integer(document: dict[str, JsonValue], field_name: str) -> int:
    value = document.get(field_name)

    if type(value) is not int:
        raise ProfileStoreBoundaryError(field_name)

    return value


def _optional_integer(document: dict[str, JsonValue], field_name: str) -> int | None:
    value = document.get(field_name)

    if value is None:
        return None

    if type(value) is not int:
        raise ProfileStoreBoundaryError(field_name)

    return value


def _boolean(document: dict[str, JsonValue], field_name: str) -> bool:
    value = document.get(field_name)

    if not isinstance(value, bool):
        raise ProfileStoreBoundaryError(field_name)

    return value


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)

    try:
        os.fsync(descriptor)

    finally:
        os.close(descriptor)
