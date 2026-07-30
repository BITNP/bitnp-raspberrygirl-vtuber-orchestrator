"""模块契约说明.

职责: 提供 orchestrator.profile_store
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 定义 ProfileLifecycle
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    ACTIVE = "active"

    REVOKED = "revoked"

    EXPIRED = "expired"

    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class ProfileAuditEntry:
    """类契约说明.

    职责: 保存 ProfileAuditEntry
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: action、revision。
    """

    action: str

    revision: ProfileRevision


@dataclass(frozen=True, slots=True)
class VoiceProfileRecord:
    """类契约说明.

    职责: 保存 VoiceProfileRecord
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: profile_id、preferred_name、pu
    rpose、confirmed、expires_at_ms、lifecy
    cle。
    """

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
    """类契约说明.

    职责: 保存 VoiceProfileSnapshot
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: session_id、profile_revision、
    consent_revision、records。
    """

    session_id: SessionId

    profile_revision: ProfileRevision

    consent_revision: ConsentRevision

    records: tuple[VoiceProfileRecord, ...]


class VoiceProfileStore(Protocol):
    """类契约说明.

    职责: 声明 VoiceProfileStore
    协议接口,约束实现方必须提供的行为。
    契约: 方法: save、load。
    """

    def save(self, snapshot: VoiceProfileSnapshot) -> None:
        """函数契约说明.

        功能: 执行 save 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 snapshot:
        VoiceProfileSnapshot。 必填。
        契约: 同步调用。 返回 `None`。
        """
        ...

    def load(self, session_id: SessionId) -> VoiceProfileSnapshot | None:
        """函数契约说明.

        功能: 执行 load 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。
        契约: 同步调用。 返回
        `VoiceProfileSnapshot | None`。
        """
        ...


@dataclass(frozen=True, slots=True)
class ProfileStoreBoundaryError(ValueError):
    """类契约说明.

    职责: 保存 ProfileStoreBoundaryError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: field_name。 方法: __str__。
    """

    field_name: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"invalid voice profile record: {self.field_name}"


@dataclass(frozen=True, slots=True)
class ProfileStoreAmbiguousCommitError(OSError):
    """类契约说明.

    职责: 保存
    ProfileStoreAmbiguousCommitError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """


@final
class JsonVoiceProfileStore:
    """类契约说明.

    职责: 定义 JsonVoiceProfileStore
    的状态、行为和对外协作边界。
    契约: 方法: __init__、save、load。
    """

    def __init__(self, path: Path) -> None:
        """函数契约说明.

        功能: 初始化 JsonVoiceProfileStore
        的字段并建立实例不变式。
        参数: self 表示当前实例。 path: Path。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._path = path

    def save(self, snapshot: VoiceProfileSnapshot) -> None:
        """函数契约说明.

        功能: 执行 save 的同步逻辑,并协调 mkdir,
        with_suffix, replace, str。
        参数: self 表示当前实例。 snapshot:
        VoiceProfileSnapshot。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 load 的同步逻辑,并协调 _object,
        SessionId, tuple,
        VoiceProfileSnapshot。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。
        契约: 同步调用。 返回
        `VoiceProfileSnapshot | None`。
        可能抛出 ProfileStoreBoundaryError。
        """
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
    """函数契约说明.

    功能: 执行 _record 的同步逻辑,并协调 _object,
    _text, tuple, VoiceProfileRecord。
    参数: value: JsonValue。 必填。 index:
    int。 必填。
    契约: 同步调用。 返回 `VoiceProfileRecord`。
    可能抛出 ProfileStoreBoundaryError。
    """
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
    """函数契约说明.

    功能: 执行 _object 的同步逻辑,并协调 isinstance,
    ProfileStoreBoundaryError。
    参数: value: JsonValue。 必填。
    field_name: str。 必填。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    可能抛出 ProfileStoreBoundaryError。
    """
    if not isinstance(value, dict):
        raise ProfileStoreBoundaryError(field_name)

    return value


def _array(document: dict[str, JsonValue], field_name: str) -> list[JsonValue]:
    """函数契约说明.

    功能: 执行 _array 的同步逻辑,并协调 get,
    isinstance,
    ProfileStoreBoundaryError。
    参数: document: dict[str, JsonValue]。
    必填。 field_name: str。 必填。
    契约: 同步调用。 返回 `list[JsonValue]`。 可能抛出
    ProfileStoreBoundaryError。
    """
    value = document.get(field_name)

    if not isinstance(value, list):
        raise ProfileStoreBoundaryError(field_name)

    return value


def _text(document: dict[str, JsonValue], field_name: str) -> str:
    """函数契约说明.

    功能: 执行 _text 的同步逻辑,并协调 get,
    ProfileStoreBoundaryError,
    isinstance, strip。
    参数: document: dict[str, JsonValue]。
    必填。 field_name: str。 必填。
    契约: 同步调用。 返回 `str`。 可能抛出
    ProfileStoreBoundaryError。
    """
    value = document.get(field_name)

    if not isinstance(value, str) or value.strip() == "":
        raise ProfileStoreBoundaryError(field_name)

    return value


def _integer(document: dict[str, JsonValue], field_name: str) -> int:
    """函数契约说明.

    功能: 执行 _integer 的同步逻辑,并协调 get, type,
    ProfileStoreBoundaryError。
    参数: document: dict[str, JsonValue]。
    必填。 field_name: str。 必填。
    契约: 同步调用。 返回 `int`。 可能抛出
    ProfileStoreBoundaryError。
    """
    value = document.get(field_name)

    if type(value) is not int:
        raise ProfileStoreBoundaryError(field_name)

    return value


def _optional_integer(document: dict[str, JsonValue], field_name: str) -> int | None:
    """函数契约说明.

    功能: 执行 _optional_integer 的同步逻辑,并协调
    get, type,
    ProfileStoreBoundaryError。
    参数: document: dict[str, JsonValue]。
    必填。 field_name: str。 必填。
    契约: 同步调用。 返回 `int | None`。 可能抛出
    ProfileStoreBoundaryError。
    """
    value = document.get(field_name)

    if value is None:
        return None

    if type(value) is not int:
        raise ProfileStoreBoundaryError(field_name)

    return value


def _boolean(document: dict[str, JsonValue], field_name: str) -> bool:
    """函数契约说明.

    功能: 执行 _boolean 的同步逻辑,并协调 get,
    isinstance,
    ProfileStoreBoundaryError。
    参数: document: dict[str, JsonValue]。
    必填。 field_name: str。 必填。
    契约: 同步调用。 返回 `bool`。 可能抛出
    ProfileStoreBoundaryError。
    """
    value = document.get(field_name)

    if not isinstance(value, bool):
        raise ProfileStoreBoundaryError(field_name)

    return value


def _fsync_directory(directory: Path) -> None:
    """函数契约说明.

    功能: 执行 _fsync_directory 的同步逻辑,并协调
    open, fsync, close。
    参数: directory: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """
    descriptor = os.open(directory, os.O_RDONLY)

    try:
        os.fsync(descriptor)

    finally:
        os.close(descriptor)
