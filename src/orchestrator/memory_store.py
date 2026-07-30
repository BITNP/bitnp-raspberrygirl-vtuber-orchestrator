"""模块契约说明.

职责: 提供 orchestrator.memory_store
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 声明 MemoryStore
    协议接口,约束实现方必须提供的行为。
    契约: 方法: save、load。
    """

    def save(self, snapshot: MutableMemorySnapshot) -> None:
        """函数契约说明.

        功能: 执行 save 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 snapshot:
        MutableMemorySnapshot。 必填。
        契约: 同步调用。 返回 `None`。
        """
        ...

    def load(self, session_id: SessionId) -> MutableMemorySnapshot | None:
        """函数契约说明.

        功能: 执行 load 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。
        契约: 同步调用。 返回
        `MutableMemorySnapshot | None`。
        """
        ...


@dataclass(frozen=True, slots=True)
class MemoryStoreBoundaryError(ValueError):
    """类契约说明.

    职责: 保存 MemoryStoreBoundaryError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: field。 方法: __str__。
    """

    field: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"invalid memory record: {self.field}"


@final
class JsonMemoryStore:
    """类契约说明.

    职责: 定义 JsonMemoryStore
    的状态、行为和对外协作边界。
    契约: 方法: __init__、save、load。
    """

    def __init__(self, path: Path) -> None:
        """函数契约说明.

        功能: 初始化 JsonMemoryStore
        的字段并建立实例不变式。
        参数: self 表示当前实例。 path: Path。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._path: Path = path

    def save(self, snapshot: MutableMemorySnapshot) -> None:
        """函数契约说明.

        功能: 执行 save 的同步逻辑,并协调 mkdir,
        with_suffix, write_text,
        replace。
        参数: self 表示当前实例。 snapshot:
        MutableMemorySnapshot。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 load 的同步逻辑,并协调 _object,
        _array, tuple, any。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。
        契约: 同步调用。 返回
        `MutableMemorySnapshot | None`。
        """
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
    """函数契约说明.

    功能: 执行 _object 的同步逻辑,并协调 isinstance,
    MemoryStoreBoundaryError。
    参数: value: JsonValue。 必填。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    可能抛出 MemoryStoreBoundaryError。
    """
    if not isinstance(value, dict):
        field = "$"

        raise MemoryStoreBoundaryError(field)

    return value


def _array(document: dict[str, JsonValue], field: str) -> list[JsonValue]:
    """函数契约说明.

    功能: 执行 _array 的同步逻辑,并协调 get,
    isinstance,
    MemoryStoreBoundaryError。
    参数: document: dict[str, JsonValue]。
    必填。 field: str。 必填。
    契约: 同步调用。 返回 `list[JsonValue]`。 可能抛出
    MemoryStoreBoundaryError。
    """
    value = document.get(field)

    if not isinstance(value, list):
        raise MemoryStoreBoundaryError(field)

    return value


def _text(document: dict[str, JsonValue], field: str) -> str:
    """函数契约说明.

    功能: 执行 _text 的同步逻辑,并协调 get,
    isinstance,
    MemoryStoreBoundaryError。
    参数: document: dict[str, JsonValue]。
    必填。 field: str。 必填。
    契约: 同步调用。 返回 `str`。 可能抛出
    MemoryStoreBoundaryError。
    """
    value = document.get(field)

    if not isinstance(value, str):
        raise MemoryStoreBoundaryError(field)

    return value


def _integer(document: dict[str, JsonValue], field: str) -> int:
    """函数契约说明.

    功能: 执行 _integer 的同步逻辑,并协调 get, type,
    MemoryStoreBoundaryError。
    参数: document: dict[str, JsonValue]。
    必填。 field: str。 必填。
    契约: 同步调用。 返回 `int`。 可能抛出
    MemoryStoreBoundaryError。
    """
    value = document.get(field)

    if type(value) is not int:
        raise MemoryStoreBoundaryError(field)

    return value
