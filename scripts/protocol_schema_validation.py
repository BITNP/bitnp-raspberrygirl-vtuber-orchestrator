"""模块契约说明.

职责: 提供命令行脚本的参数处理、验证或运维流程。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, assert_never

from jsonschema import Draft202012Validator, FormatChecker

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

JsonObject: TypeAlias = dict[str, JsonValue]

SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({"1.0.0", "1.1.0"})

ALLOWED_SOURCES: Final = frozenset(
    {"orchestrator", "mic", "sound", "comments", "frontend"}
)

TURN_CORRELATED_EVENTS: Final = frozenset(
    {
        "llm.request",
        "llm.response.delta",
        "llm.response.final",
        "llm.response.error",
        "media.stream.command",
        "media.stream.state",
        "vtuber.caption.command",
        "vtuber.expression.command",
        "vtuber.action.command",
        "presentation.load.command",
        "presentation.play.command",
        "presentation.navigate.command",
        "presentation.result",
    }
)

SEGMENT_CORRELATED_EVENTS: Final = frozenset(
    {"media.stream.flush", "media.stream.flush.ack"}
)

VERSION: Final = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)


@dataclass(frozen=True, slots=True)
class ProtocolValidationError:
    """类契约说明.

    职责: 保存 ProtocolValidationError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: code、path、message。 方法:
    as_json。
    """

    code: str

    path: str

    message: str

    def as_json(self) -> JsonObject:
        """函数契约说明.

        功能: 执行 as_json 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `JsonObject`。
        """
        return {"code": self.code, "message": self.message, "path": self.path}


def read_json(path: Path) -> JsonValue:
    """函数契约说明.

    功能: 执行 read_json 的同步逻辑,并协调 loads,
    read_text。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `JsonValue`。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def validate_file(path: Path, schema_root: Path) -> list[ProtocolValidationError]:
    """函数契约说明.

    功能: 校验相关输入、协议或运行时约束。
    参数: path: Path。 必填。 schema_root:
    Path。 必填。
    契约: 同步调用。 返回
    `list[ProtocolValidationError]`。
    """
    parsed = read_json(path)

    schema = _canonical_schema(schema_root)

    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    events = parsed if isinstance(parsed, list) else [parsed]

    errors = [
        error
        for index, event in enumerate(events)
        for error in _schema_errors(validator, event, index)
    ]

    if errors:
        return errors

    objects = [event for event in events if isinstance(event, dict)]

    return _semantic_errors(objects)


def _canonical_schema(schema_root: Path) -> JsonObject:
    """函数契约说明.

    功能: 执行 _canonical_schema 的同步逻辑,并协调
    read_json, get, sorted,
    RuntimeError。
    参数: schema_root: Path。 必填。
    契约: 同步调用。 返回 `JsonObject`。 可能抛出
    RuntimeError。
    """
    envelope = read_json(schema_root / "envelope.schema.json")

    event_data = read_json(schema_root / "event-data.schema.json")

    if not isinstance(envelope, dict) or not isinstance(event_data, dict):
        raise RuntimeError("canonical protocol schemas must be JSON objects")

    event_types = event_data.get("event_types")

    definitions = event_data.get("definitions")

    if not isinstance(event_types, dict) or not isinstance(definitions, dict):
        raise RuntimeError("event-data schema must define event_types and definitions")

    properties = envelope.get("properties")

    if not isinstance(properties, dict):
        raise RuntimeError("envelope schema must define properties")

    event_type_names: list[JsonValue] = sorted(event_types)

    source_names: list[JsonValue] = sorted(ALLOWED_SOURCES)

    properties["event_type"] = {"enum": event_type_names}

    properties["source"] = {"enum": source_names}

    branches: list[JsonValue] = [
        _event_branch(name, definition) for name, definition in event_types.items()
    ]

    schema: JsonObject = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": envelope["required"],
        "properties": properties,
        "definitions": definitions,
        "allOf": branches,
    }

    return schema


def _event_branch(event_type: str, definition: JsonValue) -> JsonObject:
    """函数契约说明.

    功能: 执行 _event_branch 的同步逻辑,并协调 dict,
    isinstance, RuntimeError, get。
    参数: event_type: str。 必填。 definition:
    JsonValue。 必填。
    契约: 同步调用。 返回 `JsonObject`。 可能抛出
    RuntimeError。
    """
    if not isinstance(definition, dict):
        raise RuntimeError("event definition must be a JSON object")

    data_schema = dict(definition)

    data_schema["type"] = "object"

    data_schema["additionalProperties"] = False

    if "properties" not in data_schema:
        required = data_schema.get("required")

        if not isinstance(required, list):
            raise RuntimeError("event definition must declare string required fields")

        properties: JsonObject = {}

        for field in required:
            match field:
                case str():
                    properties[field] = {}

                case None | bool() | int() | float() | list() | dict():
                    raise RuntimeError(
                        "event definition must declare string required fields"
                    )

                case unreachable:
                    assert_never(unreachable)

        data_schema["properties"] = properties

    branch: JsonObject = {
        "if": {"properties": {"event_type": {"const": event_type}}},
        "then": {"properties": {"data": data_schema}},
    }

    return branch


def _schema_errors(
    validator: Draft202012Validator, event: JsonValue, index: int
) -> list[ProtocolValidationError]:
    """函数契约说明.

    功能: 执行 _schema_errors 的同步逻辑,并协调
    ProtocolValidationError, sorted,
    _json_path, iter_errors。
    参数: validator: Draft202012Validator。
    必填。 event: JsonValue。 必填。 index:
    int。 必填。
    契约: 同步调用。 返回
    `list[ProtocolValidationError]`。
    可能抛出 ProtocolValidationError。
    """
    return [
        ProtocolValidationError(
            code="schema_validation",
            path=_json_path(index, error.absolute_path),
            message=error.message,
        )
        for error in sorted(
            validator.iter_errors(event), key=lambda error: list(error.path)
        )
    ]


def _semantic_errors(events: list[JsonObject]) -> list[ProtocolValidationError]:
    """函数契约说明.

    功能: 执行 _semantic_errors 的同步逻辑,并协调
    set, enumerate, extend,
    _version_errors。
    参数: events: list[JsonObject]。 必填。
    契约: 同步调用。 返回
    `list[ProtocolValidationError]`。
    """
    errors: list[ProtocolValidationError] = []

    event_ids: set[str] = set()

    sequences: dict[str, int] = {}

    for index, event in enumerate(events):
        errors.extend(_version_errors(event, index))

        errors.extend(_event_id_errors(event, event_ids, index))

        errors.extend(_sequence_errors(event, sequences, index))

        errors.extend(_correlation_errors(event, index))

    return errors


def _version_errors(event: JsonObject, index: int) -> list[ProtocolValidationError]:
    """函数契约说明.

    功能: 执行 _version_errors 的同步逻辑,并协调
    isinstance, fullmatch,
    ProtocolValidationError, _json_path。
    参数: event: JsonObject。 必填。 index:
    int。 必填。
    契约: 同步调用。 返回
    `list[ProtocolValidationError]`。
    可能抛出 ProtocolValidationError。
    """
    version = event["schema_version"]

    if not isinstance(version, str) or VERSION.fullmatch(version) is None:
        return [
            ProtocolValidationError(
                "invalid_schema_version",
                _json_path(index, ("schema_version",)),
                "schema_version must be semantic version text",
            )
        ]

    if version not in SUPPORTED_SCHEMA_VERSIONS:
        return [
            ProtocolValidationError(
                "unsupported_schema_version",
                _json_path(index, ("schema_version",)),
                f"consumer does not declare compatibility with {version}",
            )
        ]

    return []


def _event_id_errors(
    event: JsonObject, event_ids: set[str], index: int
) -> list[ProtocolValidationError]:
    """函数契约说明.

    功能: 执行 _event_id_errors 的同步逻辑,并协调
    add, isinstance,
    ProtocolValidationError, _json_path。
    参数: event: JsonObject。 必填。
    event_ids: set[str]。 必填。 index: int。
    必填。
    契约: 同步调用。 返回
    `list[ProtocolValidationError]`。
    可能抛出 ProtocolValidationError。
    """
    event_id = event["event_id"]

    if not isinstance(event_id, str):
        return []

    if event_id in event_ids:
        return [
            ProtocolValidationError(
                "duplicate_event_id",
                _json_path(index, ("event_id",)),
                "event_id must be unique within a validated collection",
            )
        ]

    event_ids.add(event_id)

    return []


def _sequence_errors(
    event: JsonObject, sequences: dict[str, int], index: int
) -> list[ProtocolValidationError]:
    """函数契约说明.

    功能: 执行 _sequence_errors 的同步逻辑,并协调
    get, isinstance,
    ProtocolValidationError, _json_path。
    参数: event: JsonObject。 必填。
    sequences: dict[str, int]。 必填。
    index: int。 必填。
    契约: 同步调用。 返回
    `list[ProtocolValidationError]`。
    可能抛出 ProtocolValidationError。
    """
    session_id = event["session_id"]

    sequence = event["seq"]

    if not isinstance(session_id, str) or not isinstance(sequence, int):
        return []

    previous = sequences.get(session_id)

    sequences[session_id] = sequence

    if previous is not None and sequence <= previous:
        return [
            ProtocolValidationError(
                "sequence_regression",
                _json_path(index, ("seq",)),
                "seq must increase monotonically within session_id",
            )
        ]

    return []


def _correlation_errors(event: JsonObject, index: int) -> list[ProtocolValidationError]:
    """函数契约说明.

    功能: 执行 _correlation_errors 的同步逻辑,并协调
    fromkeys, ProtocolValidationError,
    _json_path, isinstance。
    参数: event: JsonObject。 必填。 index:
    int。 必填。
    契约: 同步调用。 返回
    `list[ProtocolValidationError]`。
    可能抛出 ProtocolValidationError。
    """
    event_type = event["event_type"]

    required_fields = dict.fromkeys(
        ("turn_id",)
        if event_type in TURN_CORRELATED_EVENTS
        else ()
        + (("turn_id", "segment_id") if event_type in SEGMENT_CORRELATED_EVENTS else ())
    )

    return [
        ProtocolValidationError(
            "missing_correlation",
            _json_path(index, (field,)),
            f"{field} is required for {event_type}",
        )
        for field in required_fields
        if not isinstance(value := event.get(field), str) or value.strip() == ""
    ]


def _json_path(index: int, path: Sequence[object]) -> str:
    """函数契约说明.

    功能: 执行 _json_path 的同步逻辑,并协调 join,
    str。
    参数: index: int。 必填。 path:
    Sequence[object]。 必填。
    契约: 同步调用。 返回 `str`。
    """
    parts = ".".join(str(part) for part in path)

    return f"$[{index}]" if parts == "" else f"$[{index}].{parts}"
