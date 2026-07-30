"""模块契约说明.

职责: 提供 orchestrator.lecturer_script
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from orchestrator.json_boundary import JsonBoundaryError, JsonValue, parse_json_value

if TYPE_CHECKING:
    from pathlib import Path


type JsonObject = Mapping[str, JsonValue]


@dataclass(slots=True)
class LectureScriptError(ValueError):
    """类契约说明.

    职责: 保存 LectureScriptError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: field_name、reason。 方法:
    __str__。
    """

    field_name: str

    reason: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"{self.field_name}: {self.reason}"


@dataclass(frozen=True, slots=True)
class LectureSlide:
    """类契约说明.

    职责: 保存 LectureSlide
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: id、title、page。
    """

    id: str

    title: str

    page: int


@dataclass(frozen=True, slots=True)
class LectureStep:
    """类契约说明.

    职责: 保存 LectureStep
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: id、narration、slide、expressio
    n、action、scene。
    """

    id: str

    narration: str

    slide: LectureSlide

    expression: str

    action: str

    scene: str


@dataclass(frozen=True, slots=True)
class LectureScript:
    """类契约说明.

    职责: 保存 LectureScript
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: title、voice、steps。
    """

    title: str

    voice: str

    steps: tuple[LectureStep, ...]


def parse_lecture_script(path: Path) -> LectureScript:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `LectureScript`。
    """
    raw = _json_loads(path.read_text(encoding="utf-8"))

    data = _require_object(raw, "$")

    steps = _require_steps(data)

    return LectureScript(
        title=_require_str(data, "title"),
        voice=_optional_str(data, "voice", default="raspberry-default"),
        steps=steps,
    )


def _require_steps(data: JsonObject) -> tuple[LectureStep, ...]:
    """函数契约说明.

    功能: 执行 _require_steps 的同步逻辑,并协调 get,
    tuple, LectureScriptError,
    isinstance。
    参数: data: JsonObject。 必填。
    契约: 同步调用。 返回 `tuple[LectureStep,
    ...]`。 可能抛出 LectureScriptError。
    """
    raw_steps = data.get("steps")

    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        field_name = "steps"

        reason = "expected non-empty list"

        raise LectureScriptError(field_name, reason)

    return tuple(
        _parse_step(_require_object(raw_step, f"steps[{index}]"), index)
        for index, raw_step in enumerate(raw_steps)
    )


def _parse_step(data: JsonObject, index: int) -> LectureStep:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: data: JsonObject。 必填。 index:
    int。 必填。
    契约: 同步调用。 返回 `LectureStep`。
    """
    field = f"steps[{index}]"

    return LectureStep(
        id=_require_str(data, f"{field}.id"),
        narration=_require_str(data, f"{field}.narration"),
        slide=_parse_slide(_require_object(data.get("slide"), f"{field}.slide"), field),
        expression=_require_str(data, f"{field}.expression"),
        action=_require_str(data, f"{field}.action"),
        scene=_require_str(data, f"{field}.scene"),
    )


def _parse_slide(data: JsonObject, parent_field: str) -> LectureSlide:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: data: JsonObject。 必填。
    parent_field: str。 必填。
    契约: 同步调用。 返回 `LectureSlide`。
    """
    return LectureSlide(
        id=_require_str(data, f"{parent_field}.slide.id"),
        title=_require_str(data, f"{parent_field}.slide.title"),
        page=_require_positive_int(data, f"{parent_field}.slide.page"),
    )


def _require_object(value: JsonValue, field_name: str) -> JsonObject:
    """函数契约说明.

    功能: 执行 _require_object 的同步逻辑,并协调
    isinstance, LectureScriptError。
    参数: value: JsonValue。 必填。
    field_name: str。 必填。
    契约: 同步调用。 返回 `JsonObject`。 可能抛出
    LectureScriptError。
    """
    if isinstance(value, dict):
        return value

    raise LectureScriptError(field_name, "expected object")


def _require_str(data: JsonObject, field_name: str) -> str:
    """函数契约说明.

    功能: 执行 _require_str 的同步逻辑,并协调 get,
    LectureScriptError, rsplit,
    isinstance。
    参数: data: JsonObject。 必填。
    field_name: str。 必填。
    契约: 同步调用。 返回 `str`。 可能抛出
    LectureScriptError。
    """
    key = field_name.rsplit(".", 1)[-1]

    value = data.get(key)

    if isinstance(value, str) and value != "":
        return value

    raise LectureScriptError(field_name, "expected non-empty string")


def _optional_str(data: JsonObject, field_name: str, *, default: str) -> str:
    """函数契约说明.

    功能: 执行 _optional_str 的同步逻辑,并协调 get,
    LectureScriptError, isinstance。
    参数: data: JsonObject。 必填。
    field_name: str。 必填。 default: str。
    必填。
    契约: 同步调用。 返回 `str`。 可能抛出
    LectureScriptError。
    """
    value = data.get(field_name)

    if value is None:
        return default

    if isinstance(value, str) and value != "":
        return value

    raise LectureScriptError(field_name, "expected non-empty string")


def _require_positive_int(data: JsonObject, field_name: str) -> int:
    """函数契约说明.

    功能: 执行 _require_positive_int
    的同步逻辑,并协调 get, LectureScriptError,
    rsplit, isinstance。
    参数: data: JsonObject。 必填。
    field_name: str。 必填。
    契约: 同步调用。 返回 `int`。 可能抛出
    LectureScriptError。
    """
    key = field_name.rsplit(".", 1)[-1]

    value = data.get(key)

    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value

    raise LectureScriptError(field_name, "expected positive integer")


def _json_loads(text: str) -> JsonValue:
    """函数契约说明.

    功能: 执行 _json_loads 的同步逻辑,并协调
    parse_json_value,
    LectureScriptError。
    参数: text: str。 必填。
    契约: 同步调用。 返回 `JsonValue`。 可能抛出
    LectureScriptError。
    """
    try:
        return parse_json_value(text)

    except JsonBoundaryError as error:
        raise LectureScriptError(error.field_name, error.reason) from error
