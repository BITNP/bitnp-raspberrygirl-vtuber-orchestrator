"""Lecturer-mode script parsing owned by Orchestrator."""

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
    """Raised when a user-authored lecture script is malformed."""

    field_name: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.field_name}: {self.reason}"


@dataclass(frozen=True, slots=True)
class LectureSlide:
    """Slide command associated with one narration step."""

    id: str
    title: str
    page: int


@dataclass(frozen=True, slots=True)
class LectureStep:
    """One lecturer-mode narration and frontend command step."""

    id: str
    narration: str
    slide: LectureSlide
    expression: str
    action: str
    scene: str


@dataclass(frozen=True, slots=True)
class LectureScript:
    """Parsed lecture script consumed by the Orchestrator demo."""

    title: str
    voice: str
    steps: tuple[LectureStep, ...]


def parse_lecture_script(path: Path) -> LectureScript:
    """Parse a JSON lecture script file into typed Orchestrator data."""
    raw = _json_loads(path.read_text(encoding="utf-8"))
    data = _require_object(raw, "$")
    steps = _require_steps(data)
    return LectureScript(
        title=_require_str(data, "title"),
        voice=_optional_str(data, "voice", default="raspberry-default"),
        steps=steps,
    )


def _require_steps(data: JsonObject) -> tuple[LectureStep, ...]:
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
    return LectureSlide(
        id=_require_str(data, f"{parent_field}.slide.id"),
        title=_require_str(data, f"{parent_field}.slide.title"),
        page=_require_positive_int(data, f"{parent_field}.slide.page"),
    )


def _require_object(value: JsonValue, field_name: str) -> JsonObject:
    if isinstance(value, dict):
        return value
    raise LectureScriptError(field_name, "expected object")


def _require_str(data: JsonObject, field_name: str) -> str:
    key = field_name.rsplit(".", 1)[-1]
    value = data.get(key)
    if isinstance(value, str) and value != "":
        return value
    raise LectureScriptError(field_name, "expected non-empty string")


def _optional_str(data: JsonObject, field_name: str, *, default: str) -> str:
    value = data.get(field_name)
    if value is None:
        return default
    if isinstance(value, str) and value != "":
        return value
    raise LectureScriptError(field_name, "expected non-empty string")


def _require_positive_int(data: JsonObject, field_name: str) -> int:
    key = field_name.rsplit(".", 1)[-1]
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise LectureScriptError(field_name, "expected positive integer")


def _json_loads(text: str) -> JsonValue:
    try:
        return parse_json_value(text)
    except JsonBoundaryError as error:
        raise LectureScriptError(error.field_name, error.reason) from error
