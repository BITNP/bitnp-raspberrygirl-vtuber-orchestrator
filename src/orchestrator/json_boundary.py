from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, NoReturn, cast, override

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)

ROOT_FIELD: Final = "$"
MAX_JSON_DEPTH: Final = 32


@dataclass(slots=True)
class JsonBoundaryError(ValueError):
    field_name: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.field_name}: {self.reason}"


def parse_json_value(text: str) -> JsonValue:
    """Parse standards-compliant JSON while closing ambiguous input forms."""
    try:
        value = cast(
            "JsonValue",
            json.loads(
                text,
                object_pairs_hook=_closed_object,
                parse_constant=_reject_constant,
            ),
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        if isinstance(error, JsonBoundaryError):
            raise
        raise JsonBoundaryError(ROOT_FIELD, "invalid JSON") from error
    _validate_depth(value, depth=0)
    return value


def _closed_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise JsonBoundaryError(ROOT_FIELD, f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise JsonBoundaryError(ROOT_FIELD, f"non-finite number: {value}")


def _validate_depth(value: JsonValue, *, depth: int) -> None:
    if depth > MAX_JSON_DEPTH:
        raise JsonBoundaryError(ROOT_FIELD, "maximum nesting depth exceeded")
    if isinstance(value, dict):
        for child in value.values():
            _validate_depth(child, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            _validate_depth(child, depth=depth + 1)
