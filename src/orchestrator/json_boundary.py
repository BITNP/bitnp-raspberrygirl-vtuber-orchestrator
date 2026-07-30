
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, final, override

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)


ROOT_FIELD: Final = "$"

END_REASON: Final = "expected end of JSON"

VALUE_REASON: Final = "expected JSON value"

STRING_REASON: Final = "unterminated string"

ESCAPE_REASON: Final = "unsupported string escape"

EOF_REASON: Final = "unexpected end of JSON"


@dataclass(slots=True)
class JsonBoundaryError(ValueError):

    field_name: str

    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.field_name}: {self.reason}"


@final
class _JsonParser:

    def __init__(self, text: str) -> None:
        self._text: str = text

        self._index: int = 0

    def parse(self) -> JsonValue:
        value = self._value()

        self._skip_ws()

        if self._index != len(self._text):
            raise _root_error(END_REASON)

        return value

    def _value(self) -> JsonValue:
        self._skip_ws()

        char = self._peek()

        if char == "{":
            value = self._object()

        elif char == "[":
            value = self._array()

        elif char == '"':
            value = self._string()

        elif char in "-0123456789":
            value = self._number()

        elif self._consume_literal("true"):
            value = True

        elif self._consume_literal("false"):
            value = False

        elif self._consume_literal("null"):
            value = None

        else:
            raise _root_error(VALUE_REASON)

        return value

    def _object(self) -> dict[str, JsonValue]:
        self._expect("{")

        result: dict[str, JsonValue] = {}

        self._skip_ws()

        if self._consume("}"):
            return result

        while True:
            self._skip_ws()

            key = self._string()

            self._skip_ws()

            self._expect(":")

            result[key] = self._value()

            self._skip_ws()

            if self._consume("}"):
                return result

            self._expect(",")

    def _array(self) -> list[JsonValue]:
        self._expect("[")

        result: list[JsonValue] = []

        self._skip_ws()

        if self._consume("]"):
            return result

        while True:
            result.append(self._value())

            self._skip_ws()

            if self._consume("]"):
                return result

            self._expect(",")

    def _string(self) -> str:
        self._expect('"')

        chars: list[str] = []

        while self._index < len(self._text):
            char = self._text[self._index]

            self._index += 1

            if char == '"':
                return "".join(chars)

            if char == "\\":
                chars.append(self._escape())

            else:
                chars.append(char)

        raise _root_error(STRING_REASON)

    def _escape(self) -> str:
        char = self._peek()

        self._index += 1

        escapes = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }

        if char in escapes:
            return escapes[char]

        if char == "u":
            hex_digits = self._text[self._index : self._index + 4]

            self._index += 4

            return chr(int(hex_digits, 16))

        raise _root_error(ESCAPE_REASON)

    def _number(self) -> int | float:
        start = self._index

        if self._peek() == "-":
            self._index += 1

        while self._index < len(self._text) and self._text[self._index].isdigit():
            self._index += 1

        if self._index < len(self._text) and self._text[self._index] == ".":
            self._index += 1

            while self._index < len(self._text) and self._text[self._index].isdigit():
                self._index += 1

            return float(self._text[start : self._index])

        return int(self._text[start : self._index])

    def _skip_ws(self) -> None:
        while self._index < len(self._text) and self._text[self._index].isspace():
            self._index += 1

    def _peek(self) -> str:
        if self._index >= len(self._text):
            raise _root_error(EOF_REASON)

        return self._text[self._index]

    def _consume(self, expected: str) -> bool:
        if self._index < len(self._text) and self._text[self._index] == expected:
            self._index += 1

            return True

        return False

    def _consume_literal(self, literal: str) -> bool:
        if self._text.startswith(literal, self._index):
            self._index += len(literal)

            return True

        return False

    def _expect(self, expected: str) -> None:
        if not self._consume(expected):
            reason = f"expected {expected}"

            raise _root_error(reason)


def parse_json_value(text: str) -> JsonValue:
    return _JsonParser(text).parse()


def _root_error(reason: str) -> JsonBoundaryError:
    return JsonBoundaryError(ROOT_FIELD, reason)
