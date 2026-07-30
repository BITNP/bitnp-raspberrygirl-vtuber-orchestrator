"""模块契约说明.

职责: 提供 orchestrator.json_boundary
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 JsonBoundaryError
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


@final
class _JsonParser:
    """类契约说明.

    职责: 定义 _JsonParser 的状态、行为和对外协作边界。
    契约: 方法: __init__、parse、_value、_objec
    t、_array、_string。
    """

    def __init__(self, text: str) -> None:
        """函数契约说明.

        功能: 初始化 _JsonParser 的字段并建立实例不变式。
        参数: self 表示当前实例。 text: str。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._text: str = text

        self._index: int = 0

    def parse(self) -> JsonValue:
        """函数契约说明.

        功能: 从边界输入解析类型化值。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `JsonValue`。
        """
        value = self._value()

        self._skip_ws()

        if self._index != len(self._text):
            raise _root_error(END_REASON)

        return value

    def _value(self) -> JsonValue:
        """函数契约说明.

        功能: 执行 _value 的同步逻辑,并协调
        _skip_ws, _peek, _object,
        _array。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `JsonValue`。
        """
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
        """函数契约说明.

        功能: 执行 _object 的同步逻辑,并协调
        _expect, _skip_ws, _consume,
        _string。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `dict[str,
        JsonValue]`。
        """
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
        """函数契约说明.

        功能: 执行 _array 的同步逻辑,并协调 _expect,
        _skip_ws, _consume, append。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `list[JsonValue]`。
        """
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
        """函数契约说明.

        功能: 执行 _string 的同步逻辑,并协调
        _expect, _root_error, len, join。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
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
        """函数契约说明.

        功能: 执行 _escape 的同步逻辑,并协调 _peek,
        _root_error, chr, int。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
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
        """函数契约说明.

        功能: 执行 _number 的同步逻辑,并协调 int,
        _peek, isdigit, float。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `int | float`。
        """
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
        """函数契约说明.

        功能: 执行 _skip_ws 的同步逻辑,并协调
        isspace, len。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        while self._index < len(self._text) and self._text[self._index].isspace():
            self._index += 1

    def _peek(self) -> str:
        """函数契约说明.

        功能: 执行 _peek 的同步逻辑,并协调 len,
        _root_error。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        if self._index >= len(self._text):
            raise _root_error(EOF_REASON)

        return self._text[self._index]

    def _consume(self, expected: str) -> bool:
        """函数契约说明.

        功能: 执行 _consume 的同步逻辑,并协调 len。
        参数: self 表示当前实例。 expected: str。
        必填。
        契约: 同步调用。 返回 `bool`。
        """
        if self._index < len(self._text) and self._text[self._index] == expected:
            self._index += 1

            return True

        return False

    def _consume_literal(self, literal: str) -> bool:
        """函数契约说明.

        功能: 执行 _consume_literal
        的同步逻辑,并协调 startswith, len。
        参数: self 表示当前实例。 literal: str。
        必填。
        契约: 同步调用。 返回 `bool`。
        """
        if self._text.startswith(literal, self._index):
            self._index += len(literal)

            return True

        return False

    def _expect(self, expected: str) -> None:
        """函数契约说明.

        功能: 执行 _expect 的同步逻辑,并协调
        _consume, _root_error。
        参数: self 表示当前实例。 expected: str。
        必填。
        契约: 同步调用。 返回 `None`。
        """
        if not self._consume(expected):
            reason = f"expected {expected}"

            raise _root_error(reason)


def parse_json_value(text: str) -> JsonValue:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: text: str。 必填。
    契约: 同步调用。 返回 `JsonValue`。
    """
    return _JsonParser(text).parse()


def _root_error(reason: str) -> JsonBoundaryError:
    """函数契约说明.

    功能: 执行 _root_error 的同步逻辑,并协调
    JsonBoundaryError。
    参数: reason: str。 必填。
    契约: 同步调用。 返回 `JsonBoundaryError`。
    可能抛出 JsonBoundaryError。
    """
    return JsonBoundaryError(ROOT_FIELD, reason)
