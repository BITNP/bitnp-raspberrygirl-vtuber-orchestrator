"""模块契约说明.

职责: 提供 orchestrator.provider_streaming
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection, ResponseNotReady
from threading import Lock
from typing import TYPE_CHECKING, Literal, override
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


ProviderCapability = Literal["streaming", "final_only"]

_SUCCESS_MIN = 200

_SUCCESS_MAX = 300


@dataclass(frozen=True, slots=True)
class ProviderDeadlines:
    """类契约说明.

    职责: 保存 ProviderDeadlines
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: connect_seconds、read_seconds
    、total_seconds。
    """

    connect_seconds: float = 5.0

    read_seconds: float = 30.0

    total_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """类契约说明.

    职责: 保存 ProviderRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: url、body、headers、stage。
    """

    url: str

    body: bytes

    headers: dict[str, str]

    stage: str


@dataclass(slots=True)
class ProviderResponseError(OSError):
    """类契约说明.

    职责: 保存 ProviderResponseError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stage、reason。 方法: __str__。
    """

    stage: str

    reason: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"{self.stage} provider response error: {self.reason}"


class ProviderCancellationHandle:
    """类契约说明.

    职责: 定义 ProviderCancellationHandle
    的状态、行为和对外协作边界。
    契约: 方法: __init__、cancelled、reason、ca
    ncel、bind。
    """

    def __init__(self) -> None:
        """函数契约说明.

        功能: 初始化
        ProviderCancellationHandle
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self._callbacks: dict[int, Callable[[], None]] = {}

        self._cancelled: bool = False

        self._reason: str | None = None

        self._next_callback_id: int = 0

        self._lock: Lock = Lock()

    @property
    def cancelled(self) -> bool:
        """函数契约说明.

        功能: 执行 cancelled 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `bool`。
        """
        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> str | None:
        """函数契约说明.

        功能: 执行 reason 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str | None`。
        """
        with self._lock:
            return self._reason

    def cancel(self, *, reason: str) -> bool:
        """函数契约说明.

        功能: 执行 cancel 的同步逻辑,并协调 tuple,
        callback, values。
        参数: self 表示当前实例。 reason: str。
        必填。
        契约: 同步调用。 返回 `bool`。
        """
        with self._lock:
            if self._cancelled:
                return False

            self._cancelled = True

            self._reason = reason

            callbacks = tuple(self._callbacks.values())

        for callback in callbacks:
            callback()

        return True

    def bind(self, callback: Callable[[], None]) -> Callable[[], None]:
        """函数契约说明.

        功能: 执行 bind 的同步逻辑,并协调 callback,
        pop。
        参数: self 表示当前实例。 callback:
        Callable[[], None]。 必填。
        契约: 同步调用。 返回 `Callable[[],
        None]`。
        """
        with self._lock:
            if self._cancelled:
                _ = callback()

                return _noop

            callback_id = self._next_callback_id

            self._next_callback_id += 1

            self._callbacks[callback_id] = callback

        def release() -> None:
            """函数契约说明.

            功能: 执行 release 的同步逻辑,并协调
            pop。
            参数: 无显式业务参数。
            契约: 同步调用。 返回 `None`。
            """
            with self._lock:
                _ = self._callbacks.pop(callback_id, None)

        return release


def post_sse(
    request: ProviderRequest,
    deadlines: ProviderDeadlines,
    cancellation: ProviderCancellationHandle | None,
) -> Iterator[str]:
    """函数契约说明.

    功能: 执行 post_sse 的同步逻辑,并协调
    _connection, monotonic, bind,
    _raise_non_success。
    参数: request: ProviderRequest。 必填。
    deadlines: ProviderDeadlines。 必填。
    cancellation:
    ProviderCancellationHandle | None。
    必填。
    契约: 同步调用。 返回迭代或生成器协议。 返回
    `Iterator[str]`。 可能抛出
    ProviderResponseError。
    """
    connection = _connection(request.url, deadlines, request.stage)

    release = _noop if cancellation is None else cancellation.bind(connection.close)

    response_release = _noop

    started = time.monotonic()

    try:
        if cancellation is not None and cancellation.cancelled:
            return

        try:
            connection.request(
                "POST", _path(request.url), body=request.body, headers=request.headers
            )

            if cancellation is not None and cancellation.cancelled:
                return b""

            _set_read_timeout(connection, deadlines.read_seconds)

            response = connection.getresponse()

        except (OSError, TimeoutError) as error:
            if cancellation is not None and cancellation.cancelled:
                return

            raise ProviderResponseError(
                stage=request.stage, reason="connect"
            ) from error

        _raise_non_success(response, request.stage)

        response_release = (
            _noop if cancellation is None else cancellation.bind(response.close)
        )

        try:
            yield from _sse_lines(
                response, deadlines, cancellation, request.stage, started
            )

        finally:
            response_release()

    finally:
        release()

        connection.close()


def post_bytes(
    request: ProviderRequest,
    deadlines: ProviderDeadlines,
    cancellation: ProviderCancellationHandle | None,
) -> bytes:
    """函数契约说明.

    功能: 执行 post_bytes 的同步逻辑,并协调
    _connection, monotonic, bind,
    _raise_non_success。
    参数: request: ProviderRequest。 必填。
    deadlines: ProviderDeadlines。 必填。
    cancellation:
    ProviderCancellationHandle | None。
    必填。
    契约: 同步调用。 返回 `bytes`。 可能抛出
    ProviderResponseError。
    """
    connection = _connection(request.url, deadlines, request.stage)

    release = _noop if cancellation is None else cancellation.bind(connection.close)

    response_release = _noop

    started = time.monotonic()

    try:
        if cancellation is not None and cancellation.cancelled:
            return b""

        try:
            connection.request(
                "POST", _path(request.url), body=request.body, headers=request.headers
            )

            _set_read_timeout(connection, deadlines.read_seconds)

            response = connection.getresponse()

            response_release = (
                _noop if cancellation is None else cancellation.bind(response.close)
            )

            try:
                data = response.read()

            except (AttributeError, ResponseNotReady):
                if cancellation is not None and cancellation.cancelled:
                    return b""

                raise

        except (OSError, TimeoutError, ProviderResponseError) as error:
            if cancellation is not None and cancellation.cancelled:
                return b""

            if isinstance(error, ProviderResponseError):
                raise

            raise ProviderResponseError(stage=request.stage, reason="read") from error

        _raise_non_success(response, request.stage)

        if time.monotonic() - started > deadlines.total_seconds:
            raise ProviderResponseError(stage=request.stage, reason="total")

        return data

    finally:
        response_release()

        release()

        connection.close()


def _sse_lines(
    response: HTTPResponse,
    deadlines: ProviderDeadlines,
    cancellation: ProviderCancellationHandle | None,
    stage: str,
    started: float,
) -> Iterator[str]:
    """函数契约说明.

    功能: 执行 _sse_lines 的同步逻辑,并协调
    _decode_sse_data,
    ProviderResponseError, readline,
    monotonic。
    参数: response: HTTPResponse。 必填。
    deadlines: ProviderDeadlines。 必填。
    cancellation:
    ProviderCancellationHandle | None。
    必填。 stage: str。 必填。 started: float。
    必填。
    契约: 同步调用。 返回迭代或生成器协议。 返回
    `Iterator[str]`。 可能抛出
    ProviderResponseError。
    """
    while True:
        if cancellation is not None and cancellation.cancelled:
            return

        if deadlines.total_seconds - (time.monotonic() - started) <= 0:
            raise ProviderResponseError(stage=stage, reason="total")

        try:
            line = response.readline()

        except (OSError, TimeoutError) as error:
            if cancellation is not None and cancellation.cancelled:
                return

            raise ProviderResponseError(stage=stage, reason="read") from error

        if line == b"":
            return

        decoded = _decode_sse_data(line, stage)

        if decoded is not None:
            yield decoded


def _raise_non_success(response: HTTPResponse, stage: str) -> None:
    """函数契约说明.

    功能: 执行 _raise_non_success 的同步逻辑,并协调
    ProviderResponseError。
    参数: response: HTTPResponse。 必填。
    stage: str。 必填。
    契约: 同步调用。 返回 `None`。 可能抛出
    ProviderResponseError。
    """
    if response.status < _SUCCESS_MIN or response.status >= _SUCCESS_MAX:
        raise ProviderResponseError(stage=stage, reason=f"status_{response.status}")


def _connection(
    url: str, deadlines: ProviderDeadlines, stage: str
) -> HTTPConnection | HTTPSConnection:
    """函数契约说明.

    功能: 执行 _connection 的同步逻辑,并协调
    urlsplit, HTTPConnection,
    HTTPSConnection,
    ProviderResponseError。
    参数: url: str。 必填。 deadlines:
    ProviderDeadlines。 必填。 stage: str。
    必填。
    契约: 同步调用。 返回 `HTTPConnection |
    HTTPSConnection`。 可能抛出
    ProviderResponseError。
    """
    parsed = urlsplit(url)

    match parsed.scheme:
        case "http":
            return HTTPConnection(parsed.netloc, timeout=deadlines.connect_seconds)

        case "https":
            return HTTPSConnection(parsed.netloc, timeout=deadlines.connect_seconds)

        case _:
            raise ProviderResponseError(stage=stage, reason="endpoint")


def _path(url: str) -> str:
    """函数契约说明.

    功能: 执行 _path 的同步逻辑,并协调 urlsplit。
    参数: url: str。 必填。
    契约: 同步调用。 返回 `str`。
    """
    parsed = urlsplit(url)

    path = parsed.path if parsed.path != "" else "/"

    return path if parsed.query == "" else f"{path}?{parsed.query}"


def _set_read_timeout(
    connection: HTTPConnection | HTTPSConnection, seconds: float
) -> None:
    """函数契约说明.

    功能: 执行 _set_read_timeout 的同步逻辑,并协调
    settimeout, ProviderResponseError。
    参数: connection: HTTPConnection |
    HTTPSConnection。 必填。 seconds: float。
    必填。
    契约: 同步调用。 返回 `None`。 可能抛出
    ProviderResponseError。
    """
    if connection.sock is None:
        raise ProviderResponseError(stage="transport", reason="connection")

    _ = connection.sock.settimeout(seconds)


def _decode_sse_data(line: bytes, stage: str) -> str | None:
    """函数契约说明.

    功能: 执行 _decode_sse_data 的同步逻辑,并协调
    lstrip, rstrip, startswith,
    ProviderResponseError。
    参数: line: bytes。 必填。 stage: str。 必填。
    契约: 同步调用。 返回 `str | None`。 可能抛出
    ProviderResponseError。
    """
    try:
        decoded = line.decode().rstrip("\r\n")

    except UnicodeDecodeError as error:
        raise ProviderResponseError(stage=stage, reason="encoding") from error

    if decoded == "" or decoded.startswith(":"):
        return None

    if not decoded.startswith("data:"):
        raise ProviderResponseError(stage=stage, reason="sse")

    return decoded.removeprefix("data:").lstrip()


def _noop() -> None:
    """函数契约说明.

    功能: 执行 _noop 的同步逻辑,并维持签名契约。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """
