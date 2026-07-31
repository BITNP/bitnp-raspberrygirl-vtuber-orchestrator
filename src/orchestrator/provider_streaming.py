
from __future__ import annotations

import time
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection, ResponseNotReady
from threading import Lock
from typing import TYPE_CHECKING, Literal, override
from urllib.parse import urlsplit

from orchestrator.tls import build_tls_context

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


ProviderCapability = Literal["streaming", "final_only"]

_SUCCESS_MIN = 200

_SUCCESS_MAX = 300


@dataclass(frozen=True, slots=True)
class ProviderDeadlines:

    connect_seconds: float = 5.0

    read_seconds: float = 30.0

    total_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class ProviderRequest:

    url: str

    body: bytes

    headers: dict[str, str]

    stage: str

    ca_path: Path | None = None


@dataclass(slots=True)
class ProviderResponseError(OSError):

    stage: str

    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.stage} provider response error: {self.reason}"


class ProviderCancellationHandle:

    def __init__(self) -> None:
        self._callbacks: dict[int, Callable[[], None]] = {}

        self._cancelled: bool = False

        self._reason: str | None = None

        self._next_callback_id: int = 0

        self._lock: Lock = Lock()

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, *, reason: str) -> bool:
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
        with self._lock:
            if self._cancelled:
                _ = callback()

                return _noop

            callback_id = self._next_callback_id

            self._next_callback_id += 1

            self._callbacks[callback_id] = callback

        def release() -> None:
            with self._lock:
                _ = self._callbacks.pop(callback_id, None)

        return release


def post_sse(
    request: ProviderRequest,
    deadlines: ProviderDeadlines,
    cancellation: ProviderCancellationHandle | None,
) -> Iterator[str]:
    connection = _connection(request.url, deadlines, request.stage, request.ca_path)

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
    connection = _connection(request.url, deadlines, request.stage, request.ca_path)

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
    if response.status < _SUCCESS_MIN or response.status >= _SUCCESS_MAX:
        raise ProviderResponseError(stage=stage, reason=f"status_{response.status}")


def _connection(
    url: str,
    deadlines: ProviderDeadlines,
    stage: str,
    ca_path: Path | None,
) -> HTTPConnection | HTTPSConnection:
    parsed = urlsplit(url)

    match parsed.scheme:
        case "http":
            return HTTPConnection(parsed.netloc, timeout=deadlines.connect_seconds)

        case "https":
            context = build_tls_context(ca_path)
            if context is None:
                return HTTPSConnection(parsed.netloc, timeout=deadlines.connect_seconds)

            return HTTPSConnection(
                parsed.netloc,
                timeout=deadlines.connect_seconds,
                context=context,
            )

        case _:
            raise ProviderResponseError(stage=stage, reason="endpoint")


def _path(url: str) -> str:
    parsed = urlsplit(url)

    path = parsed.path if parsed.path != "" else "/"

    return path if parsed.query == "" else f"{path}?{parsed.query}"


def _set_read_timeout(
    connection: HTTPConnection | HTTPSConnection, seconds: float
) -> None:
    if connection.sock is None:
        raise ProviderResponseError(stage="transport", reason="connection")

    _ = connection.sock.settimeout(seconds)


def _decode_sse_data(line: bytes, stage: str) -> str | None:
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
    ...
