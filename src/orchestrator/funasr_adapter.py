"""模块契约说明.

职责: 提供 orchestrator.funasr_adapter
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.media_adapters import (
    ASRPartialEvent,
    ASRStreamEvent,
    ASRStreamRequest,
    MediaAdapterConfigError,
)
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import (
    ProviderCancellationHandle,
    ProviderDeadlines,
    ProviderResponseError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FunASRConnection(Protocol):
    """类契约说明.

    职责: 声明 _FunASRConnection
    协议接口,约束实现方必须提供的行为。
    契约: 方法: send、recv、close。
    """

    def send(self, message: str | bytes) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 message: str |
        bytes。 必填。
        契约: 同步调用。 返回 `None`。
        """
        ...

    def recv(self, timeout: float | None = None) -> str | bytes:
        """函数契约说明.

        功能: 执行 recv 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 timeout: float
        | None。 可省略。
        契约: 同步调用。 返回 `str | bytes`。
        """
        ...

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        ...


@dataclass(frozen=True, slots=True)
class FunASRWebSocketAdapter:
    """类契约说明.

    职责: 保存 FunASRWebSocketAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: endpoint、model、deadlines。
    方法: __post_init__、capability、transcr
    ibe、stream。
    """

    endpoint: str

    model: str

    deadlines: ProviderDeadlines = field(default_factory=ProviderDeadlines)

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化 FunASRWebSocketAdapter
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。 可能抛出
        MediaAdapterConfigError。
        """
        if self.endpoint.strip() == "":
            raise MediaAdapterConfigError(field_name="endpoint")

        if self.model.strip() == "":
            raise MediaAdapterConfigError(field_name="model")

    @property
    def capability(self) -> str:
        """函数契约说明.

        功能: 执行 capability 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return "streaming"

    def transcribe(  # noqa: PLR0913
        self,
        *,
        audio: bytes,
        filename: str,
        received_at_ms: int,
        segment_id: str,
        seq: int,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> ASRAudienceEvent | None:
        """函数契约说明.

        功能: 执行 transcribe 的同步逻辑,并协调
        stream, ASRStreamRequest。
        参数: self 表示当前实例。 audio: bytes。
        必填。 filename: str。 必填。
        received_at_ms: int。 必填。
        segment_id: str。 必填。 seq: int。
        必填。 cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回 `ASRAudienceEvent |
        None`。
        """
        final: ASRAudienceEvent | None = None

        for event in self.stream(
            ASRStreamRequest(audio, filename, received_at_ms, segment_id, seq),
            cancellation=cancellation,
        ):
            match event:
                case ASRPartialEvent():
                    continue

                case ASRAudienceEvent():
                    final = event

        return final

    def stream(
        self,
        request: ASRStreamRequest,
        *,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[ASRStreamEvent]:
        """函数契约说明.

        功能: 执行 stream 的同步逻辑,并协调 connect,
        bind, send, release。
        参数: self 表示当前实例。 request:
        ASRStreamRequest。 必填。
        cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回迭代或生成器协议。 返回
        `Iterator[ASRStreamEvent]`。
        """
        if cancellation is not None and cancellation.cancelled:
            return

        connection = connect(self.endpoint, open_timeout=self.deadlines.connect_seconds)

        release = _noop if cancellation is None else cancellation.bind(connection.close)

        try:
            if cancellation is not None and cancellation.cancelled:
                return

            connection.send(_start_message(request.filename))

            connection.send(request.audio)

            connection.send(_end_message())

            yield from _receive_events(
                connection, request, self.deadlines, cancellation
            )

        finally:
            release()

            connection.close()


def _start_message(filename: str) -> str:
    """函数契约说明.

    功能: 执行 _start_message 的同步逻辑,并协调
    dumps。
    参数: filename: str。 必填。
    契约: 同步调用。 返回 `str`。
    """
    return json.dumps(
        {
            "mode": "2pass",
            "chunk_size": [5, 10, 5],
            "wav_name": filename,
            "wav_format": "pcm",
            "audio_fs": 16_000,
            "is_speaking": True,
        },
        separators=(",", ":"),
    )


def _end_message() -> str:
    """函数契约说明.

    功能: 执行 _end_message 的同步逻辑,并协调 dumps。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `str`。
    """
    return json.dumps({"is_speaking": False}, separators=(",", ":"))


def _receive_events(
    connection: _FunASRConnection,
    request: ASRStreamRequest,
    deadlines: ProviderDeadlines,
    cancellation: ProviderCancellationHandle | None,
) -> Iterator[ASRStreamEvent]:
    """函数契约说明.

    功能: 执行 _receive_events 的同步逻辑,并协调
    monotonic,
    _normalize_funasr_message,
    isinstance, recv。
    参数: connection: _FunASRConnection。
    必填。 request: ASRStreamRequest。 必填。
    deadlines: ProviderDeadlines。 必填。
    cancellation:
    ProviderCancellationHandle | None。
    必填。
    契约: 同步调用。 返回迭代或生成器协议。 返回
    `Iterator[ASRStreamEvent]`。 可能抛出
    ProviderResponseError。
    """
    final_emitted = False

    started = time.monotonic()

    while True:
        try:
            message = connection.recv(timeout=deadlines.read_seconds)

        except (ConnectionClosed, OSError, TimeoutError) as error:
            if cancellation is not None and cancellation.cancelled:
                return

            if final_emitted:
                return

            raise ProviderResponseError(stage="asr", reason="read") from error

        if time.monotonic() - started > deadlines.total_seconds:
            raise ProviderResponseError(stage="asr", reason="total")

        event = _normalize_funasr_message(message, request)

        if event is None:
            continue

        if isinstance(event, ASRPartialEvent):
            yield event

            continue

        if final_emitted:
            raise ProviderResponseError(stage="asr", reason="duplicate_final")

        final_emitted = True

        yield event


def _normalize_funasr_message(
    message: str | bytes, request: ASRStreamRequest
) -> ASRStreamEvent | None:
    """函数契约说明.

    功能: 执行 _normalize_funasr_message
    的同步逻辑,并协调 get, strip,
    ASRPartialEvent, isinstance。
    参数: message: str | bytes。 必填。
    request: ASRStreamRequest。 必填。
    契约: 同步调用。 返回 `ASRStreamEvent |
    None`。 可能抛出 ProviderResponseError。
    """
    if not isinstance(message, str):
        raise ProviderResponseError(stage="asr", reason="event")

    try:
        payload = parse_json_value(message)

    except JsonBoundaryError as error:
        raise ProviderResponseError(stage="asr", reason="json") from error

    if not isinstance(payload, dict):
        raise ProviderResponseError(stage="asr", reason="event")

    text = payload.get("text")

    if not isinstance(text, str) or text.strip() == "":
        return None

    is_final = payload.get("is_final")

    if is_final is not None and not isinstance(is_final, bool):
        raise ProviderResponseError(stage="asr", reason="event")

    mode = payload.get("mode")

    if mode is not None and not isinstance(mode, str):
        raise ProviderResponseError(stage="asr", reason="event")

    normalized_text = text.strip()

    if is_final is True or mode == "2pass-offline":
        return ASRAudienceEvent(
            normalized_text,
            request.received_at_ms,
            request.segment_id,
            request.seq,
        )

    return ASRPartialEvent(
        normalized_text,
        request.received_at_ms,
        request.segment_id,
        request.seq,
    )


def _noop() -> None:
    """函数契约说明.

    功能: 执行 _noop 的同步逻辑,并维持签名契约。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """
    return
