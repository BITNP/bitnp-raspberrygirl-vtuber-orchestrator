"""模块契约说明.

职责: 提供 orchestrator.media_adapters
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPSConnection
from typing import TYPE_CHECKING, Literal, TypedDict, override
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterator


from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import (
    ProviderCancellationHandle,
    ProviderCapability,
    ProviderDeadlines,
    ProviderRequest,
    ProviderResponseError,
    post_bytes,
    post_sse,
)


@dataclass(frozen=True, slots=True)
class MediaAdapterConfigError(ValueError):
    """类契约说明.

    职责: 保存 MediaAdapterConfigError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: field_name。 方法: __str__。
    """

    field_name: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"media adapter config field is blank: {self.field_name}"


@dataclass(frozen=True, slots=True)
class ASRPartialEvent:
    """类契约说明.

    职责: 保存 ASRPartialEvent
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    text、received_at_ms、segment_id、seq。
    """

    text: str

    received_at_ms: int

    segment_id: str

    seq: int


type ASRStreamEvent = ASRPartialEvent | ASRAudienceEvent


@dataclass(frozen=True, slots=True)
class ASRStreamRequest:
    """类契约说明.

    职责: 保存 ASRStreamRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: audio、filename、received_at_m
    s、segment_id、seq。
    """

    audio: bytes

    filename: str

    received_at_ms: int

    segment_id: str

    seq: int


class VllmOmniSpeechPayload(TypedDict):
    """类契约说明.

    职责: 定义 VllmOmniSpeechPayload
    的状态、行为和对外协作边界。
    契约: 字段: model、input、voice、task_type、
    ref_audio、ref_text。
    """

    model: str

    input: str

    voice: str

    task_type: Literal["Base"]

    ref_audio: str

    ref_text: str


@dataclass(frozen=True, slots=True)
class HttpSpeechRequest:
    """类契约说明.

    职责: 保存 HttpSpeechRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: method、url、json。
    """

    method: Literal["POST"]

    url: str

    json: VllmOmniSpeechPayload


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    """类契约说明.

    职责: 保存 SynthesizedAudio
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: data、media_type。
    """

    data: bytes

    media_type: str


@dataclass(frozen=True, slots=True)
class OpenAICompatibleASRAdapter:
    """类契约说明.

    职责: 保存 OpenAICompatibleASRAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: endpoint、model、api_key、capab
    ility、deadlines。 方法: __post_init__、n
    ormalize_final、transcribe、_transcrib
    e_request、stream、_stream_openai。
    """

    endpoint: str

    model: str

    api_key: str | None = None

    capability: ProviderCapability = "final_only"

    deadlines: ProviderDeadlines = field(default_factory=ProviderDeadlines)

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化
        OpenAICompatibleASRAdapter
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        _require_endpoint_and_model(self.endpoint, self.model)

    def normalize_final(
        self,
        *,
        response: dict[str, str],
        received_at_ms: int,
        segment_id: str,
        seq: int,
    ) -> ASRAudienceEvent:
        """函数契约说明.

        功能: 执行 normalize_final 的同步逻辑,并协调
        strip, ASRAudienceEvent,
        MediaAdapterConfigError, get。
        参数: self 表示当前实例。 response:
        dict[str, str]。 必填。
        received_at_ms: int。 必填。
        segment_id: str。 必填。 seq: int。
        必填。
        契约: 同步调用。 返回 `ASRAudienceEvent`。
        可能抛出 MediaAdapterConfigError。
        """
        text = response.get("text", "").strip()

        if text == "":
            raise MediaAdapterConfigError(field_name="response.text")

        return ASRAudienceEvent(text, received_at_ms, segment_id, seq)

    def transcribe(  # noqa: PLR0913
        self,
        *,
        audio: bytes,
        filename: str,
        received_at_ms: int,
        segment_id: str,
        seq: int,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> ASRAudienceEvent:
        """函数契约说明.

        功能: 执行 transcribe 的同步逻辑,并协调
        _transcribe_request,
        ASRStreamRequest,
        MediaAdapterConfigError。
        参数: self 表示当前实例。 audio: bytes。
        必填。 filename: str。 必填。
        received_at_ms: int。 必填。
        segment_id: str。 必填。 seq: int。
        必填。 cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回 `ASRAudienceEvent`。
        可能抛出 MediaAdapterConfigError。
        """
        event = self._transcribe_request(
            ASRStreamRequest(audio, filename, received_at_ms, segment_id, seq),
            cancellation=cancellation,
        )

        if event is None:
            raise MediaAdapterConfigError(field_name="cancellation")

        return event

    def _transcribe_request(
        self,
        request: ASRStreamRequest,
        *,
        cancellation: ProviderCancellationHandle | None,
    ) -> ASRAudienceEvent | None:
        """函数契约说明.

        功能: 执行 _transcribe_request
        的同步逻辑,并协调
        _multipart_transcription_body,
        post_bytes, get,
        normalize_final。
        参数: self 表示当前实例。 request:
        ASRStreamRequest。 必填。
        cancellation:
        ProviderCancellationHandle |
        None。 必填。
        契约: 同步调用。 返回 `ASRAudienceEvent |
        None`。 可能抛出
        MediaAdapterConfigError。
        """
        boundary = "orchestrator-asr-boundary"

        body = _multipart_transcription_body(
            boundary, self.model, request.filename, request.audio
        )

        response = post_bytes(
            ProviderRequest(
                f"{self.endpoint.rstrip('/')}/audio/transcriptions",
                body,
                _headers(self.api_key, f"multipart/form-data; boundary={boundary}"),
                "asr",
            ),
            deadlines=self.deadlines,
            cancellation=cancellation,
        )

        if cancellation is not None and cancellation.cancelled:
            return None

        try:
            payload = parse_json_value(response.decode())

        except JsonBoundaryError as error:
            raise MediaAdapterConfigError(field_name=error.field_name) from error

        if not isinstance(payload, dict):
            raise MediaAdapterConfigError(field_name="response")

        text = payload.get("text")

        if not isinstance(text, str):
            raise MediaAdapterConfigError(field_name="response.text")

        return self.normalize_final(
            response={"text": text},
            received_at_ms=request.received_at_ms,
            segment_id=request.segment_id,
            seq=request.seq,
        )

    def stream(
        self,
        request: ASRStreamRequest,
        *,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[ASRStreamEvent]:
        """函数契约说明.

        功能: 执行 stream 的同步逻辑,并协调
        _transcribe_request,
        _stream_openai。
        参数: self 表示当前实例。 request:
        ASRStreamRequest。 必填。
        cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回迭代或生成器协议。 返回
        `Iterator[ASRStreamEvent]`。
        """
        match self.capability:
            case "final_only":
                if cancellation is None or not cancellation.cancelled:
                    event = self._transcribe_request(
                        request,
                        cancellation=cancellation,
                    )

                    if event is not None:
                        yield event

            case "streaming":
                yield from self._stream_openai(
                    request=request,
                    cancellation=cancellation,
                )

    def _stream_openai(
        self,
        *,
        request: ASRStreamRequest,
        cancellation: ProviderCancellationHandle | None,
    ) -> Iterator[ASRStreamEvent]:
        """函数契约说明.

        功能: 执行 _stream_openai 的同步逻辑,并协调
        _multipart_transcription_body,
        post_sse, ProviderRequest,
        _normalize_asr_sse。
        参数: self 表示当前实例。 request:
        ASRStreamRequest。 必填。
        cancellation:
        ProviderCancellationHandle |
        None。 必填。
        契约: 同步调用。 返回迭代或生成器协议。 返回
        `Iterator[ASRStreamEvent]`。 可能抛出
        ProviderResponseError。
        """
        boundary = "orchestrator-asr-boundary"

        body = _multipart_transcription_body(
            boundary, self.model, request.filename, request.audio
        )

        final_emitted = False

        for data in post_sse(
            ProviderRequest(
                f"{self.endpoint.rstrip('/')}/audio/transcriptions",
                body,
                _headers(self.api_key, f"multipart/form-data; boundary={boundary}"),
                "asr",
            ),
            deadlines=self.deadlines,
            cancellation=cancellation,
        ):
            if data == "[DONE]":
                break

            event = _normalize_asr_sse(
                data=data,
                received_at_ms=request.received_at_ms,
                segment_id=request.segment_id,
                seq=request.seq,
            )

            match event:
                case ASRPartialEvent():
                    yield event

                case ASRAudienceEvent():
                    if final_emitted:
                        raise ProviderResponseError(
                            stage="asr", reason="duplicate_final"
                        )

                    final_emitted = True

                    yield event

        if (cancellation is None or not cancellation.cancelled) and not final_emitted:
            raise ProviderResponseError(stage="asr", reason="missing_final")


@dataclass(frozen=True, slots=True)
class VllmOmniTTSAdapter:
    """类契约说明.

    职责: 保存 VllmOmniTTSAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: endpoint、model、api_key。 方法:
    __post_init__、build_speech_request、s
    ynthesize。
    """

    endpoint: str

    model: str

    api_key: str | None = None

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化 VllmOmniTTSAdapter
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        _require_endpoint_and_model(self.endpoint, self.model)

    def build_speech_request(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
    ) -> HttpSpeechRequest:
        """函数契约说明.

        功能: 构造协议对象、配置或测试夹具。
        参数: self 表示当前实例。 text: str。 必填。
        voice: str。 必填。 ref_audio: str。
        必填。 ref_text: str。 必填。
        契约: 同步调用。 返回
        `HttpSpeechRequest`。
        """
        return HttpSpeechRequest(
            method="POST",
            url=f"{self.endpoint.rstrip('/')}/audio/speech",
            json={
                "model": self.model.strip(),
                "input": text,
                "voice": voice,
                "task_type": "Base",
                "ref_audio": ref_audio,
                "ref_text": ref_text,
            },
        )

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:
        """函数契约说明.

        功能: 执行 synthesize 的同步逻辑,并协调
        build_speech_request, _post,
        SynthesizedAudio, encode。
        参数: self 表示当前实例。 text: str。 必填。
        voice: str。 必填。 ref_audio: str。
        必填。 ref_text: str。 必填。
        cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回 `SynthesizedAudio`。
        """
        speech = self.build_speech_request(
            text=text,
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

        response = _post(
            speech.url,
            json.dumps(speech.json).encode(),
            _headers(self.api_key, "application/json"),
            cancellation,
        )

        return SynthesizedAudio(data=response.data, media_type=response.media_type)


def _require_endpoint_and_model(endpoint: str, model: str) -> None:
    """函数契约说明.

    功能: 执行 _require_endpoint_and_model
    的同步逻辑,并协调 strip,
    MediaAdapterConfigError。
    参数: endpoint: str。 必填。 model: str。
    必填。
    契约: 同步调用。 返回 `None`。 可能抛出
    MediaAdapterConfigError。
    """
    if endpoint.strip() == "":
        raise MediaAdapterConfigError(field_name="endpoint")

    if model.strip() == "":
        raise MediaAdapterConfigError(field_name="model")


def _headers(api_key: str | None, content_type: str) -> dict[str, str]:
    """函数契约说明.

    功能: 执行 _headers 的同步逻辑,并协调 strip。
    参数: api_key: str | None。 必填。
    content_type: str。 必填。
    契约: 同步调用。 返回 `dict[str, str]`。
    """
    headers = {"Content-Type": content_type}

    if api_key is not None and api_key.strip() != "":
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    return headers


def _normalize_asr_sse(
    *, data: str, received_at_ms: int, segment_id: str, seq: int
) -> ASRStreamEvent:
    """函数契约说明.

    功能: 执行 _normalize_asr_sse 的同步逻辑,并协调
    get, ASRPartialEvent,
    parse_json_value, isinstance。
    参数: data: str。 必填。 received_at_ms:
    int。 必填。 segment_id: str。 必填。 seq:
    int。 必填。
    契约: 同步调用。 返回 `ASRStreamEvent`。 可能抛出
    ProviderResponseError。
    """
    try:
        payload = parse_json_value(data)

    except JsonBoundaryError as error:
        raise ProviderResponseError(stage="asr", reason="json") from error

    if not isinstance(payload, dict):
        raise ProviderResponseError(stage="asr", reason="event")

    text = payload.get("text")

    is_final = payload.get("is_final")

    if (
        not isinstance(text, str)
        or text.strip() == ""
        or not isinstance(is_final, bool)
    ):
        raise ProviderResponseError(stage="asr", reason="event")

    if is_final:
        return ASRAudienceEvent(text.strip(), received_at_ms, segment_id, seq)

    return ASRPartialEvent(text.strip(), received_at_ms, segment_id, seq)


@dataclass(frozen=True, slots=True)
class _HttpResponse:
    """类契约说明.

    职责: 保存 _HttpResponse
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: data、media_type。
    """

    data: bytes

    media_type: str


def _post(
    url: str,
    body: bytes,
    headers: dict[str, str],
    cancellation: ProviderCancellationHandle | None,
) -> _HttpResponse:
    """函数契约说明.

    功能: 执行 _post 的同步逻辑,并协调 urlsplit,
    HTTPConnection, bind, request。
    参数: url: str。 必填。 body: bytes。 必填。
    headers: dict[str, str]。 必填。
    cancellation:
    ProviderCancellationHandle | None。
    必填。
    契约: 同步调用。 返回 `_HttpResponse`。 可能抛出
    MediaAdapterConfigError。
    """
    parsed = urlsplit(url)

    path = parsed.path if parsed.path != "" else "/"

    if parsed.query != "":
        path = f"{path}?{parsed.query}"

    if parsed.scheme == "http":
        connection: HTTPConnection | HTTPSConnection = HTTPConnection(
            parsed.netloc,
            timeout=30,
        )

    elif parsed.scheme == "https":
        connection = HTTPSConnection(parsed.netloc, timeout=30)

    else:
        raise MediaAdapterConfigError(field_name="endpoint")

    release = None if cancellation is None else cancellation.bind(connection.close)

    try:
        if cancellation is not None and cancellation.cancelled:
            return _HttpResponse(data=b"", media_type="application/octet-stream")

        connection.request("POST", path, body=body, headers=headers)

        response = connection.getresponse()

        content_type = response.getheader("Content-Type", "application/octet-stream")

        media_type = content_type.split(";", 1)[0]

        return _HttpResponse(data=response.read(), media_type=media_type)

    finally:
        if release is not None:
            release()

        connection.close()


def _multipart_transcription_body(
    boundary: str,
    model: str,
    filename: str,
    audio: bytes,
) -> bytes:
    """函数契约说明.

    功能: 执行 _multipart_transcription_body
    的同步逻辑,并协调 encode。
    参数: boundary: str。 必填。 model: str。
    必填。 filename: str。 必填。 audio: bytes。
    必填。
    契约: 同步调用。 返回 `bytes`。
    """
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{model}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()

    return prefix + audio + f"\r\n--{boundary}--\r\n".encode()
