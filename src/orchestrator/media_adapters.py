
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

    field_name: str

    @override
    def __str__(self) -> str:
        return f"media adapter config field is blank: {self.field_name}"


@dataclass(frozen=True, slots=True)
class ASRPartialEvent:

    text: str

    received_at_ms: int

    segment_id: str

    seq: int


type ASRStreamEvent = ASRPartialEvent | ASRAudienceEvent


@dataclass(frozen=True, slots=True)
class ASRStreamRequest:

    audio: bytes

    filename: str

    received_at_ms: int

    segment_id: str

    seq: int


class VllmOmniSpeechPayload(TypedDict):

    model: str

    input: str

    voice: str

    task_type: Literal["Base"]

    ref_audio: str

    ref_text: str


@dataclass(frozen=True, slots=True)
class HttpSpeechRequest:

    method: Literal["POST"]

    url: str

    json: VllmOmniSpeechPayload


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:

    data: bytes

    media_type: str


@dataclass(frozen=True, slots=True)
class OpenAICompatibleASRAdapter:

    endpoint: str

    model: str

    api_key: str | None = None

    capability: ProviderCapability = "final_only"

    deadlines: ProviderDeadlines = field(default_factory=ProviderDeadlines)

    def __post_init__(self) -> None:
        _require_endpoint_and_model(self.endpoint, self.model)

    def normalize_final(
        self,
        *,
        response: dict[str, str],
        received_at_ms: int,
        segment_id: str,
        seq: int,
    ) -> ASRAudienceEvent:
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

    endpoint: str

    model: str

    api_key: str | None = None

    def __post_init__(self) -> None:
        _require_endpoint_and_model(self.endpoint, self.model)

    def build_speech_request(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
    ) -> HttpSpeechRequest:
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
    if endpoint.strip() == "":
        raise MediaAdapterConfigError(field_name="endpoint")

    if model.strip() == "":
        raise MediaAdapterConfigError(field_name="model")


def _headers(api_key: str | None, content_type: str) -> dict[str, str]:
    headers = {"Content-Type": content_type}

    if api_key is not None and api_key.strip() != "":
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    return headers


def _normalize_asr_sse(
    *, data: str, received_at_ms: int, segment_id: str, seq: int
) -> ASRStreamEvent:
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

    data: bytes

    media_type: str


def _post(
    url: str,
    body: bytes,
    headers: dict[str, str],
    cancellation: ProviderCancellationHandle | None,
) -> _HttpResponse:
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
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{model}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()

    return prefix + audio + f"\r\n--{boundary}--\r\n".encode()
