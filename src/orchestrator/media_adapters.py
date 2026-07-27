"""Orchestrator-owned ASR and TTS provider boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from typing import Literal, TypedDict, override
from urllib.parse import urlsplit

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.pipeline_contracts import ASRAudienceEvent


@dataclass(frozen=True, slots=True)
class MediaAdapterConfigError(ValueError):
    """Raised when a media provider configuration value is blank or malformed."""

    field_name: str

    @override
    def __str__(self) -> str:
        return f"media adapter config field is blank: {self.field_name}"


class VllmOmniSpeechPayload(TypedDict):
    """Documented vLLM-Omni Qwen voice-cloning request body."""

    model: str
    input: str
    voice: str
    task_type: Literal["Base"]
    ref_audio: str
    ref_text: str


@dataclass(frozen=True, slots=True)
class HttpSpeechRequest:
    """Typed HTTP request data for an OpenAI-compatible speech endpoint."""

    method: Literal["POST"]
    url: str
    json: VllmOmniSpeechPayload


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    """Audio bytes returned by a TTS provider, before RTP routing."""

    data: bytes
    media_type: str


@dataclass(frozen=True, slots=True)
class OpenAICompatibleASRAdapter:
    """OpenAI-compatible ASR boundary that emits Orchestrator input records."""

    endpoint: str
    model: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate endpoint and model before a request can be built."""
        _require_endpoint_and_model(self.endpoint, self.model)

    def normalize_final(
        self,
        *,
        response: dict[str, str],
        received_at_ms: int,
        segment_id: str,
        seq: int,
    ) -> ASRAudienceEvent:
        """Normalize a completed provider response into the input boundary."""
        text = response.get("text", "").strip()
        if text == "":
            raise MediaAdapterConfigError(field_name="response.text")
        return ASRAudienceEvent(text, received_at_ms, segment_id, seq)

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        received_at_ms: int,
        segment_id: str,
        seq: int,
    ) -> ASRAudienceEvent:
        """Submit audio to the configured provider and normalize its final text."""
        boundary = "orchestrator-asr-boundary"
        body = _multipart_transcription_body(boundary, self.model, filename, audio)
        response = _post(
            f"{self.endpoint.rstrip('/')}/audio/transcriptions",
            body,
            _headers(self.api_key, f"multipart/form-data; boundary={boundary}"),
        )
        try:
            payload = parse_json_value(response.data.decode())
        except JsonBoundaryError as error:
            raise MediaAdapterConfigError(field_name=error.field_name) from error
        if not isinstance(payload, dict):
            raise MediaAdapterConfigError(field_name="response")
        text = payload.get("text")
        if not isinstance(text, str):
            raise MediaAdapterConfigError(field_name="response.text")
        return self.normalize_final(
            response={"text": text},
            received_at_ms=received_at_ms,
            segment_id=segment_id,
            seq=seq,
        )


@dataclass(frozen=True, slots=True)
class VllmOmniTTSAdapter:
    """vLLM-Omni audio-speech boundary returning provider media bytes."""

    endpoint: str
    model: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate endpoint and model before a request can be built."""
        _require_endpoint_and_model(self.endpoint, self.model)

    def build_speech_request(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
    ) -> HttpSpeechRequest:
        """Build the documented vLLM-Omni Qwen cloning speech request."""
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
    ) -> SynthesizedAudio:
        """Request provider audio bytes without making RTP assumptions."""
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


@dataclass(frozen=True, slots=True)
class _HttpResponse:
    data: bytes
    media_type: str


def _post(url: str, body: bytes, headers: dict[str, str]) -> _HttpResponse:
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
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        content_type = response.getheader("Content-Type", "application/octet-stream")
        media_type = content_type.split(";", 1)[0]
        return _HttpResponse(data=response.read(), media_type=media_type)
    finally:
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
