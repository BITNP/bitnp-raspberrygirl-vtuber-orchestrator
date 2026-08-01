from __future__ import annotations

import base64
import json
import logging
import ssl
from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, override
from urllib.parse import unquote, urlsplit

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    Stream,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from openai.types.audio import TranscriptionStreamEvent


from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import (
    ProviderCancellationHandle,
    ProviderCapability,
    ProviderDeadlines,
    ProviderRequest,
    ProviderResponseError,
    post_sse,
)
from orchestrator.tls import build_tls_context
from orchestrator.tts_rtp import Pcm16leChunk

_LOGGER = logging.getLogger(__name__)


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

    ca_path: Path | None = None

    def __post_init__(self) -> None:
        _require_endpoint_and_model(self.endpoint, self.model)

    def normalize_final(
        self,
        *,
        response: dict[str, str],
        received_at_ms: int,
        segment_id: str,
        seq: int,
    ) -> ASRAudienceEvent | None:
        text = response.get("text", "").strip()

        if text == "":
            return None

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
    ) -> ASRAudienceEvent | None:
        return self._transcribe_request(
            ASRStreamRequest(audio, filename, received_at_ms, segment_id, seq),
            cancellation=cancellation,
        )

    def _transcribe_request(
        self,
        request: ASRStreamRequest,
        *,
        cancellation: ProviderCancellationHandle | None,
    ) -> ASRAudienceEvent | None:
        _LOGGER.debug(
            "asr_request endpoint=%s model=%s segment=%s audio_bytes=%d filename=%s",
            self.endpoint,
            self.model,
            request.segment_id,
            len(request.audio),
            request.filename,
        )

        client = self._client()
        release = _bind_cancellation(cancellation, client.close)
        try:
            response = client.audio.transcriptions.create(
                model=self.model,
                file=(request.filename, request.audio, "application/octet-stream"),
                timeout=self._timeout(),
            )
            if cancellation is not None and cancellation.cancelled:
                return None
            text = response.text
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
            json.JSONDecodeError,
        ) as error:
            if cancellation is not None and cancellation.cancelled:
                return None
            raise _asr_provider_error(error) from error
        finally:
            release()
            client.close()

        event = self.normalize_final(
            response={"text": text},
            received_at_ms=request.received_at_ms,
            segment_id=request.segment_id,
            seq=request.seq,
        )
        if event is None:
            _LOGGER.debug("asr_response kind=empty segment=%s", request.segment_id)
        else:
            _LOGGER.debug(
                "asr_response kind=final segment=%s chars=%d",
                event.segment_id,
                len(event.text),
            )
        return event

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

    def _stream_openai(  # noqa: C901, PLR0912
        self,
        *,
        request: ASRStreamRequest,
        cancellation: ProviderCancellationHandle | None,
    ) -> Iterator[ASRStreamEvent]:
        client = self._client()
        release = _bind_cancellation(cancellation, client.close)
        stream: Stream[TranscriptionStreamEvent] | None = None
        try:
            stream = client.audio.transcriptions.create(
                model=self.model,
                file=(request.filename, request.audio, "application/octet-stream"),
                stream=True,
                timeout=self._timeout(),
            )
            stream_release = _bind_cancellation(cancellation, stream.close)
            try:
                final_emitted = False
                for event in stream:
                    if cancellation is not None and cancellation.cancelled:
                        return
                    match event.type:
                        case "transcript.text.delta":
                            text = event.delta.strip()
                            if text != "":
                                yield ASRPartialEvent(
                                    text,
                                    request.received_at_ms,
                                    request.segment_id,
                                    request.seq,
                                )
                        case "transcript.text.done":
                            if final_emitted:
                                raise ProviderResponseError(
                                    stage="asr", reason="duplicate_final"
                                )
                            final_emitted = True
                            text = event.text.strip()
                            if text != "":
                                yield ASRAudienceEvent(
                                    text,
                                    request.received_at_ms,
                                    request.segment_id,
                                    request.seq,
                                )
                if (
                    cancellation is None or not cancellation.cancelled
                ) and not final_emitted:
                    raise ProviderResponseError(stage="asr", reason="missing_final")
            finally:
                stream_release()
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
            json.JSONDecodeError,
        ) as error:
            if cancellation is not None and cancellation.cancelled:
                return
            raise _asr_provider_error(error) from error
        finally:
            if stream is not None:
                stream.close()
            release()
            client.close()

    def _client(self) -> OpenAI:
        verify: bool | ssl.SSLContext = (
            True
            if self.ca_path is None
            else ssl.create_default_context(cafile=self.ca_path)
        )
        timeout = self._timeout()
        return OpenAI(
            api_key=self.api_key or "not-needed-for-openai-compatible-asr",
            base_url=f"{self.endpoint.rstrip('/')}/",
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(verify=verify, timeout=timeout, trust_env=False),
        )

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=self.deadlines.total_seconds,
            connect=self.deadlines.connect_seconds,
            read=self.deadlines.read_seconds,
            write=self.deadlines.total_seconds,
        )


@dataclass(frozen=True, slots=True)
class VllmOmniTTSAdapter:
    endpoint: str

    model: str

    api_key: str | None = None

    ca_path: Path | None = None

    timeout_seconds: float = 120.0

    capability: Literal["final_only", "streaming_sse"] = "final_only"

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
                "ref_audio": _portable_reference_audio(ref_audio),
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
            self.ca_path,
            self.timeout_seconds,
        )

        return SynthesizedAudio(data=response.data, media_type=response.media_type)

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[Pcm16leChunk]:
        """Consume vLLM-Omni speech.audio SSE events without buffering a clip."""
        speech = self.build_speech_request(
            text=text,
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        payload = dict(speech.json)
        payload.update(
            {
                "stream": True,
                "stream_format": "sse",
                "response_format": "pcm",
                "speed": 1.0,
            }
        )
        done = False
        resampler = _Pcm24khzTo16khzResampler()
        for data in post_sse(
            ProviderRequest(
                speech.url,
                json.dumps(payload).encode(),
                _headers(self.api_key, "application/json"),
                "tts",
                self.ca_path,
            ),
            deadlines=ProviderDeadlines(
                connect_seconds=5.0,
                read_seconds=self.timeout_seconds,
                total_seconds=self.timeout_seconds,
            ),
            cancellation=cancellation,
        ):
            if cancellation is not None and cancellation.cancelled:
                return
            chunk = _normalize_tts_sse(data)
            if chunk is None:
                done = True
                break
            converted = resampler.push(chunk)
            if converted:
                yield Pcm16leChunk(converted)
        if (cancellation is None or not cancellation.cancelled) and not done:
            raise ProviderResponseError(stage="tts", reason="missing_done")


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


def _bind_cancellation(
    cancellation: ProviderCancellationHandle | None,
    callback: Callable[[], None],
) -> Callable[[], None]:
    if cancellation is None:
        return lambda: None
    return cancellation.bind(callback)


def _asr_provider_error(
    error: APIError | httpx.HTTPError | json.JSONDecodeError,
) -> ProviderResponseError:
    if isinstance(error, APIStatusError):
        return ProviderResponseError(stage="asr", reason=f"status_{error.status_code}")
    if isinstance(error, (APITimeoutError, httpx.TimeoutException)):
        return ProviderResponseError(stage="asr", reason="read")
    if isinstance(error, APIConnectionError):
        return ProviderResponseError(stage="asr", reason="connect")
    return ProviderResponseError(stage="asr", reason="response")


def _portable_reference_audio(ref_audio: str) -> str:
    stripped = ref_audio.strip()

    if stripped.startswith("data:"):
        return stripped

    parsed = urlsplit(stripped)

    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))

        # A configured provider may own a mounted reference file.  Preserve the
        # URI when it is not locally available instead of failing the whole TTS
        # request before the provider can return a typed error.
        return _audio_data_url(path) if path.is_file() else ref_audio

    if parsed.scheme == "" and Path(stripped).is_absolute():
        path = Path(stripped)

        return _audio_data_url(path) if path.is_file() else ref_audio

    return ref_audio


def _audio_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")

    return f"data:audio/wav;base64,{encoded}"


def _normalize_tts_sse(data: str) -> bytes | None:
    """Return a PCM delta, ``None`` for done, or raise for a typed error."""
    try:
        payload = parse_json_value(data)
    except JsonBoundaryError as error:
        raise ProviderResponseError(stage="tts", reason="json") from error
    if not isinstance(payload, dict):
        raise ProviderResponseError(stage="tts", reason="event")
    match payload.get("type"):
        case "speech.audio.delta":
            encoded = payload.get("audio")
            if payload.get("response_format") != "pcm" or not isinstance(encoded, str):
                raise ProviderResponseError(stage="tts", reason="event")
            try:
                return base64.b64decode(encoded, validate=True)
            except ValueError as error:
                raise ProviderResponseError(stage="tts", reason="base64") from error
        case "speech.audio.done":
            return None
        case "speech.audio.error":
            raise ProviderResponseError(stage="tts", reason="server")
        case _:
            raise ProviderResponseError(stage="tts", reason="event")


class _Pcm24khzTo16khzResampler:
    """Linear 24 kHz → 16 kHz PCM16LE resampler preserving SSE boundaries.

    An SSE delta is an arbitrary byte partition, not an independently sampled
    clip.  Keep the fractional source position and one-sample look-ahead so
    joining separately received deltas is byte-identical to resampling their
    concatenation.
    """

    def __init__(self) -> None:
        self._samples: list[int] = []
        self._sample_offset: int = 0
        self._next_position_halves: int = 0

    def push(self, pcm: bytes) -> bytes:
        if len(pcm) % 2 != 0:
            raise ProviderResponseError(stage="tts", reason="incomplete_pcm")
        self._samples.extend(
            int.from_bytes(pcm[index : index + 2], "little", signed=True)
            for index in range(0, len(pcm), 2)
        )
        output = bytearray()
        end = self._sample_offset + len(self._samples)
        while True:
            source_index = self._next_position_halves // 2
            # Require interpolation look-ahead even at an integer position.
            # This keeps output count equal to floor(input_samples * 2 / 3).
            if source_index + 1 >= end:
                break
            left = self._samples[source_index - self._sample_offset]
            if self._next_position_halves % 2 == 0:
                value = left
            else:
                right = self._samples[source_index + 1 - self._sample_offset]
                value = (left + right) // 2
            output.extend(value.to_bytes(2, "little", signed=True))
            self._next_position_halves += 3

        next_source_index = self._next_position_halves // 2
        discard = max(0, next_source_index - self._sample_offset)
        if discard:
            del self._samples[:discard]
            self._sample_offset += discard
        return bytes(output)


@dataclass(frozen=True, slots=True)
class _HttpResponse:
    data: bytes

    media_type: str


def _post(  # noqa: PLR0913
    url: str,
    body: bytes,
    headers: dict[str, str],
    cancellation: ProviderCancellationHandle | None,
    ca_path: Path | None,
    timeout_seconds: float = 30.0,
) -> _HttpResponse:
    parsed = urlsplit(url)

    path = parsed.path if parsed.path != "" else "/"

    if parsed.query != "":
        path = f"{path}?{parsed.query}"

    if parsed.scheme == "http":
        connection: HTTPConnection | HTTPSConnection = HTTPConnection(
            parsed.netloc,
            timeout=timeout_seconds,
        )

    elif parsed.scheme == "https":
        context = build_tls_context(ca_path)
        if context is None:
            connection = HTTPSConnection(parsed.netloc, timeout=timeout_seconds)
        else:
            connection = HTTPSConnection(
                parsed.netloc, timeout=timeout_seconds, context=context
            )

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
