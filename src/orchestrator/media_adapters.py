from __future__ import annotations

import base64
import io
import json
import logging
import ssl
import wave
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    override,
)
from urllib.parse import unquote, urlsplit

import dashscope
import httpx
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer
from openai import (
    NOT_GIVEN,
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    Stream,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from openai.types.audio import TranscriptionStreamEvent


from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.log_summary import (
    binary_summary,
    reference_audio_summary,
)
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import (
    ProviderCancellationHandle,
    ProviderCapability,
    ProviderDeadlines,
    ProviderResponseError,
)
from orchestrator.tts_rtp import Pcm16leChunk

_LOGGER = logging.getLogger(__name__)
_ALIYUN_REQUEST_LOG = (
    "tts_request provider=aliyun_cosyvoice transport=dashscope-realtime"
    " url=%s model=%s voice=%s mode=%s input=%r"
)
_ALIYUN_PCM_RESPONSE_LOG = (
    "tts_response provider=aliyun_cosyvoice transport=dashscope-realtime"
    " media_type=audio/pcm %s"
)
_ALIYUN_WAV_RESPONSE_LOG = (
    "tts_response provider=aliyun_cosyvoice transport=dashscope-realtime"
    " media_type=audio/wav %s"
)
_ALIYUN_COMPLETE_LOG = (
    "tts_response provider=aliyun_cosyvoice transport=dashscope-realtime"
    " outcome=complete request_id=%s first_package_delay_ms=%.1f"
)
_ALIYUN_ERROR_LOG = (
    "tts_response_error provider=aliyun_cosyvoice transport=dashscope-realtime"
    " stage=%s detail=%s"
)
_ALIYUN_STALLED_LOG = (
    "tts_response_error provider=aliyun_cosyvoice transport=dashscope-realtime"
    " reason=worker_stalled thread=%s"
)
_ALIYUN_SAMPLE_RATE = 16_000
_ALIYUN_CHANNELS = 1
_ALIYUN_SAMPLE_BYTES = 2
_ALIYUN_ERROR_CHARS = 512
_ALIYUN_COMPLETE_TIMEOUT_MILLIS = 60_000
_ALIYUN_CANCEL_TIMEOUT_MILLIS = 2_000
_ALIYUN_WORKER_JOIN_SECONDS = 5.0
_ALIYUN_SYNTHESIS_THREAD = "aliyun-cosyvoice-realtime"
_ALIYUN_WEBSOCKET_SCHEMES = frozenset({"ws", "wss"})
_ALIYUN_REALTIME_AUDIO_FORMAT = AudioFormat.PCM_16000HZ_MONO_16BIT

_PCM16_SAMPLE_BYTES = 2

_QWEN3TTSCPP_STREAM_RESPONSE_LOG = (
    "tts_response provider=qwen3ttscpp transport=http media_type=audio/pcm"
    " stream=chunked outcome=complete pcm16le_bytes=%d"
)


@dataclass(frozen=True, slots=True)
class MediaAdapterConfigError(ValueError):
    field_name: str

    reason: Literal["blank", "invalid"] = "blank"

    @override
    def __str__(self) -> str:
        match self.reason:
            case "invalid":
                return f"media adapter config field is invalid: {self.field_name}"

            case _:
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

    voice: NotRequired[str]


class VllmOmniExtensionParameters(TypedDict):
    task_type: Literal["Base"]

    ref_audio: str

    ref_text: str


class AudioCppSpeechPayload(TypedDict):
    model: str

    input: str

    response_format: NotRequired[Literal["wav", "pcm"]]

    stream_format: NotRequired[Literal["sse"]]

    voice: NotRequired[str]

    voice_ref: NotRequired[str]

    reference_text: NotRequired[str]


class Qwen3TtsCppSpeechPayload(TypedDict):
    model: str

    input: str

    response_format: Literal["wav", "pcm"]

    voice: NotRequired[str]


@dataclass(frozen=True, slots=True)
class HttpSpeechRequest:
    method: Literal["POST"]

    url: str

    json: VllmOmniSpeechPayload

    extra_body: VllmOmniExtensionParameters


@dataclass(frozen=True, slots=True)
class AudioCppSpeechRequest:
    method: Literal["POST"]

    url: str

    json: AudioCppSpeechPayload


@dataclass(frozen=True, slots=True)
class Qwen3TtsCppSpeechRequest:
    method: Literal["POST"]

    url: str

    json: Qwen3TtsCppSpeechPayload


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
                "asr_response kind=final segment=%s text=%r",
                event.segment_id,
                event.text,
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
                                _LOGGER.debug(
                                    "asr_response kind=partial segment=%s text=%r",
                                    request.segment_id,
                                    text,
                                )
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
                                _LOGGER.debug(
                                    "asr_response kind=final segment=%s text=%r",
                                    request.segment_id,
                                    text,
                                )
                                yield ASRAudienceEvent(
                                    text,
                                    request.received_at_ms,
                                    request.segment_id,
                                    request.seq,
                                )
                        case _:
                            continue
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

    capability: ProviderCapability = "final_only"

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
        # Qwen voice cloning selects the speaker solely from ref_audio/ref_text.
        # Its OpenAI-compatible endpoint rejects every named ``voice`` value.
        _ = voice
        return HttpSpeechRequest(
            method="POST",
            url=f"{self.endpoint.rstrip('/')}/audio/speech",
            json={
                "model": self.model.strip(),
                "input": text,
            },
            extra_body={
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
        _log_tts_request(speech)

        client = self._client()
        release = _bind_cancellation(cancellation, client.close)
        data = b""
        try:
            response = client.audio.speech.create(
                model=speech.json["model"],
                input=speech.json["input"],
                voice=cast("str", cast("object", NOT_GIVEN)),
                response_format="wav",
                extra_body=speech.extra_body,
                timeout=self.timeout_seconds,
            )
            if cancellation is not None and cancellation.cancelled:
                return SynthesizedAudio(data=b"", media_type="application/octet-stream")
            data = response.content
            _LOGGER.debug(
                "tts_response transport=http media_type=%s %s",
                "audio/wav",
                binary_summary(data),
            )
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
        ) as error:
            if cancellation is not None and cancellation.cancelled:
                return SynthesizedAudio(data=b"", media_type="application/octet-stream")
            raise _tts_provider_error(error) from error
        finally:
            release()
            client.close()
        return SynthesizedAudio(data=data, media_type="audio/wav")

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
        _log_tts_request(speech)
        client = self._client()
        release = _bind_cancellation(cancellation, client.close)
        try:
            # ``audio.speech.create`` buffers the complete HTTP response even
            # when the server selects SSE.  The streaming wrapper is required
            # to expose each event as it arrives from vLLM-Omni.
            with client.audio.speech.with_streaming_response.create(
                model=speech.json["model"],
                input=speech.json["input"],
                voice=cast("str", cast("object", NOT_GIVEN)),
                response_format="pcm",
                speed=1.0,
                stream_format="sse",
                extra_body={**speech.extra_body, "stream": True},
                timeout=self.timeout_seconds,
            ) as response:
                response_release = _bind_cancellation(cancellation, response.close)
                try:
                    done = False
                    resampler = _Pcm24khzTo16khzResampler()
                    for line in response.iter_lines():
                        if cancellation is not None and cancellation.cancelled:
                            return
                        data = _sse_data(line)
                        if data is None:
                            continue
                        chunk = _normalize_tts_sse(data)
                        if chunk is None:
                            done = True
                            break
                        converted = resampler.push(chunk)
                        if converted:
                            yield Pcm16leChunk(converted)
                    if (
                        cancellation is None or not cancellation.cancelled
                    ) and not done:
                        raise ProviderResponseError(stage="tts", reason="missing_done")
                finally:
                    response_release()
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
        ) as error:
            if cancellation is not None and cancellation.cancelled:
                return
            raise _tts_provider_error(error) from error
        finally:
            release()
            client.close()

    def _client(self) -> OpenAI:
        verify: bool | ssl.SSLContext = (
            True
            if self.ca_path is None
            else ssl.create_default_context(cafile=self.ca_path)
        )
        return OpenAI(
            api_key=self.api_key or "not-needed-for-vllm-omni",
            base_url=f"{self.endpoint.rstrip('/')}/",
            timeout=self.timeout_seconds,
            max_retries=0,
            http_client=httpx.Client(
                verify=verify,
                timeout=self.timeout_seconds,
                trust_env=False,
            ),
        )


class _RealtimeSpeechSynthesizer(Protocol):
    """DashScope realtime synthesizer surface this adapter drives."""

    def streaming_call(self, text: str) -> None:
        """Send synthesis input on the current realtime task."""

        ...

    def streaming_complete(self, complete_timeout_millis: int) -> None:
        """Finish text input and wait for the remaining audio."""

        ...

    def streaming_cancel(self, complete_timeout_millis: int) -> None:
        """Ask the server to drop the current task and its pending audio."""

        ...

    def close(self) -> None:
        """Close the realtime WebSocket connection."""

        ...

    def get_last_request_id(self) -> str:
        """Return the Model Studio request ID of the last task."""

        ...

    def get_first_package_delay(self) -> float:
        """Return the first-package delay of the last task in milliseconds."""

        ...


class _AliyunCosyVoiceRealtimeCallback(ResultCallback):
    """Bridge DashScope realtime callbacks onto a blocking event queue.

    The SDK delivers audio and terminal events from its WebSocket receiver
    thread, so the callbacks only enqueue plain data and never block it.
    ``bytes`` items are PCM deltas, ``None`` ends the stream successfully, and
    an ``Exception`` item reports a typed provider failure.
    """

    def __init__(self, events: Queue[bytes | Exception | None]) -> None:
        self._events: Queue[bytes | Exception | None] = events

        self._lock: Lock = Lock()

        self._settled: bool = False

    @property
    def settled(self) -> bool:
        with self._lock:
            return self._settled

    @override
    def on_data(self, data: bytes) -> None:
        if data == b"" or self.settled:
            return
        self._events.put(bytes(data))

    @override
    def on_complete(self) -> None:
        self.finish(None)

    @override
    def on_error(self, message: str) -> None:
        self.finish(_aliyun_task_failure(message))

    @override
    def on_close(self) -> None:
        self.finish(ProviderResponseError(stage="tts", reason="closed"))

    def finish(self, outcome: bytes | Exception | None) -> None:
        """Publish one terminal outcome unless the stream already settled."""
        with self._lock:
            if self._settled:
                return
            self._settled = True
        self._events.put(outcome)


class _DashScopeRealtimeSynthesizer(SpeechSynthesizer):
    """The official ``SpeechSynthesizer`` plus WebSocket-level termination.

    ``ResultCallback`` reports task completion and server-side task failures,
    but a socket that dies without a ``task-finished`` message only stops the
    SDK receiver thread.  Relaying the WebSocket handlers lets the adapter fail
    the stream immediately instead of waiting for its completion timeout.
    """

    def __init__(
        self,
        *,
        callback: _AliyunCosyVoiceRealtimeCallback,
        model: str,
        voice: str,
        audio_format: AudioFormat,
        endpoint: str,
    ) -> None:
        # The SDK types its constructor parameters only partially, so this call
        # is covered by the ``_RealtimeSpeechSynthesizer`` protocol instead.
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            model=model,
            voice=voice,
            format=audio_format,
            url=endpoint,
            callback=callback,
        )
        self._realtime_callback: _AliyunCosyVoiceRealtimeCallback = callback

    @override
    def on_error(self, ws: object, error: object) -> None:
        super().on_error(ws, error)  # pyright: ignore[reportUnknownMemberType]
        self._realtime_callback.finish(
            ProviderResponseError(stage="tts", reason="connect")
        )

    @override
    def on_close(
        self,
        ws: object,
        close_status_code: object,
        close_msg: object,
    ) -> None:
        super().on_close(  # pyright: ignore[reportUnknownMemberType]
            ws,
            close_status_code,
            close_msg,
        )
        self._realtime_callback.finish(
            ProviderResponseError(stage="tts", reason="closed")
        )


@dataclass(frozen=True, slots=True)
class AliyunCosyVoiceTTSAdapter:
    """Alibaba Cloud Model Studio realtime CosyVoice adapter.

    Synthesis runs through the official DashScope ``tts_v2`` SDK, which speaks
    the Model Studio realtime duplex WebSocket protocol and returns audio
    incrementally.  ``stream_pcm16le`` forwards those PCM deltas as they
    arrive, and ``synthesize`` buffers one realtime clip into a 16 kHz mono
    PCM16 WAV container so the buffered mode stays cancellable too.  Reference
    material belongs to Model Studio's separate voice-cloning API: a synthesis
    task only accepts the resulting system or cloned voice ID.
    """

    endpoint: str

    model: str

    api_key: str | None = None

    # The DashScope realtime SDK owns its WebSocket TLS configuration, so this
    # shared adapter field is accepted without being applied.
    ca_path: Path | None = None

    timeout_seconds: float = 120.0

    capability: ProviderCapability = "final_only"

    def __post_init__(self) -> None:
        _require_endpoint_and_model(self.endpoint, self.model)
        if self.api_key is None or self.api_key.strip() == "":
            raise MediaAdapterConfigError(field_name="api_key")
        if urlsplit(self.endpoint).scheme not in _ALIYUN_WEBSOCKET_SCHEMES:
            raise MediaAdapterConfigError(field_name="endpoint", reason="invalid")

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:
        pcm16le = b"".join(
            chunk.data
            for chunk in self.stream_pcm16le(
                text=text,
                voice=voice,
                ref_audio=ref_audio,
                ref_text=ref_text,
                cancellation=cancellation,
            )
        )
        if cancellation is not None and cancellation.cancelled:
            return SynthesizedAudio(data=b"", media_type="application/octet-stream")
        _LOGGER.debug(_ALIYUN_WAV_RESPONSE_LOG, binary_summary(pcm16le))
        return SynthesizedAudio(
            data=_pcm16le_wav(pcm16le),
            media_type="audio/wav",
        )

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[Pcm16leChunk]:
        # Model Studio's separate voice-cloning API owns reference material.
        _ = ref_audio, ref_text
        normalized_voice = voice.strip()
        if normalized_voice == "":
            raise MediaAdapterConfigError(field_name="voice")
        events: Queue[bytes | Exception | None] = Queue()
        callback = _AliyunCosyVoiceRealtimeCallback(events)
        synthesizer = self._open_realtime_synthesizer(
            voice=normalized_voice,
            callback=callback,
        )
        _LOGGER.debug(
            _ALIYUN_REQUEST_LOG,
            self.endpoint,
            self.model.strip(),
            normalized_voice,
            self.capability,
            text,
        )
        stopped = Event()
        worker = Thread(
            target=_run_aliyun_cosyvoice_task,
            args=(synthesizer, text, callback, stopped),
            name=_ALIYUN_SYNTHESIS_THREAD,
            daemon=True,
        )

        def stop_consumer() -> None:
            stopped.set()
            events.put(None)

        release = _bind_cancellation(cancellation, stop_consumer)
        worker.start()
        try:
            while True:
                event = events.get()
                if event is None:
                    if cancellation is not None and cancellation.cancelled:
                        return
                    _LOGGER.debug(
                        _ALIYUN_COMPLETE_LOG,
                        synthesizer.get_last_request_id(),
                        synthesizer.get_first_package_delay(),
                    )
                    break
                if isinstance(event, Exception):
                    if cancellation is not None and cancellation.cancelled:
                        return
                    raise event
                if cancellation is not None and cancellation.cancelled:
                    return
                _LOGGER.debug(_ALIYUN_PCM_RESPONSE_LOG, binary_summary(event))
                yield Pcm16leChunk(event)
        finally:
            release()
            _stop_aliyun_cosyvoice_task(synthesizer, callback, stopped, worker)

    def _open_realtime_synthesizer(
        self,
        *,
        voice: str,
        callback: _AliyunCosyVoiceRealtimeCallback,
    ) -> _RealtimeSpeechSynthesizer:
        api_key = self.api_key
        if api_key is None or api_key.strip() == "":
            raise MediaAdapterConfigError(field_name="api_key")
        return _build_aliyun_cosyvoice_synthesizer(
            endpoint=self.endpoint,
            api_key=api_key.strip(),
            model=self.model.strip(),
            voice=voice,
            callback=callback,
        )


@dataclass(frozen=True, slots=True)
class AudioCppTTSAdapter:
    """audio.cpp OpenAI-compatible speech adapter.

    Non-streaming models return a complete WAV. Models configured by audio.cpp
    with ``mode: streaming`` may instead expose OpenAI-shaped PCM SSE events.
    """

    endpoint: str

    model: str

    api_key: str | None = None

    ca_path: Path | None = None

    timeout_seconds: float = 120.0

    capability: ProviderCapability = "final_only"

    def __post_init__(self) -> None:
        _require_endpoint_and_model(self.endpoint, self.model)

    def build_speech_request(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        streaming: bool = False,
    ) -> AudioCppSpeechRequest:
        payload: AudioCppSpeechPayload = {
            "model": self.model.strip(),
            "input": text,
            "response_format": "pcm" if streaming else "wav",
        }
        if streaming:
            payload["stream_format"] = "sse"
        if voice.strip() != "":
            payload["voice"] = voice.strip()
        if ref_audio.strip() != "":
            payload["voice_ref"] = _portable_reference_audio(ref_audio)
        if ref_text.strip() != "":
            payload["reference_text"] = ref_text
        return AudioCppSpeechRequest(
            method="POST",
            url=f"{self.endpoint.rstrip('/')}/audio/speech",
            json=payload,
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
        _log_audio_cpp_tts_request(speech)
        client = self._client()
        release = _bind_cancellation(cancellation, client.close)
        try:
            response = client.post(
                speech.url,
                json=speech.json,
                headers=self._headers(accept="audio/wav"),
            )
            _ = response.raise_for_status()
            if cancellation is not None and cancellation.cancelled:
                return SynthesizedAudio(data=b"", media_type="application/octet-stream")
            data = response.content
            _LOGGER.debug(
                "tts_response provider=audio_cpp transport=http media_type=%s %s",
                "audio/wav",
                binary_summary(data),
            )
            return SynthesizedAudio(data=data, media_type="audio/wav")
        except httpx.HTTPError as error:
            if cancellation is not None and cancellation.cancelled:
                return SynthesizedAudio(data=b"", media_type="application/octet-stream")
            raise _tts_provider_error(error) from error
        finally:
            release()
            client.close()

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[Pcm16leChunk]:
        speech = self.build_speech_request(
            text=text,
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
            streaming=True,
        )
        _log_audio_cpp_tts_request(speech)
        client = self._client()
        release = _bind_cancellation(cancellation, client.close)
        try:
            with client.stream(
                "POST",
                speech.url,
                json=speech.json,
                headers=self._headers(accept="text/event-stream"),
            ) as response:
                _ = response.raise_for_status()
                done = False
                resampler = _Pcm24khzTo16khzResampler()
                for line in response.iter_lines():
                    if cancellation is not None and cancellation.cancelled:
                        return
                    data = _sse_data(line)
                    if data is None:
                        continue
                    chunk = _normalize_tts_sse(
                        data,
                        allow_missing_response_format=True,
                    )
                    if chunk is None:
                        done = True
                        break
                    converted = resampler.push(chunk)
                    if converted:
                        yield Pcm16leChunk(converted)
                if (cancellation is None or not cancellation.cancelled) and not done:
                    raise ProviderResponseError(stage="tts", reason="missing_done")
        except httpx.HTTPError as error:
            if cancellation is not None and cancellation.cancelled:
                return
            raise _tts_provider_error(error) from error
        finally:
            release()
            client.close()

    def _client(self) -> httpx.Client:
        verify: bool | ssl.SSLContext = (
            True
            if self.ca_path is None
            else ssl.create_default_context(cafile=self.ca_path)
        )
        return httpx.Client(
            verify=verify,
            timeout=self.timeout_seconds,
            trust_env=False,
        )

    def _headers(self, *, accept: str) -> dict[str, str]:
        headers = {"Accept": accept}
        if self.api_key is not None and self.api_key.strip() != "":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


@dataclass(frozen=True, slots=True)
class Qwen3TtsCppTTSAdapter:
    """qwentts.cpp OpenAI-compatible speech adapter.

    The engine owns a registry of named speakers, so the dialect selects the
    speaker with ``voice`` and needs no cloning reference.  ``response_format``
    alone decides the transfer: ``wav`` buffers one complete sentence, while
    ``pcm`` answers with chunked raw s16le 24 kHz mono audio.  That chunked
    body carries no event framing or terminator; the end of the body is the
    end of speech.
    """

    endpoint: str

    model: str

    api_key: str | None = None

    ca_path: Path | None = None

    timeout_seconds: float = 120.0

    capability: ProviderCapability = "final_only"

    def __post_init__(self) -> None:
        _require_endpoint_and_model(self.endpoint, self.model)

    def build_speech_request(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        streaming: bool = False,
    ) -> Qwen3TtsCppSpeechRequest:
        # Speaker selection belongs to the engine registry in this dialect, so
        # a locally configured cloning pair stays unused on purpose.
        _ = ref_audio, ref_text
        payload: Qwen3TtsCppSpeechPayload = {
            "model": self.model.strip(),
            "input": text,
            "response_format": "pcm" if streaming else "wav",
        }
        if voice.strip() != "":
            payload["voice"] = voice.strip()
        return Qwen3TtsCppSpeechRequest(
            method="POST",
            url=f"{self.endpoint.rstrip('/')}/audio/speech",
            json=payload,
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
        _log_qwen3ttscpp_tts_request(speech)
        client = self._client()
        release = _bind_cancellation(cancellation, client.close)
        try:
            response = client.post(
                speech.url,
                json=speech.json,
                headers=self._headers(accept="audio/wav"),
            )
            _ = response.raise_for_status()
            if cancellation is not None and cancellation.cancelled:
                return SynthesizedAudio(data=b"", media_type="application/octet-stream")
            data = response.content
            _LOGGER.debug(
                "tts_response provider=qwen3ttscpp transport=http media_type=%s %s",
                "audio/wav",
                binary_summary(data),
            )
            return SynthesizedAudio(data=data, media_type="audio/wav")
        except httpx.HTTPError as error:
            if cancellation is not None and cancellation.cancelled:
                return SynthesizedAudio(data=b"", media_type="application/octet-stream")
            raise _tts_provider_error(error) from error
        finally:
            release()
            client.close()

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[Pcm16leChunk]:
        """Consume raw chunked PCM while the engine is still synthesizing it."""
        speech = self.build_speech_request(
            text=text,
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
            streaming=True,
        )
        _log_qwen3ttscpp_tts_request(speech)
        client = self._client()
        release = _bind_cancellation(cancellation, client.close)
        try:
            with client.stream(
                "POST",
                speech.url,
                json=speech.json,
                headers=self._headers(accept="audio/pcm"),
            ) as response:
                _ = response.raise_for_status()
                resampler = _Pcm24khzTo16khzResampler()
                emitted = 0
                for pcm in _raw_pcm16le_chunks(response.iter_bytes(), cancellation):
                    converted = resampler.push(pcm)
                    if converted:
                        emitted += len(converted)
                        yield Pcm16leChunk(converted)
                if cancellation is None or not cancellation.cancelled:
                    _LOGGER.debug(_QWEN3TTSCPP_STREAM_RESPONSE_LOG, emitted)
        except httpx.HTTPError as error:
            if cancellation is not None and cancellation.cancelled:
                return
            raise _tts_provider_error(error) from error
        finally:
            release()
            client.close()

    def _client(self) -> httpx.Client:
        verify: bool | ssl.SSLContext = (
            True
            if self.ca_path is None
            else ssl.create_default_context(cafile=self.ca_path)
        )
        return httpx.Client(
            verify=verify,
            timeout=self.timeout_seconds,
            trust_env=False,
        )

    def _headers(self, *, accept: str) -> dict[str, str]:
        headers = {"Accept": accept}
        if self.api_key is not None and self.api_key.strip() != "":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _require_endpoint_and_model(endpoint: str, model: str) -> None:
    if endpoint.strip() == "":
        raise MediaAdapterConfigError(field_name="endpoint")

    if model.strip() == "":
        raise MediaAdapterConfigError(field_name="model")


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


def _tts_provider_error(error: APIError | httpx.HTTPError) -> ProviderResponseError:
    if isinstance(error, APIStatusError):
        return ProviderResponseError(stage="tts", reason=f"status_{error.status_code}")
    if isinstance(error, httpx.HTTPStatusError):
        return ProviderResponseError(
            stage="tts", reason=f"status_{error.response.status_code}"
        )
    if isinstance(error, (APITimeoutError, httpx.TimeoutException)):
        return ProviderResponseError(stage="tts", reason="read")
    if isinstance(error, APIConnectionError):
        return ProviderResponseError(stage="tts", reason="connect")
    return ProviderResponseError(stage="tts", reason="response")


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


def _build_aliyun_cosyvoice_synthesizer(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    voice: str,
    callback: _AliyunCosyVoiceRealtimeCallback,
) -> _RealtimeSpeechSynthesizer:
    """Open one Model Studio realtime task through the official DashScope SDK.

    The SDK reads its credential from the ``dashscope.api_key`` module global
    when a task starts, so the deployment key has to be installed here.  It is
    never logged, echoed into a request payload, or stored on the adapter.
    """
    dashscope.api_key = api_key
    return _DashScopeRealtimeSynthesizer(
        callback=callback,
        model=model,
        voice=voice,
        audio_format=_ALIYUN_REALTIME_AUDIO_FORMAT,
        endpoint=endpoint,
    )


def _run_aliyun_cosyvoice_task(
    synthesizer: _RealtimeSpeechSynthesizer,
    text: str,
    callback: _AliyunCosyVoiceRealtimeCallback,
    stopped: Event,
) -> None:
    """Drive one realtime task while the SDK streams PCM through the callback.

    This thread owns task start-up and completion only.  Its completion wait is
    bounded so an abandoned task cannot hold a thread forever, and every exit
    publishes exactly one terminal event to the consumer queue.
    """
    try:
        if stopped.is_set():
            return
        synthesizer.streaming_call(text)
        _ = synthesizer.streaming_complete(
            complete_timeout_millis=_ALIYUN_COMPLETE_TIMEOUT_MILLIS
        )
    except Exception as error:  # noqa: BLE001 - relayed as a typed provider error
        _LOGGER.debug(_ALIYUN_ERROR_LOG, "task", _aliyun_detail(error))
        callback.finish(_aliyun_sdk_error(error))
    finally:
        callback.finish(None)


def _stop_aliyun_cosyvoice_task(
    synthesizer: _RealtimeSpeechSynthesizer,
    callback: _AliyunCosyVoiceRealtimeCallback,
    stopped: Event,
    worker: Thread,
) -> None:
    """Release the realtime task the consumer stopped reading."""
    stopped.set()
    if not callback.settled:
        try:
            _ = synthesizer.streaming_cancel(
                complete_timeout_millis=_ALIYUN_CANCEL_TIMEOUT_MILLIS
            )
        except Exception as error:  # noqa: BLE001 - keep the stream outcome
            _LOGGER.debug(_ALIYUN_ERROR_LOG, "cancel", _aliyun_detail(error))
        finally:
            callback.finish(None)
    synthesizer.close()
    worker.join(timeout=_ALIYUN_WORKER_JOIN_SECONDS)
    if worker.is_alive():
        _LOGGER.debug(_ALIYUN_STALLED_LOG, _ALIYUN_SYNTHESIS_THREAD)


def _aliyun_sdk_error(error: BaseException) -> ProviderResponseError:
    if isinstance(error, TimeoutError):
        return ProviderResponseError(stage="tts", reason="read")
    if isinstance(error, ConnectionError):
        return ProviderResponseError(stage="tts", reason="connect")
    return ProviderResponseError(stage="tts", reason="response")


def _aliyun_task_failure(message: str) -> ProviderResponseError:
    _LOGGER.error(_ALIYUN_ERROR_LOG, "server", _aliyun_detail(message))
    return ProviderResponseError(stage="tts", reason="server")


def _aliyun_detail(value: object) -> str:
    return " ".join(str(value).split())[:_ALIYUN_ERROR_CHARS]


def _pcm16le_wav(pcm16le: bytes) -> bytes:
    """Wrap realtime PCM in the WAV container of the buffered media path."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(_ALIYUN_CHANNELS)
        audio.setsampwidth(_ALIYUN_SAMPLE_BYTES)
        audio.setframerate(_ALIYUN_SAMPLE_RATE)
        audio.writeframes(pcm16le)
    return buffer.getvalue()


def _log_tts_request(speech: HttpSpeechRequest) -> None:
    """Log request text while keeping reference audio out of the record."""
    _LOGGER.debug(
        "tts_request url=%s model=%s mode=voice_clone input=%r ref_audio=(%s) ref_text=%r",  # noqa: E501
        speech.url,
        speech.json["model"],
        speech.json["input"],
        reference_audio_summary(speech.extra_body["ref_audio"]),
        speech.extra_body["ref_text"],
    )


def _log_audio_cpp_tts_request(speech: AudioCppSpeechRequest) -> None:
    """Log text and request shape without recording reference audio bytes."""
    voice_ref = speech.json.get("voice_ref")
    message = "tts_request provider=audio_cpp url=%s model=%s mode=%s input=%r"
    message += " voice=%r voice_ref=(%s) reference_text=%r"
    _LOGGER.debug(
        message,
        speech.url,
        speech.json["model"],
        "streaming_sse" if speech.json.get("stream_format") == "sse" else "final_only",
        speech.json["input"],
        speech.json.get("voice"),
        "default" if voice_ref is None else reference_audio_summary(voice_ref),
        speech.json.get("reference_text"),
    )


def _log_qwen3ttscpp_tts_request(speech: Qwen3TtsCppSpeechRequest) -> None:
    """Log the request shape while keeping credentials out of the record."""
    _LOGGER.debug(
        "tts_request provider=qwen3ttscpp url=%s model=%s mode=%s voice=%r input=%r",
        speech.url,
        speech.json["model"],
        "streaming_pcm" if speech.json["response_format"] == "pcm" else "final_only",
        speech.json.get("voice"),
        speech.json["input"],
    )


def _normalize_tts_sse(
    data: str,
    *,
    allow_missing_response_format: bool = False,
) -> bytes | None:
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
            response_format = payload.get("response_format")
            format_is_valid = response_format == "pcm" or (
                allow_missing_response_format and response_format is None
            )
            if not format_is_valid or not isinstance(encoded, str):
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


def _sse_data(line: str) -> str | None:
    if line == "" or not line.startswith("data: "):
        return None
    return line.removeprefix("data: ")


def _raw_pcm16le_chunks(
    chunks: Iterable[bytes],
    cancellation: ProviderCancellationHandle | None,
) -> Iterator[bytes]:
    """Yield whole PCM16 samples from a body that carries no event framing.

    A chunked HTTP body is an arbitrary byte partition, so a chunk may end in
    the middle of one little-endian sample.  Hold that byte until its neighbour
    arrives, and fail closed when the body ends mid-sample instead of letting a
    truncated sample reach the RTP packetizer.
    """
    pending = b""
    for chunk in chunks:
        if cancellation is not None and cancellation.cancelled:
            return
        pending += chunk
        usable = len(pending) - len(pending) % _PCM16_SAMPLE_BYTES
        if usable == 0:
            continue
        ready = pending[:usable]
        pending = pending[usable:]
        yield ready
    if pending != b"":
        raise ProviderResponseError(stage="tts", reason="incomplete_pcm")


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
