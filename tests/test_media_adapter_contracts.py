from __future__ import annotations

import base64
import io
import json
import logging
import ssl
import wave
from queue import Queue
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self, final

import dashscope
import pytest

from orchestrator import media_adapters
from orchestrator.config import load_config_from_env
from orchestrator.llm import (
    AliyunCosyVoiceTTSAdapter,
    AudioCppTTSAdapter,
    OpenAICompatibleASRAdapter,
    Qwen3TtsCppTTSAdapter,
    VllmOmniTTSAdapter,
)
from orchestrator.media_adapters import MediaAdapterConfigError, SynthesizedAudio
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import (
    ProviderCancellationHandle,
    ProviderCapability,
    ProviderResponseError,
)
from orchestrator.tts_rtp import Pcm16leChunk

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def ca_path(tmp_path: Path) -> Path:
    certificate = ssl.create_default_context().get_ca_certs(binary_form=True)[0]
    path = tmp_path / "ca.pem"
    _ = path.write_text(ssl.DER_cert_to_PEM_cert(certificate), encoding="ascii")
    return path


_FAKE_REALTIME_WAIT_SECONDS = 5.0


@final
class _FakeRealtimeSynthesizer:
    """Scripted stand-in for the DashScope realtime ``SpeechSynthesizer``.

    The class attributes hold the script so a test can reshape the fake before
    the adapter's generator opens it; instances record what the adapter did.
    """

    pcm: tuple[bytes, ...] = ()

    failure: str | None = None

    completes: bool = True

    def __init__(
        self,
        callback: media_adapters._AliyunCosyVoiceRealtimeCallback,  # pyright: ignore[reportPrivateUsage]
    ) -> None:
        self.callback = callback

        self.submitted: list[str] = []

        self.completion_timeouts: list[int] = []

        self.cancellation_timeouts: list[int] = []

        self.running = Event()

        self.closed = Event()

    def streaming_call(self, text: str) -> None:
        self.submitted.append(text)
        for chunk in self.pcm:
            self.callback.on_data(chunk)
        if self.failure is not None:
            self.callback.on_error(self.failure)
        elif self.completes:
            self.callback.on_complete()

    def streaming_complete(self, complete_timeout_millis: int) -> None:
        self.completion_timeouts.append(complete_timeout_millis)
        if not self.completes:
            _ = self.running.wait(timeout=_FAKE_REALTIME_WAIT_SECONDS)

    def streaming_cancel(self, complete_timeout_millis: int) -> None:
        self.cancellation_timeouts.append(complete_timeout_millis)
        self.running.set()

    def close(self) -> None:
        self.closed.set()
        self.running.set()

    def get_last_request_id(self) -> str:
        return "request-test"

    def get_first_package_delay(self) -> float:
        return 12.5


def _install_realtime_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, str]], list[_FakeRealtimeSynthesizer]]:
    """Replace the DashScope realtime factory with a scripted fake."""
    opened: list[dict[str, str]] = []

    synthesizers: list[_FakeRealtimeSynthesizer] = []

    def build(
        *,
        endpoint: str,
        api_key: str,
        model: str,
        voice: str,
        callback: media_adapters._AliyunCosyVoiceRealtimeCallback,  # pyright: ignore[reportPrivateUsage]
    ) -> _FakeRealtimeSynthesizer:
        synthesizer = _FakeRealtimeSynthesizer(callback)
        opened.append(
            {
                "endpoint": endpoint,
                "api_key": api_key,
                "model": model,
                "voice": voice,
            }
        )
        synthesizers.append(synthesizer)
        return synthesizer

    monkeypatch.setattr(media_adapters, "_build_aliyun_cosyvoice_synthesizer", build)
    return opened, synthesizers


_REALTIME_ENDPOINT = (
    "wss://workspace.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
)


def _realtime_adapter(
    *, capability: ProviderCapability = "streaming"
) -> AliyunCosyVoiceTTSAdapter:
    return AliyunCosyVoiceTTSAdapter(
        endpoint=_REALTIME_ENDPOINT,
        model="cosyvoice-v3-flash",
        api_key="test-api-key",
        capability=capability,
    )


def test_default_mock_media_providers_need_no_credentials_or_network() -> None:
    # Given: the normal replay environment has no provider configuration.

    config = load_config_from_env({})

    # When: Orchestrator resolves the media provider defaults.

    # Then: both media paths are mock-owned and require no secret or endpoint.

    assert config.asr_provider == "mock"

    assert config.tts_provider == "mock"

    assert config.asr_api_key is None

    assert config.tts_api_key is None

    assert config.asr_endpoint is None

    assert config.tts_endpoint is None


def test_openai_compatible_asr_normalizes_final_at_orchestrator_boundary() -> None:
    # Given: a provider-shaped final transcription from a configured local endpoint.

    adapter = OpenAICompatibleASRAdapter(
        endpoint="http://127.0.0.1:8000/v1",
        model="local-asr",
    )

    # When: Orchestrator converts the OpenAI-compatible result into its input contract.

    result = adapter.normalize_final(
        response={"text": "  explain BitNet  "},
        received_at_ms=1_250,
        segment_id="asr-local-0001",
        seq=4,
    )

    # Then: downstream code receives a normalized ASR event, not provider JSON.

    assert result == ASRAudienceEvent(
        text="explain BitNet",
        received_at_ms=1_250,
        segment_id="asr-local-0001",
        seq=4,
    )


def test_openai_compatible_asr_treats_blank_final_as_no_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a configured provider returns its normal empty final for a silent endpoint.

    adapter = OpenAICompatibleASRAdapter(
        endpoint="http://127.0.0.1:8000/v1",
        model="local-asr",
    )

    def create_blank_transcription(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(text="   ")

    client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=create_blank_transcription)
        ),
        close=lambda: None,
    )

    def build_client(_adapter: OpenAICompatibleASRAdapter) -> SimpleNamespace:
        return client

    monkeypatch.setattr(
        OpenAICompatibleASRAdapter,
        "_client",
        build_client,
    )

    # When: Orchestrator transcribes the endpointed audio.

    result = adapter.transcribe(
        audio=b"wav",
        filename="onsite-l16.wav",
        received_at_ms=20,
        segment_id="asr-local-0001",
        seq=1,
    )

    # Then: silence is discarded rather than treated as a provider misconfiguration.

    assert result is None


def test_vllm_omni_builds_opt_in_fake_local_speech_request() -> None:
    # Given: an explicitly configured fake-local vLLM-Omni surface and clone reference.

    adapter = VllmOmniTTSAdapter(
        endpoint="http://127.0.0.1:8001/v1",
        model="vllm-omni",
    )

    # When: Orchestrator builds a request without making a network call.

    request = adapter.build_speech_request(
        text="欢迎来到 BitNet 讲解。",
        voice="raspberry",
        ref_audio="https://media.example.test/raspberry.wav",
        ref_text="参考音色文本",
    )

    # Then: only documented cloning fields are present on the audio-speech request.

    assert request.method == "POST"

    assert request.url == "http://127.0.0.1:8001/v1/audio/speech"

    assert request.json == {"model": "vllm-omni", "input": "欢迎来到 BitNet 讲解。"}
    assert request.extra_body == {
        "task_type": "Base",
        "ref_audio": "https://media.example.test/raspberry.wav",
        "ref_text": "参考音色文本",
    }


def test_aliyun_cosyvoice_requires_realtime_websocket_endpoint() -> None:
    # Given: the non-realtime HTTP SpeechSynthesizer resource of the old client.

    # When / Then: it is rejected before any realtime task can be opened.

    with pytest.raises(MediaAdapterConfigError, match="endpoint"):
        _ = AliyunCosyVoiceTTSAdapter(
            endpoint=(
                "https://workspace.cn-beijing.maas.aliyuncs.com"
                "/api/v1/services/audio/tts/SpeechSynthesizer"
            ),
            model="cosyvoice-v3-flash",
            api_key="test-api-key",
        )

    # Then: the realtime WebSocket inference endpoint is accepted.

    assert _realtime_adapter().capability == "streaming"


def test_aliyun_cosyvoice_requires_an_api_key() -> None:
    with pytest.raises(MediaAdapterConfigError, match="api_key"):
        _ = AliyunCosyVoiceTTSAdapter(
            endpoint=_REALTIME_ENDPOINT,
            model="cosyvoice-v3-flash",
        )


def test_aliyun_cosyvoice_builds_official_dashscope_realtime_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the deployment credential the official SDK reads from its globals.

    monkeypatch.setattr(dashscope, "api_key", None)
    events: Queue[bytes | Exception | None] = Queue()
    callback = media_adapters._AliyunCosyVoiceRealtimeCallback(  # pyright: ignore[reportPrivateUsage]
        events
    )

    # When: the adapter opens a CosyVoice realtime task.

    synthesizer = media_adapters._build_aliyun_cosyvoice_synthesizer(  # pyright: ignore[reportPrivateUsage]
        endpoint=_REALTIME_ENDPOINT,
        api_key="test-api-key",
        model="cosyvoice-v3-flash",
        voice="longanyang",
        callback=callback,
    )

    # Then: it drives the official SpeechSynthesizer with 16 kHz PCM output.

    assert dashscope.api_key == "test-api-key"
    assert isinstance(
        synthesizer,
        media_adapters._DashScopeRealtimeSynthesizer,  # pyright: ignore[reportPrivateUsage]
    )
    assert synthesizer.aformat == "pcm"
    assert synthesizer.sample_rate == 16_000


def test_aliyun_cosyvoice_streams_realtime_pcm_through_dashscope_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcm = b"\x10\x20" * 320
    monkeypatch.setattr(_FakeRealtimeSynthesizer, "pcm", (pcm[:320], pcm[320:]))
    opened, synthesizers = _install_realtime_synthesizer(monkeypatch)
    adapter = _realtime_adapter()

    chunks = tuple(
        adapter.stream_pcm16le(
            text="你好。我是树莓娘。",
            voice="longanyang",
            ref_audio="data:audio/wav;base64,not-sent",
            ref_text="不随合成请求发送",
        )
    )

    assert [chunk.data for chunk in chunks] == [pcm[:320], pcm[320:]]
    assert [chunk.sample_rate for chunk in chunks] == [16_000, 16_000]
    assert opened == [
        {
            "endpoint": _REALTIME_ENDPOINT,
            "api_key": "test-api-key",
            "model": "cosyvoice-v3-flash",
            "voice": "longanyang",
        }
    ]
    assert synthesizers[0].submitted == ["你好。我是树莓娘。"]
    assert synthesizers[0].completion_timeouts == [60_000]
    assert synthesizers[0].closed.is_set()


def test_aliyun_cosyvoice_reports_realtime_task_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure_message = json.dumps(
        {
            "header": {
                "event": "task-failed",
                "error_code": "InvalidParameter",
                "error_message": "[cosyvoice:]Engine return error code: 418",
            }
        }
    )
    monkeypatch.setattr(_FakeRealtimeSynthesizer, "failure", failure_message)
    _opened, synthesizers = _install_realtime_synthesizer(monkeypatch)

    with (
        caplog.at_level(logging.ERROR, logger="orchestrator.media_adapters"),
        pytest.raises(ProviderResponseError, match="server"),
    ):
        _ = tuple(
            _realtime_adapter().stream_pcm16le(
                text="测试。",
                voice="cosyvoice-v3-flash-raspberry-clone",
                ref_audio="should-not-be-sent",
                ref_text="should-not-be-sent",
            )
        )

    assert "InvalidParameter" in caplog.text
    assert "Engine return error code: 418" in caplog.text
    assert "test-api-key" not in caplog.text
    assert synthesizers[0].closed.is_set()


def test_aliyun_cosyvoice_cancels_realtime_task_when_the_stream_is_abandoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a realtime task that is still producing audio when input supersedes it.

    pcm = b"\x10\x20" * 320
    monkeypatch.setattr(_FakeRealtimeSynthesizer, "pcm", (pcm,))
    monkeypatch.setattr(_FakeRealtimeSynthesizer, "completes", False)
    _opened, synthesizers = _install_realtime_synthesizer(monkeypatch)
    cancellation = ProviderCancellationHandle()
    stream = _realtime_adapter().stream_pcm16le(
        text="你好。",
        voice="longanyang",
        ref_audio="",
        ref_text="",
        cancellation=cancellation,
    )

    # When: the first chunk is delivered and the turn is cancelled.

    assert next(stream) == Pcm16leChunk(pcm)
    _ = cancellation.cancel(reason="superseded")

    # Then: the adapter stops the realtime task and releases its socket.

    with pytest.raises(StopIteration):
        _ = next(stream)

    assert synthesizers[0].cancellation_timeouts == [2_000]
    assert synthesizers[0].closed.is_set()


def test_aliyun_cosyvoice_buffers_realtime_pcm_into_wav_for_final_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcm = b"\x01\x02" * 160
    monkeypatch.setattr(_FakeRealtimeSynthesizer, "pcm", (pcm,))
    _opened, synthesizers = _install_realtime_synthesizer(monkeypatch)

    result = _realtime_adapter(capability="final_only").synthesize(
        text="你好",
        voice="longanyang",
        ref_audio="",
        ref_text="",
    )

    assert result.media_type == "audio/wav"
    with wave.open(io.BytesIO(result.data), "rb") as audio:
        assert (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
        ) == (1, 2, 16_000)
        assert audio.readframes(audio.getnframes()) == pcm
    assert synthesizers[0].completion_timeouts == [60_000]


def test_audio_cpp_builds_non_streaming_request_using_model_default_voice() -> None:
    adapter = AudioCppTTSAdapter(
        endpoint="http://127.0.0.1:8080/v1",
        model="pocket-tts",
    )

    request = adapter.build_speech_request(
        text="你好",
        voice="",
        ref_audio="",
        ref_text="",
    )

    assert request.url == "http://127.0.0.1:8080/v1/audio/speech"
    assert request.json == {
        "model": "pocket-tts",
        "input": "你好",
        "response_format": "wav",
    }


def test_audio_cpp_maps_request_voice_overrides_to_documented_fields() -> None:
    request = AudioCppTTSAdapter(
        endpoint="http://127.0.0.1:8080/v1",
        model="voxcpm2",
    ).build_speech_request(
        text="讲解",
        voice="preset-a",
        ref_audio="https://media.example.test/reference.wav",
        ref_text="参考文本",
    )

    assert request.json == {
        "model": "voxcpm2",
        "input": "讲解",
        "response_format": "wav",
        "voice": "preset-a",
        "voice_ref": "https://media.example.test/reference.wav",
        "reference_text": "参考文本",
    }


def test_audio_cpp_final_only_requests_wav_without_streaming_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AudioCppTTSAdapter(
        endpoint="http://127.0.0.1:8080/v1",
        model="pocket-tts",
    )
    calls: list[dict[str, object]] = []

    @final
    class Response:
        content: bytes = b"RIFFaudio-cpp-wav"

        def raise_for_status(self) -> None:
            return

    @final
    class Client:
        def post(self, _url: str, **kwargs: object) -> Response:
            calls.append(kwargs)
            return Response()

        def close(self) -> None:
            return

    def build_client(_adapter: AudioCppTTSAdapter) -> Client:
        return Client()

    monkeypatch.setattr(AudioCppTTSAdapter, "_client", build_client)

    result = adapter.synthesize(
        text="你好",
        voice="",
        ref_audio="",
        ref_text="",
    )

    assert result.data == b"RIFFaudio-cpp-wav"
    assert calls == [
        {
            "json": {
                "model": "pocket-tts",
                "input": "你好",
                "response_format": "wav",
            },
            "headers": {"Accept": "audio/wav"},
        }
    ]


def test_audio_cpp_streaming_uses_sse_shape_and_resamples_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AudioCppTTSAdapter(
        endpoint="http://127.0.0.1:8080/v1",
        model="voxcpm2-stream",
        capability="streaming",
    )
    calls: list[dict[str, object]] = []
    encoded = base64.b64encode(b"\x10\x20" * 480).decode()

    @final
    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return

        def iter_lines(self) -> list[str]:
            return [
                "data: "
                + json.dumps(
                    {
                        "type": "speech.audio.delta",
                        "audio": encoded,
                    }
                ),
                'data: {"type":"speech.audio.done"}',
                "data: [DONE]",
            ]

    @final
    class Client:
        def stream(self, method: str, url: str, **kwargs: object) -> Response:
            calls.append({"method": method, "url": url, **kwargs})
            return Response()

        def close(self) -> None:
            return

    def build_client(_adapter: AudioCppTTSAdapter) -> Client:
        return Client()

    monkeypatch.setattr(AudioCppTTSAdapter, "_client", build_client)

    chunks = tuple(
        adapter.stream_pcm16le(
            text="讲解",
            voice="",
            ref_audio="",
            ref_text="",
        )
    )

    assert len(chunks) == 1
    assert len(chunks[0].data) == 640
    assert calls[0]["json"] == {
        "model": "voxcpm2-stream",
        "input": "讲解",
        "response_format": "pcm",
        "stream_format": "sse",
    }
    assert calls[0]["headers"] == {"Accept": "text/event-stream"}


def test_vllm_omni_encodes_local_reference_path_as_data_url(tmp_path: Path) -> None:
    # Given: the local Qwen server cannot read Orchestrator-local reference files.

    reference = tmp_path / "raspberry.wav"
    _ = reference.write_bytes(b"RIFFreference-wav")

    # When: Orchestrator builds the provider request from an absolute path.

    request = VllmOmniTTSAdapter(
        endpoint="http://127.0.0.1:8001/v1",
        model="vllm-omni",
    ).build_speech_request(
        text="欢迎来到 BitNet 讲解。",
        voice="raspberry",
        ref_audio=str(reference),
        ref_text="参考音色文本",
    )

    # Then: the provider receives portable audio bytes, not a host-local path.

    assert request.extra_body["ref_audio"] == _data_url(reference.read_bytes())


def test_vllm_omni_encodes_file_uri_reference_as_data_url(tmp_path: Path) -> None:
    # Given: a deployment config uses a file URI for the local reference WAV.

    reference = tmp_path / "raspberry.wav"
    _ = reference.write_bytes(b"RIFFreference-uri-wav")

    # When: Orchestrator builds the provider request.

    request = VllmOmniTTSAdapter(
        endpoint="http://127.0.0.1:8001/v1",
        model="vllm-omni",
    ).build_speech_request(
        text="欢迎来到 BitNet 讲解。",
        voice="raspberry",
        ref_audio=reference.as_uri(),
        ref_text="参考音色文本",
    )

    # Then: the provider receives a data URL accepted by the local Qwen server.

    assert request.extra_body["ref_audio"] == _data_url(reference.read_bytes())


def test_vllm_omni_preserves_existing_reference_data_url() -> None:
    # Given: the reference audio is already provider-portable.

    ref_audio = _data_url(b"RIFFalready-portable")

    # When: Orchestrator builds the provider request.

    request = VllmOmniTTSAdapter(
        endpoint="http://127.0.0.1:8001/v1",
        model="vllm-omni",
    ).build_speech_request(
        text="欢迎来到 BitNet 讲解。",
        voice="raspberry",
        ref_audio=ref_audio,
        ref_text="参考音色文本",
    )

    # Then: no second encoding corrupts the existing data URL.

    assert request.extra_body["ref_audio"] == ref_audio


def test_tts_logs_only_summaries_for_reference_audio_and_response(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    reference_payload = "private-reference-audio" * 100
    response_payload = b"private-api-response" * 100
    adapter = VllmOmniTTSAdapter(
        endpoint="http://127.0.0.1:8001/v1",
        model="vllm-omni",
    )

    def create_speech(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(content=response_payload)

    client = SimpleNamespace(
        audio=SimpleNamespace(speech=SimpleNamespace(create=create_speech)),
        close=lambda: None,
    )

    def build_client(_adapter: VllmOmniTTSAdapter) -> SimpleNamespace:
        return client

    monkeypatch.setattr(VllmOmniTTSAdapter, "_client", build_client)

    with caplog.at_level(logging.DEBUG, logger="orchestrator.media_adapters"):
        result = adapter.synthesize(
            text="讲解" * 100,
            voice="raspberry",
            ref_audio=f"data:audio/wav;base64,{reference_payload}",
            ref_text="参考文本" * 100,
        )

    log_output = caplog.text
    assert result.data == response_payload
    assert "tts_request" in log_output
    assert "tts_response" in log_output
    assert "payload_chars=2300" in log_output
    assert "bytes=2000" in log_output
    assert reference_payload not in log_output
    assert response_payload.decode() not in log_output


def test_tts_sse_uses_unbuffered_openai_streaming_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an SSE response wrapper whose events are exposed only through the
    # OpenAI client's unbuffered streaming surface.

    adapter = VllmOmniTTSAdapter(
        endpoint="http://127.0.0.1:8001/v1",
        model="vllm-omni",
        capability="streaming",
    )
    calls: list[dict[str, object]] = []
    pcm_24khz = base64.b64encode(b"\x10\x20" * 480).decode()

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def close(self) -> None:
            return None

        def iter_lines(self) -> list[str]:
            delta = {
                "type": "speech.audio.delta",
                "response_format": "pcm",
                "audio": pcm_24khz,
            }
            return [
                f"data: {json.dumps(delta)}",
                'data: {"type":"speech.audio.done"}',
            ]

    def create_speech(**kwargs: object) -> Response:
        calls.append(kwargs)
        return Response()

    client = SimpleNamespace(
        audio=SimpleNamespace(
            speech=SimpleNamespace(
                with_streaming_response=SimpleNamespace(create=create_speech)
            )
        ),
        close=lambda: None,
    )

    def build_client(_adapter: VllmOmniTTSAdapter) -> SimpleNamespace:
        return client

    monkeypatch.setattr(VllmOmniTTSAdapter, "_client", build_client)

    # When: the adapter consumes provider SSE.

    chunks = tuple(
        adapter.stream_pcm16le(
            text="讲解",
            voice="raspberry",
            ref_audio="file:///voice.wav",
            ref_text="参考文本",
        )
    )

    # Then: it selected the non-buffering OpenAI surface and emitted PCM.

    assert len(chunks) == 1
    assert len(chunks[0].data) == 640
    assert calls[0]["stream_format"] == "sse"


def test_qwen3ttscpp_builds_registered_voice_requests_for_both_transfers() -> None:
    # Given: the qwentts.cpp engine, whose speaker registry owns voice choice.

    adapter = Qwen3TtsCppTTSAdapter(
        endpoint="http://127.0.0.1:9766/v1",
        model="local-qwen3-tts",
    )

    # When: Orchestrator builds the buffered and chunked requests.

    buffered = adapter.build_speech_request(
        text="欢迎来到 BitNet 讲解。",
        voice="paimeng",
        ref_audio="file:///unused-clone-reference.wav",
        ref_text="未使用的克隆参考文本",
    )
    streaming = adapter.build_speech_request(
        text="欢迎来到 BitNet 讲解。",
        voice="paimeng",
        ref_audio="",
        ref_text="",
        streaming=True,
    )

    # Then: the speaker id is sent directly and no clone fields reach the engine.

    assert buffered.url == "http://127.0.0.1:9766/v1/audio/speech"
    assert buffered.json == {
        "model": "local-qwen3-tts",
        "input": "欢迎来到 BitNet 讲解。",
        "response_format": "wav",
        "voice": "paimeng",
    }
    assert streaming.json == {
        "model": "local-qwen3-tts",
        "input": "欢迎来到 BitNet 讲解。",
        "response_format": "pcm",
        "voice": "paimeng",
    }


def test_qwen3ttscpp_buffered_synthesis_returns_provider_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an engine that answers ``response_format: "wav"`` with one clip.

    adapter = Qwen3TtsCppTTSAdapter(
        endpoint="http://127.0.0.1:9766/v1",
        model="local-qwen3-tts",
    )
    calls: list[dict[str, object]] = []

    class Response:
        content: bytes = b"RIFFqwen3ttscpp"

        def raise_for_status(self) -> None:
            return None

    class Client:
        def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> Response:
            calls.append({"url": url, "json": json, "headers": headers})
            return Response()

        def close(self) -> None:
            return None

    def build_client(_adapter: Qwen3TtsCppTTSAdapter) -> Client:
        return Client()

    monkeypatch.setattr(Qwen3TtsCppTTSAdapter, "_client", build_client)

    # When: the buffered synthesis path runs.

    audio = adapter.synthesize(
        text="欢迎来到 BitNet 讲解。",
        voice="paimeng",
        ref_audio="",
        ref_text="",
    )

    # Then: the engine clip is returned and the request used the WAV transfer.

    assert audio == SynthesizedAudio(data=b"RIFFqwen3ttscpp", media_type="audio/wav")
    assert calls[0]["url"] == "http://127.0.0.1:9766/v1/audio/speech"
    assert calls[0]["json"] == {
        "model": "local-qwen3-tts",
        "input": "欢迎来到 BitNet 讲解。",
        "response_format": "wav",
        "voice": "paimeng",
    }
    assert calls[0]["headers"] == {
        "Accept": "audio/wav",
    }


def test_qwen3ttscpp_streams_raw_chunked_pcm_across_split_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a chunked PCM body whose chunks split a little-endian sample.

    adapter = Qwen3TtsCppTTSAdapter(
        endpoint="http://127.0.0.1:9766/v1",
        model="local-qwen3-tts",
        capability="streaming",
    )
    pcm_24khz = b"\x10\x20" * 480
    delivered = [pcm_24khz[:3], pcm_24khz[3:5], pcm_24khz[5:]]

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self) -> Iterator[bytes]:
            return iter(delivered)

    class Client:
        def stream(
            self,
            method: str,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> Response:
            _ = (method, url, json, headers)
            return Response()

        def close(self) -> None:
            return None

    def build_client(_adapter: Qwen3TtsCppTTSAdapter) -> Client:
        return Client()

    monkeypatch.setattr(Qwen3TtsCppTTSAdapter, "_client", build_client)

    # When: the dialect consumes the unframed body.

    chunks = tuple(
        adapter.stream_pcm16le(
            text="欢迎来到 BitNet 讲解。",
            voice="paimeng",
            ref_audio="",
            ref_text="",
        )
    )

    # Then: every whole sample is resampled 24 kHz to 16 kHz without a terminator.

    assert sum(len(chunk.data) for chunk in chunks) == 640
    assert all(chunk.sample_rate == 16_000 for chunk in chunks)


def test_qwen3ttscpp_rejects_body_that_ends_mid_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a chunked body that stops in the middle of one sample.

    adapter = Qwen3TtsCppTTSAdapter(
        endpoint="http://127.0.0.1:9766/v1",
        model="local-qwen3-tts",
        capability="streaming",
    )

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self) -> Iterator[bytes]:
            return iter([b"\x01\x02", b"\x03"])

    class Client:
        def stream(
            self,
            method: str,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> Response:
            _ = (method, url, json, headers)
            return Response()

        def close(self) -> None:
            return None

    def build_client(_adapter: Qwen3TtsCppTTSAdapter) -> Client:
        return Client()

    monkeypatch.setattr(Qwen3TtsCppTTSAdapter, "_client", build_client)

    # When / Then: the truncated sample fails closed instead of reaching RTP.

    with pytest.raises(ProviderResponseError) as error:
        _ = tuple(
            adapter.stream_pcm16le(
                text="欢迎来到 BitNet 讲解。",
                voice="paimeng",
                ref_audio="",
                ref_text="",
            )
        )

    assert error.value.reason == "incomplete_pcm"


def test_media_adapters_retain_configured_ca_path_for_provider_requests(
    ca_path: Path,
) -> None:
    # Given: configured OpenAI-compatible ASR and vLLM-Omni TTS providers.

    asr = OpenAICompatibleASRAdapter(
        endpoint="https://asr.example.test/v1",
        model="asr-model",
        ca_path=ca_path,
    )
    tts = VllmOmniTTSAdapter(
        endpoint="https://tts.example.test/v1",
        model="tts-model",
        ca_path=ca_path,
    )

    # When: the provider adapters are prepared for requests.

    # Then: both retain the shared Orchestrator CA path for their HTTPS transport.

    assert asr.ca_path == ca_path
    assert tts.ca_path == ca_path


@pytest.mark.parametrize(
    ("adapter_factory", "endpoint", "model"),
    [
        (OpenAICompatibleASRAdapter, " ", "local-asr"),
        (
            OpenAICompatibleASRAdapter,
            "http://127.0.0.1:8000/v1",
            " ",
        ),
        (VllmOmniTTSAdapter, " ", "vllm-omni"),
        (VllmOmniTTSAdapter, "http://127.0.0.1:8001/v1", " "),
        (AudioCppTTSAdapter, " ", "pocket-tts"),
        (AudioCppTTSAdapter, "http://127.0.0.1:8080/v1", " "),
    ],
)
def test_media_provider_rejects_blank_endpoint_or_model_before_network(
    adapter_factory: type[
        AudioCppTTSAdapter | OpenAICompatibleASRAdapter | VllmOmniTTSAdapter
    ],
    endpoint: str,
    model: str,
) -> None:
    # Given: malformed explicit provider configuration.

    # When / Then: construction fails before any request method can be reached.

    with pytest.raises(ValueError, match=r"endpoint|model"):
        _ = adapter_factory(endpoint=endpoint, model=model)


def test_vllm_sse_delta_is_decoded_and_resampled_to_onsite_pcm() -> None:
    payload = '{"type":"speech.audio.delta","audio":"AADoA9AH","response_format":"pcm"}'

    pcm = media_adapters._normalize_tts_sse(payload)  # pyright: ignore[reportPrivateUsage]

    assert pcm is not None
    converter = media_adapters._Pcm24khzTo16khzResampler()  # pyright: ignore[reportPrivateUsage]
    assert len(converter.push(pcm)) == 4
    assert (
        media_adapters._normalize_tts_sse(  # pyright: ignore[reportPrivateUsage]
            '{"type":"speech.audio.done","usage":{}}'
        )
        is None
    )


def test_tts_sse_resampler_preserves_pcm_across_delta_boundaries() -> None:
    pcm = b"\x00\x00\xe8\x03\xd0\x07\xb8\x0b\xa0\x0f"
    resampler = media_adapters._Pcm24khzTo16khzResampler()  # pyright: ignore[reportPrivateUsage]

    first = resampler.push(pcm[:4])
    second = resampler.push(pcm[4:])

    one_shot = media_adapters._Pcm24khzTo16khzResampler()  # pyright: ignore[reportPrivateUsage]
    expected = one_shot.push(pcm)
    assert first + second == expected


def _data_url(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")

    return f"data:audio/wav;base64,{encoded}"
