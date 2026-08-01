
from __future__ import annotations

import base64
import ssl
from typing import TYPE_CHECKING

import pytest

from orchestrator import media_adapters
from orchestrator.config import load_config_from_env
from orchestrator.llm import OpenAICompatibleASRAdapter, VllmOmniTTSAdapter
from orchestrator.pipeline_contracts import ASRAudienceEvent

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ca_path(tmp_path: Path) -> Path:
    certificate = ssl.create_default_context().get_ca_certs(binary_form=True)[0]
    path = tmp_path / "ca.pem"
    _ = path.write_text(ssl.DER_cert_to_PEM_cert(certificate), encoding="ascii")
    return path


class _TtsResponse:
    def getheader(self, _name: str, _default: str) -> str:
        return "audio/wav"

    def read(self) -> bytes:
        return b"audio"


class _TtsConnection:
    def request(
        self, _method: str, _path: str, *, body: bytes, headers: dict[str, str]
    ) -> None:
        _ = (body, headers)

    def getresponse(self) -> _TtsResponse:
        return _TtsResponse()

    def close(self) -> None:
        return


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

    assert request.json == {
        "model": "vllm-omni",
        "input": "欢迎来到 BitNet 讲解。",
        "voice": "raspberry",
        "task_type": "Base",
        "ref_audio": "https://media.example.test/raspberry.wav",
        "ref_text": "参考音色文本",
    }


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

    assert request.json["ref_audio"] == _data_url(reference.read_bytes())


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

    assert request.json["ref_audio"] == _data_url(reference.read_bytes())


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

    assert request.json["ref_audio"] == ref_audio


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


def test_vllm_omni_https_connection_receives_verified_configured_ca_context(
    monkeypatch: pytest.MonkeyPatch, ca_path: Path, tmp_path: Path
) -> None:
    # Given: a secure vLLM-Omni endpoint and configured local CA bundle.


    contexts: list[ssl.SSLContext] = []

    def connect(
        _host: str, *, timeout: int, context: ssl.SSLContext
    ) -> _TtsConnection:
        _ = timeout
        contexts.append(context)
        return _TtsConnection()

    monkeypatch.setattr(media_adapters, "HTTPSConnection", connect)

    reference = tmp_path / "voice.wav"
    _ = reference.write_bytes(b"RIFFvoice")

    # When: the production TTS adapter sends its speech request.

    audio = VllmOmniTTSAdapter(
        endpoint="https://tts.example.test/v1",
        model="tts-model",
        ca_path=ca_path,
    ).synthesize(
        text="你好", voice="raspberry", ref_audio=reference.as_uri(), ref_text="参考"
    )

    # Then: HTTPSConnection receives the existing verified CA-based context.

    assert audio.data == b"audio"
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert contexts[0].check_hostname is True


def test_vllm_omni_http_connection_omits_tls_context_even_with_configured_ca_bundle(
    monkeypatch: pytest.MonkeyPatch, ca_path: Path, tmp_path: Path
) -> None:
    # Given: a plaintext vLLM-Omni endpoint and configured local CA bundle.


    connections: list[_TtsConnection] = []

    def connect(_host: str, *, timeout: int) -> _TtsConnection:
        _ = timeout
        connection = _TtsConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(media_adapters, "HTTPConnection", connect)

    reference = tmp_path / "voice.wav"
    _ = reference.write_bytes(b"RIFFvoice")

    # When: the production TTS adapter sends its speech request.

    _ = VllmOmniTTSAdapter(
        endpoint="http://tts.example.test/v1",
        model="tts-model",
        ca_path=ca_path,
    ).synthesize(
        text="你好", voice="raspberry", ref_audio=reference.as_uri(), ref_text="参考"
    )

    # Then: HTTP uses its original constructor shape without TLS context.

    assert len(connections) == 1


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
    ],
)
def test_media_provider_rejects_blank_endpoint_or_model_before_network(
    adapter_factory: type[OpenAICompatibleASRAdapter | VllmOmniTTSAdapter],
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
