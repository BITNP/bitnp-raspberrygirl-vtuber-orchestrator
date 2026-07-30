
from __future__ import annotations

import pytest

from orchestrator.config import load_config_from_env
from orchestrator.llm import OpenAICompatibleASRAdapter, VllmOmniTTSAdapter
from orchestrator.pipeline_contracts import ASRAudienceEvent


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
        ref_audio="file:///fixtures/raspberry.wav",
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
        "ref_audio": "file:///fixtures/raspberry.wav",
        "ref_text": "参考音色文本",
    }


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
