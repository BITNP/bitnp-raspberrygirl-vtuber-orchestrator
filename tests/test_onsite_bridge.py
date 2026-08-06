from pathlib import Path

import pytest

from orchestrator import onsite_bridge
from orchestrator.config import load_config_from_env
from orchestrator.llm import VllmOmniTTSAdapter
from orchestrator.onsite_bridge import OnsiteBridgeConfigError, build_onsite_bridge
from orchestrator.openai_llm_runtime import AsyncOpenAICompatibleLLMRuntime


def test_build_onsite_bridge_rejects_missing_llm_endpoint() -> None:
    # Given: valid onsite ASR and TTS providers but an incomplete real LLM.

    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
            "ORCHESTRATOR_LLM_MODEL": "onsite-model",
            "ORCHESTRATOR_LLM_API_KEY": "onsite-test-key",
            "ORCHESTRATOR_LLM_REASONING_DIALECT": "deepseek",
            "ORCHESTRATOR_ASR_PROVIDER": "openai_compatible",
            "ORCHESTRATOR_ASR_ENDPOINT": "https://asr.example.test/v1",
            "ORCHESTRATOR_ASR_MODEL": "asr-model",
            "ORCHESTRATOR_TTS_PROVIDER": "vllm_omni",
            "ORCHESTRATOR_TTS_ENDPOINT": "https://tts.example.test/v1",
            "ORCHESTRATOR_TTS_MODEL": "tts-model",
        }
    )

    # When / Then: onsite composition refuses startup before any listener exists.

    with pytest.raises(OnsiteBridgeConfigError) as error:
        _ = build_onsite_bridge(
            config,
            voice="raspberry",
            ref_audio="file:///voice.wav",
            ref_text="reference",
        )

    assert str(error.value) == (
        "onsite bridge configuration is incomplete: llm_provider_or_llm_configuration"
    )


def test_build_onsite_bridge_has_no_orchestrator_asr_adapter() -> None:
    # Given: complete LLM/TTS configuration with no ASR settings.

    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
            "ORCHESTRATOR_LLM_ENDPOINT": "https://llm.example.test/v1",
            "ORCHESTRATOR_LLM_MODEL": "onsite-model",
            "ORCHESTRATOR_LLM_API_KEY": "onsite-test-key",
            "ORCHESTRATOR_LLM_REASONING_DIALECT": "deepseek",
            "ORCHESTRATOR_LLM_BRAIN_MODEL": "brain-model",
            "ORCHESTRATOR_LLM_MAINTENANCE_MODEL": "maintenance-model",
            "ORCHESTRATOR_TLS_CA_PATH": "/run/secrets/onsite-ca.pem",
            "ORCHESTRATOR_TTS_PROVIDER": "vllm_omni",
            "ORCHESTRATOR_TTS_ENDPOINT": "https://tts.example.test/v1",
            "ORCHESTRATOR_TTS_MODEL": "tts-model",
        }
    )

    # When: the onsite bridge composes the output-only provider set.

    bridge = build_onsite_bridge(
        config,
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
    )

    # Then: the output bridge has no local ASR or transcription surface.
    assert not hasattr(bridge, "asr")
    assert not hasattr(bridge, "transcribe_endpoint")
    assert isinstance(bridge.llm, AsyncOpenAICompatibleLLMRuntime)
    assert bridge.llm.reasoning_dialect == "deepseek"
    assert bridge.llm.brain_model == "brain-model"
    assert bridge.llm.maintenance_model == "maintenance-model"


def test_build_onsite_bridge_propagates_ca_path_to_http_provider_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: complete HTTPS provider configuration with one shared CA bundle.

    ca_path = Path("/run/secrets/onsite-ca.pem")
    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
            "ORCHESTRATOR_LLM_ENDPOINT": "https://llm.example.test/v1",
            "ORCHESTRATOR_LLM_MODEL": "onsite-model",
            "ORCHESTRATOR_LLM_API_KEY": "onsite-test-key",
            "ORCHESTRATOR_LLM_REASONING_DIALECT": "deepseek",
            "ORCHESTRATOR_TTS_PROVIDER": "vllm_omni",
            "ORCHESTRATOR_TTS_ENDPOINT": "https://tts.example.test/v1",
            "ORCHESTRATOR_TTS_MODEL": "tts-model",
            "ORCHESTRATOR_TLS_CA_PATH": str(ca_path),
        }
    )
    llm_ca_paths: list[Path | None] = []

    def build_llm(
        endpoint: str,
        model: str,
        api_key: str,
        reasoning_dialect: str,
        *,
        ca_path: Path | None,
        **kwargs: object,
    ) -> AsyncOpenAICompatibleLLMRuntime:
        _ = reasoning_dialect, kwargs
        llm_ca_paths.append(ca_path)
        return AsyncOpenAICompatibleLLMRuntime(
            endpoint,
            model,
            api_key,
            "deepseek",
            ca_path=ca_path,
        )

    monkeypatch.setattr(onsite_bridge, "AsyncOpenAICompatibleLLMRuntime", build_llm)

    # When: the onsite composition root builds only LLM/TTS HTTP adapters.

    bridge = build_onsite_bridge(
        config,
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
    )
    # Then: each remaining adapter retains the one configured CA path.

    assert llm_ca_paths == [ca_path]
    assert isinstance(bridge.tts, VllmOmniTTSAdapter)
    assert bridge.tts.ca_path == ca_path


def test_onsite_bridge_has_no_retired_direct_transcription_surface() -> None:
    bridge = build_onsite_bridge(
        load_config_from_env(
            {
                "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
                "ORCHESTRATOR_LLM_ENDPOINT": "https://llm.example.test/v1",
                "ORCHESTRATOR_LLM_MODEL": "onsite-model",
                "ORCHESTRATOR_LLM_API_KEY": "onsite-test-key",
                "ORCHESTRATOR_LLM_REASONING_DIALECT": "deepseek",
                "ORCHESTRATOR_TTS_PROVIDER": "vllm_omni",
                "ORCHESTRATOR_TTS_ENDPOINT": "https://tts.example.test/v1",
                "ORCHESTRATOR_TTS_MODEL": "tts-model",
            }
        ),
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
    )

    assert not hasattr(bridge, "ingest_mic_rtp")
    assert not hasattr(bridge, "transcribe_endpoint")
