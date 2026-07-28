import pytest

from orchestrator.config import load_config_from_env
from orchestrator.onsite_bridge import OnsiteBridgeConfigError, build_onsite_bridge


def test_build_onsite_bridge_rejects_missing_llm_endpoint() -> None:
    # Given: valid onsite ASR and TTS providers but an incomplete real LLM.
    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
            "ORCHESTRATOR_LLM_MODEL": "onsite-model",
            "ORCHESTRATOR_LLM_API_KEY": "onsite-test-key",
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
