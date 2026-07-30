
import json
from dataclasses import dataclass, field

import pytest

from orchestrator.config import load_config_from_env
from orchestrator.funasr_adapter import FunASRWebSocketAdapter
from orchestrator.onsite_bridge import OnsiteBridgeConfigError, build_onsite_bridge


@dataclass(slots=True)
class _NativeFunASRConnection:

    sent: list[str | bytes] = field(default_factory=list)

    responses: list[str] = field(
        default_factory=lambda: ['{"text":"已识别","is_final":true}']
    )

    def send(self, message: str | bytes) -> None:

        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str:

        _ = timeout

        if self.responses:
            return self.responses.pop(0)

        message = "closed"

        raise OSError(message)

    def close(self) -> None:

        return


def _install_native_connection(
    monkeypatch: pytest.MonkeyPatch, connection: _NativeFunASRConnection
) -> None:

    def connect(*args: object, **kwargs: object) -> _NativeFunASRConnection:

        _ = (args, kwargs)

        return connection

    monkeypatch.setattr("orchestrator.funasr_adapter.connect", connect)


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


def test_build_onsite_bridge_selects_native_funasr_streaming_adapter() -> None:
    # Given: complete onsite configuration for native FunASR WebSocket ASR.


    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
            "ORCHESTRATOR_LLM_ENDPOINT": "https://llm.example.test/v1",
            "ORCHESTRATOR_LLM_MODEL": "onsite-model",
            "ORCHESTRATOR_LLM_API_KEY": "onsite-test-key",
            "ORCHESTRATOR_ASR_PROVIDER": "funasr",
            "ORCHESTRATOR_ASR_ENDPOINT": "ws://asr.example.test:10095",
            "ORCHESTRATOR_ASR_MODEL": "paraformer",
            "ORCHESTRATOR_TTS_PROVIDER": "vllm_omni",
            "ORCHESTRATOR_TTS_ENDPOINT": "https://tts.example.test/v1",
            "ORCHESTRATOR_TTS_MODEL": "tts-model",
        }
    )

    # When: the onsite bridge composes the selected ASR provider.

    bridge = build_onsite_bridge(
        config,
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
    )

    # Then: native WebSocket streaming is selected without changing RTP boundaries.

    assert isinstance(bridge.asr, FunASRWebSocketAdapter)


def test_native_funasr_bridge_declares_pcm_for_raw_pcm16le_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an onsite bridge using native FunASR and a captured provider session.


    connection = _NativeFunASRConnection()

    _install_native_connection(monkeypatch, connection)

    bridge = build_onsite_bridge(
        load_config_from_env(
            {
                "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
                "ORCHESTRATOR_LLM_ENDPOINT": "https://llm.example.test/v1",
                "ORCHESTRATOR_LLM_MODEL": "onsite-model",
                "ORCHESTRATOR_LLM_API_KEY": "onsite-test-key",
                "ORCHESTRATOR_ASR_PROVIDER": "funasr",
                "ORCHESTRATOR_ASR_ENDPOINT": "ws://asr.example.test:10095",
                "ORCHESTRATOR_ASR_MODEL": "paraformer",
                "ORCHESTRATOR_TTS_PROVIDER": "vllm_omni",
                "ORCHESTRATOR_TTS_ENDPOINT": "https://tts.example.test/v1",
                "ORCHESTRATOR_TTS_MODEL": "tts-model",
            }
        ),
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
    )

    # When: the actual bridge transcribes one canonical network-order L16 utterance.

    result = bridge.transcribe_endpoint(b"\x12\x34\xab\xcd", sequence=1)

    # Then: native PCM control and payload agree, with no RIFF container admitted.

    assert result is not None

    assert json.loads(connection.sent[0])["wav_format"] == "pcm"

    assert connection.sent[1] == b"\x34\x12\xcd\xab"

    assert connection.sent[1][:4] != b"RIFF"
