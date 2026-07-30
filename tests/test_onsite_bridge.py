"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import json
from dataclasses import dataclass, field

import pytest

from orchestrator.config import load_config_from_env
from orchestrator.funasr_adapter import FunASRWebSocketAdapter
from orchestrator.onsite_bridge import OnsiteBridgeConfigError, build_onsite_bridge


@dataclass(slots=True)
class _NativeFunASRConnection:
    """类契约说明.

    职责: 保存 _NativeFunASRConnection
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: sent、responses。 方法:
    send、recv、close。
    """

    sent: list[str | bytes] = field(default_factory=list)

    responses: list[str] = field(
        default_factory=lambda: ['{"text":"已识别","is_final":true}']
    )

    def send(self, message: str | bytes) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 message: str |
        bytes。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str:
        """函数契约说明.

        功能: 执行 recv 的同步逻辑,并协调 OSError,
        pop。
        参数: self 表示当前实例。 timeout: float
        | None。 可省略。
        契约: 同步调用。 返回 `str`。 可能抛出
        OSError。
        """

        _ = timeout

        if self.responses:
            return self.responses.pop(0)

        message = "closed"

        raise OSError(message)

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return


def _install_native_connection(
    monkeypatch: pytest.MonkeyPatch, connection: _NativeFunASRConnection
) -> None:
    """函数契约说明.

    功能: 执行 _install_native_connection
    的同步逻辑,并协调 setattr。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。 connection:
    _NativeFunASRConnection。 必填。
    契约: 同步调用。 返回 `None`。
    """

    def connect(*args: object, **kwargs: object) -> _NativeFunASRConnection:
        """函数契约说明.

        功能: 执行 connect 的同步逻辑,并产出 _。
        参数: *args: object。 必填。 **kwargs:
        object。 必填。
        契约: 同步调用。 返回
        `_NativeFunASRConnection`。
        """

        _ = (args, kwargs)

        return connection

    monkeypatch.setattr("orchestrator.funasr_adapter.connect", connect)


def test_build_onsite_bridge_rejects_missing_llm_endpoint() -> None:
    # Given: valid onsite ASR and TTS providers but an incomplete real LLM.

    """函数契约说明.

    功能: 验证 build onsite bridge rejects
    missing llm endpoint 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 build onsite bridge selects
    native funasr streaming adapter
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 native funasr bridge declares
    pcm for raw pcm16le payload
    的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。
    契约: 同步调用。 返回 `None`。
    """

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
