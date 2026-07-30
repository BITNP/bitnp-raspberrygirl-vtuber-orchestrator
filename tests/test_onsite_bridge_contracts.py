"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import io
import wave

from orchestrator.media_adapters import SynthesizedAudio
from orchestrator.onsite_bridge_contracts import l16_from_wav, wav_from_l16


def test_wav_from_l16_converts_network_samples_to_pcm16_little_endian() -> None:
    # Given: non-symmetric signed PCM samples in canonical RTP L16 network order.

    """函数契约说明.

    功能: 验证 wav from l16 converts network
    samples to pcm16 little endian
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    network_l16 = b"\x12\x34\xab\xcd"

    # When: the payload crosses into the WAV/ASR boundary.

    wav = wav_from_l16(network_l16)

    # Then: WAV PCM16 bytes use the required little-endian sample representation.

    assert _wav_payload(wav) == b"\x34\x12\xcd\xab"


def test_l16_from_wav_converts_pcm16_little_endian_to_network_samples() -> None:
    # Given: non-symmetric PCM16 little-endian samples from the TTS/WAV boundary.

    """函数契约说明.

    功能: 验证 l16 from wav converts pcm16
    little endian to network samples
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    response = SynthesizedAudio(_wav(b"\x34\x12\xcd\xab"), "audio/wav")

    # When: the response crosses into canonical RTP L16 payload bytes.

    network_l16 = l16_from_wav(response)

    # Then: outgoing RTP receives big-endian network-order PCM samples.

    assert network_l16 == b"\x12\x34\xab\xcd"


def _wav(payload: bytes) -> bytes:
    """函数契约说明.

    功能: 执行 _wav 的同步逻辑,并协调 BytesIO,
    getvalue, open, setnchannels。
    参数: payload: bytes。 必填。
    契约: 同步调用。 返回 `bytes`。
    """

    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(2)

        audio.setframerate(16_000)

        audio.writeframes(payload)

    return output.getvalue()


def _wav_payload(data: bytes) -> bytes:
    """函数契约说明.

    功能: 执行 _wav_payload 的同步逻辑,并协调 open,
    readframes, BytesIO, getnframes。
    参数: data: bytes。 必填。
    契约: 同步调用。 返回 `bytes`。
    """

    with wave.open(io.BytesIO(data), "rb") as audio:
        return audio.readframes(audio.getnframes())
