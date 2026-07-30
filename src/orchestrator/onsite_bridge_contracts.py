"""模块契约说明.

职责: 提供
orchestrator.onsite_bridge_contracts
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, override

from orchestrator.transport_hub import L16_FRAME_BYTES

if TYPE_CHECKING:
    from orchestrator.media_adapters import SynthesizedAudio
    from orchestrator.pipeline_contracts import ASRAudienceEvent
    from orchestrator.provider_streaming import ProviderCancellationHandle


_SAMPLE_RATE: Final = 16_000

_PCM16_BYTES: Final = 2


@dataclass(frozen=True, slots=True)
class OnsiteBridgeConfigError(ValueError):
    """类契约说明.

    职责: 保存 OnsiteBridgeConfigError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: field_name。 方法: __str__。
    """

    field_name: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"onsite bridge configuration is incomplete: {self.field_name}"


@dataclass(frozen=True, slots=True)
class OnsiteBridgeMediaError(ValueError):
    """类契约说明.

    职责: 保存 OnsiteBridgeMediaError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。 方法: __str__。
    """

    reason: Literal["media_type", "wave_format"]

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        match self.reason:
            case "media_type":
                return "onsite TTS must return audio/wav"

            case "wave_format":
                return "onsite TTS WAV must be 16 kHz mono PCM16"


class AsrAdapter(Protocol):
    """类契约说明.

    职责: 声明 AsrAdapter 协议接口,约束实现方必须提供的行为。
    契约: 方法: transcribe。
    """

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
        """函数契约说明.

        功能: 执行 transcribe 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 audio: bytes。
        必填。 filename: str。 必填。
        received_at_ms: int。 必填。
        segment_id: str。 必填。 seq: int。
        必填。 cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回 `ASRAudienceEvent |
        None`。
        """
        ...


class TtsAdapter(Protocol):
    """类契约说明.

    职责: 声明 TtsAdapter 协议接口,约束实现方必须提供的行为。
    契约: 方法: synthesize。
    """

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:
        """函数契约说明.

        功能: 执行 synthesize 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 text: str。 必填。
        voice: str。 必填。 ref_audio: str。
        必填。 ref_text: str。 必填。
        cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回 `SynthesizedAudio`。
        """
        ...


def wav_from_l16(payload: bytes) -> bytes:
    """函数契约说明.

    功能: 执行 wav_from_l16 的同步逻辑,并协调
    BytesIO, getvalue, open,
    setnchannels。
    参数: payload: bytes。 必填。
    契约: 同步调用。 返回 `bytes`。
    """
    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(_PCM16_BYTES)

        audio.setframerate(_SAMPLE_RATE)

        audio.writeframes(_swap_pcm16_byte_order(payload))

    return output.getvalue()


def pcm16le_from_l16(payload: bytes) -> bytes:
    """函数契约说明.

    功能: 执行 pcm16le_from_l16 的同步逻辑,并协调
    _swap_pcm16_byte_order。
    参数: payload: bytes。 必填。
    契约: 同步调用。 返回 `bytes`。
    """
    return _swap_pcm16_byte_order(payload)


def l16_from_wav(response: SynthesizedAudio) -> bytes:
    """函数契约说明.

    功能: 执行 l16_from_wav 的同步逻辑,并协调
    OnsiteBridgeMediaError, open,
    _swap_pcm16_byte_order, BytesIO。
    参数: response: SynthesizedAudio。 必填。
    契约: 同步调用。 返回 `bytes`。 可能抛出
    OnsiteBridgeMediaError。
    """
    if response.media_type != "audio/wav":
        raise OnsiteBridgeMediaError(reason="media_type")

    with wave.open(io.BytesIO(response.data), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != _PCM16_BYTES
            or audio.getframerate() != _SAMPLE_RATE
            or audio.getcomptype() != "NONE"
        ):
            raise OnsiteBridgeMediaError(reason="wave_format")

        return _swap_pcm16_byte_order(audio.readframes(audio.getnframes()))


def pad_l16_frames(audio: bytes) -> bytes:
    """函数契约说明.

    功能: 执行 pad_l16_frames 的同步逻辑,并协调 len,
    bytes。
    参数: audio: bytes。 必填。
    契约: 同步调用。 返回 `bytes`。
    """
    remainder = len(audio) % L16_FRAME_BYTES

    if remainder == 0:
        return audio

    return audio + bytes(L16_FRAME_BYTES - remainder)


def generated_ssrc(mic_ssrc: int) -> int:
    """函数契约说明.

    功能: 执行 generated_ssrc 的同步逻辑,并产出
    generated。
    参数: mic_ssrc: int。 必填。
    契约: 同步调用。 返回 `int`。
    """
    generated = mic_ssrc ^ 0xA5A5_A5A5

    return 1 if generated == 0 else generated


def _swap_pcm16_byte_order(payload: bytes) -> bytes:
    """函数契约说明.

    功能: 执行 _swap_pcm16_byte_order
    的同步逻辑,并协调 range, join, len。
    参数: payload: bytes。 必填。
    契约: 同步调用。 返回 `bytes`。
    """
    samples = range(0, len(payload), _PCM16_BYTES)

    return b"".join(payload[offset : offset + _PCM16_BYTES][::-1] for offset in samples)
