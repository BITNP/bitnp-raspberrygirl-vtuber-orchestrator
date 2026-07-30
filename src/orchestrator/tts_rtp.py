"""模块契约说明.

职责: 提供 orchestrator.tts_rtp
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NewType,
    Protocol,
    override,
    runtime_checkable,
)
from zlib import crc32

if TYPE_CHECKING:
    from orchestrator.provider_streaming import ProviderCancellationHandle
    from orchestrator.streaming_contracts import CancellationEpoch, StreamKey


PcmSampleRate = NewType("PcmSampleRate", int)

PcmChannels = NewType("PcmChannels", int)


_SAMPLE_RATE: Final = PcmSampleRate(16_000)

_CHANNELS: Final = PcmChannels(1)

_PCM_SAMPLE_BYTES: Final = 2

_RTP_HEADER_BYTES: Final = 12

_L16_FRAME_BYTES: Final = 640

_SAMPLES_PER_FRAME: Final = 320

_RTP_TIMESTAMP_START: Final = 96_000

_RTP_VERSION_PAYLOAD_TYPE: Final = b"\x80\x60"


@dataclass(frozen=True, slots=True)
class PcmChunkError(ValueError):
    """类契约说明.

    职责: 保存 PcmChunkError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。 方法: __str__。
    """

    reason: Literal["sample_rate", "channels", "incomplete_sample"]

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return self.reason


@dataclass(frozen=True, slots=True)
class Pcm16leChunk:
    """类契约说明.

    职责: 保存 Pcm16leChunk
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: data、sample_rate、channels。
    方法: __post_init__。
    """

    data: bytes

    sample_rate: PcmSampleRate = _SAMPLE_RATE

    channels: PcmChannels = _CHANNELS

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化 Pcm16leChunk
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。 可能抛出
        PcmChunkError。
        """
        if self.sample_rate != _SAMPLE_RATE:
            raise PcmChunkError(reason="sample_rate")

        if self.channels != _CHANNELS:
            raise PcmChunkError(reason="channels")


@runtime_checkable
class StreamingTtsAdapter(Protocol):
    """类契约说明.

    职责: 声明 StreamingTtsAdapter
    协议接口,约束实现方必须提供的行为。
    契约: 方法: stream_pcm16le。
    """

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> tuple[Pcm16leChunk, ...]:
        """函数契约说明.

        功能: 执行 stream_pcm16le
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 text: str。 必填。
        voice: str。 必填。 ref_audio: str。
        必填。 ref_text: str。 必填。
        cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回
        `tuple[Pcm16leChunk, ...]`。
        """
        ...


@dataclass(slots=True)
class TtsPcmRtpPacketizer:
    """类契约说明.

    职责: 保存 TtsPcmRtpPacketizer
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、cancellation_epoch、ss
    rc、_pending、_sequence、_timestamp。
    方法: __post_init__、push、finish、cancel
    、_packet。
    """

    stream: StreamKey

    cancellation_epoch: CancellationEpoch

    ssrc: int = field(init=False)

    _pending: bytes = field(default=b"", init=False)

    _sequence: int = field(default=0, init=False)

    _timestamp: int = field(default=_RTP_TIMESTAMP_START, init=False)

    _cancelled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化 TtsPcmRtpPacketizer
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self.ssrc = generated_ssrc(self.stream, self.cancellation_epoch)

    def push(self, chunk: Pcm16leChunk) -> tuple[bytes, ...]:
        """函数契约说明.

        功能: 执行 push 的同步逻辑,并协调 tuple,
        len, _packet, range。
        参数: self 表示当前实例。 chunk:
        Pcm16leChunk。 必填。
        契约: 同步调用。 返回 `tuple[bytes,
        ...]`。
        """
        if self._cancelled:
            return ()

        self._pending += chunk.data

        complete_bytes = len(self._pending) - len(self._pending) % _PCM_SAMPLE_BYTES

        complete = self._pending[:complete_bytes]

        self._pending = self._pending[complete_bytes:]

        frame_bytes = len(complete) - len(complete) % _L16_FRAME_BYTES

        frames = complete[:frame_bytes]

        self._pending = complete[frame_bytes:] + self._pending

        return tuple(
            self._packet(frame)
            for offset in range(0, len(frames), _L16_FRAME_BYTES)
            for frame in (frames[offset : offset + _L16_FRAME_BYTES],)
        )

    def finish(self) -> tuple[bytes, ...]:
        """函数契约说明.

        功能: 执行 finish 的同步逻辑,并协调
        PcmChunkError, bytes, _packet,
        len。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[bytes,
        ...]`。 可能抛出 PcmChunkError。
        """
        if self._cancelled:
            return ()

        if len(self._pending) % _PCM_SAMPLE_BYTES != 0:
            raise PcmChunkError(reason="incomplete_sample")

        if self._pending == b"":
            return ()

        frame = self._pending + bytes(_L16_FRAME_BYTES - len(self._pending))

        self._pending = b""

        return (self._packet(frame),)

    def cancel(self) -> None:
        """函数契约说明.

        功能: 执行 cancel 的同步逻辑,并产出
        _cancelled, _pending。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self._cancelled = True

        self._pending = b""

    def _packet(self, pcm16le: bytes) -> bytes:
        """函数契约说明.

        功能: 执行 _packet 的同步逻辑,并协调 join,
        to_bytes, range, len。
        参数: self 表示当前实例。 pcm16le: bytes。
        必填。
        契约: 同步调用。 返回 `bytes`。
        """
        payload = b"".join(
            pcm16le[offset : offset + _PCM_SAMPLE_BYTES][::-1]
            for offset in range(0, len(pcm16le), _PCM_SAMPLE_BYTES)
        )

        packet = b"".join(
            (
                _RTP_VERSION_PAYLOAD_TYPE,
                self._sequence.to_bytes(2, "big"),
                self._timestamp.to_bytes(4, "big"),
                self.ssrc.to_bytes(4, "big"),
                payload,
            )
        )

        self._sequence = (self._sequence + 1) % (1 << 16)

        self._timestamp = (self._timestamp + _SAMPLES_PER_FRAME) % (1 << 32)

        return packet


def generated_ssrc(stream: StreamKey, cancellation_epoch: CancellationEpoch) -> int:
    """函数契约说明.

    功能: 执行 generated_ssrc 的同步逻辑,并协调
    crc32, encode, int。
    参数: stream: StreamKey。 必填。
    cancellation_epoch:
    CancellationEpoch。 必填。
    契约: 同步调用。 返回 `int`。
    """
    stream_seed = crc32(f"{stream.session_id}:{stream.stream_id}".encode())

    derived = stream_seed ^ int(cancellation_epoch)

    return 1 if derived == 0 else derived
