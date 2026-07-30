"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from orchestrator.streaming_contracts import CancellationEpoch, StreamKey
from orchestrator.tts_rtp import (
    Pcm16leChunk,
    PcmChannels,
    PcmChunkError,
    PcmSampleRate,
    TtsPcmRtpPacketizer,
)


@dataclass(slots=True)
class _FakeClock:
    """类契约说明.

    职责: 保存 _FakeClock
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: now_ms。 方法: advance。
    """

    now_ms: int

    def advance(self, milliseconds: int) -> None:
        """函数契约说明.

        功能: 执行 advance 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 milliseconds:
        int。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.now_ms += milliseconds


def test_streaming_pcm_chunks_emit_continuous_network_order_l16_rtp() -> None:
    # Given: uneven PCM16LE chunks that split both a sample and a 20 ms RTP frame.

    """函数契约说明.

    功能: 验证 streaming pcm chunks emit
    continuous network order l16 rtp
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    packetizer = TtsPcmRtpPacketizer(
        stream=StreamKey("session-streaming", "stream-streaming"),
        cancellation_epoch=CancellationEpoch(0),
    )

    first = Pcm16leChunk(b"\x34\x12" * 319 + b"\xcd")

    second = Pcm16leChunk(b"\xab" + b"\x78\x56" * 321)

    # When: the provider yields the chunks in order and the stream completes.

    packets = packetizer.push(first) + packetizer.push(second) + packetizer.finish()

    # Then: samples are byte-swapped, frames contiguous, and final audio padded.

    assert [packet[:2] for packet in packets] == [b"\x80\x60", b"\x80\x60", b"\x80\x60"]

    assert [int.from_bytes(packet[2:4], "big") for packet in packets] == [0, 1, 2]

    assert [int.from_bytes(packet[4:8], "big") for packet in packets] == [
        96_000,
        96_320,
        96_640,
    ]

    assert packets[0][12:] == b"\x12\x34" * 319 + b"\xab\xcd"

    assert packets[1][12:16] == b"\x56\x78" * 2

    assert packets[2][12:] == b"\x56\x78" + bytes(638)


def test_streaming_pcm_rejects_bad_metadata_and_incomplete_sample() -> None:
    # Given: a packetizer and chunk values outside the fixed provider boundary.

    """函数契约说明.

    功能: 验证 streaming pcm rejects bad
    metadata and incomplete sample
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    packetizer = TtsPcmRtpPacketizer(
        stream=StreamKey("session-streaming", "stream-malformed"),
        cancellation_epoch=CancellationEpoch(0),
    )

    # When / Then: arbitrary formats and a dangling final sample cannot become RTP.

    with pytest.raises(PcmChunkError, match="sample_rate"):
        _ = Pcm16leChunk(b"\x00\x00", sample_rate=PcmSampleRate(24_000))

    with pytest.raises(PcmChunkError, match="channels"):
        _ = Pcm16leChunk(b"\x00\x00", channels=PcmChannels(2))

    assert packetizer.push(Pcm16leChunk(b"\x01")) == ()

    with pytest.raises(PcmChunkError, match="incomplete_sample"):
        _ = packetizer.finish()


def test_generated_ssrc_is_fresh_per_epoch_and_cancelled_packetizer_emits_nothing() -> (
    None
):
    # Given: two epochs of one routed stream and a deterministic endpoint-final clock.

    """函数契约说明.

    功能: 验证 generated ssrc is fresh per
    epoch and cancelled packetizer emits
    nothing 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    stream = StreamKey("session-streaming", "stream-epochs")

    clock = _FakeClock(now_ms=1_000)

    first = TtsPcmRtpPacketizer(stream=stream, cancellation_epoch=CancellationEpoch(0))

    replacement = TtsPcmRtpPacketizer(
        stream=stream,
        cancellation_epoch=CancellationEpoch(1),
    )

    # When: first media emits, then its packetizer is cancelled before another chunk.

    clock.advance(400)

    first_packets = first.push(Pcm16leChunk(b"\x34\x12" * 320))

    first.cancel()

    cancelled_packets = first.push(Pcm16leChunk(b"\x78\x56" * 320))

    replacement_packets = replacement.push(Pcm16leChunk(b"\xbc\x9a" * 320))

    # Then: first media meets the fake-clock budget and epochs cannot share an SSRC.

    assert clock.now_ms - 1_000 <= 1_500

    assert len(first_packets) == 1

    assert cancelled_packets == ()

    assert len(replacement_packets) == 1

    assert first.ssrc != 0

    assert replacement.ssrc != 0

    assert first.ssrc != replacement.ssrc

    assert int.from_bytes(first_packets[0][8:12], "big") == first.ssrc

    assert int.from_bytes(replacement_packets[0][8:12], "big") == replacement.ssrc
