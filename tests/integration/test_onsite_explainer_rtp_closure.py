"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from orchestrator.llm import MockLLMAdapter
from orchestrator.modes import ModePolicy
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    AudioMetadata,
    MockSynthesisResult,
    PipelineConfig,
)
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.transport_hub import RtpHub

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from orchestrator.streaming_contracts import CancellationEpoch, StreamKey


if TYPE_CHECKING:
    from orchestrator.json_boundary import JsonValue


SESSION_ID: Final = "session-onsite-001"

STREAM_ID: Final = "mic-onsite-001"

MIC_PEER: Final = ("192.0.2.20", 42_000)

SOUND_PEER: Final = ("192.0.2.21", 42_001)

MIC_SSRC: Final = 0x0102_0304

TTS_SSRC: Final = 0x0506_0708

SYNTHESIZED_L16_PAYLOAD: Final = b"\x10\x20" * 320


@dataclass(slots=True)
class FakeDatagramTransport:
    """类契约说明.

    职责: 保存 FakeDatagramTransport
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: sent。 方法: sendto、close。
    """

    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 data: bytes。
        必填。 addr: tuple[str, int]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.sent.append((data, addr))

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return


@dataclass(slots=True)
class FakeOnsiteExplainerBridge:
    """类契约说明.

    职责: 保存 FakeOnsiteExplainerBridge
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: asr_final、mic_packets、answer
    s、_output、_tasks。 方法: set_output_cal
    lback、submit_mic_rtp、_emit、invalidat
    e_stream、wait_quiescent。
    """

    asr_final: str

    mic_packets: list[bytes] = field(default_factory=list)

    answers: list[str] = field(default_factory=list)

    _output: Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]] | None = (
        None
    )

    _tasks: set[asyncio.Task[None]] = field(default_factory=set)

    def set_output_callback(
        self, callback: Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]]
    ) -> None:
        """函数契约说明.

        功能: 执行 set_output_callback
        的同步逻辑,并产出 _output。
        参数: self 表示当前实例。 callback:
        Callable[[StreamKey,
        CancellationEpoch, bytes],
        Awaitable[None]]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self._output = callback

    def submit_mic_rtp(
        self, stream: StreamKey, packet: bytes, epoch: CancellationEpoch
    ) -> None:
        """函数契约说明.

        功能: 执行 submit_mic_rtp 的同步逻辑,并协调
        append, create_task, add,
        add_done_callback。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 packet: bytes。
        必填。 epoch: CancellationEpoch。
        必填。
        契约: 同步调用。 返回 `None`。
        """

        self.mic_packets.append(packet)

        task = asyncio.create_task(self._emit(stream, epoch))

        self._tasks.add(task)

        task.add_done_callback(self._tasks.discard)

    async def _emit(self, stream: StreamKey, epoch: CancellationEpoch) -> None:
        """函数契约说明.

        功能: 执行 _emit 的异步逻辑,并协调
        OrchestratorTurnPipeline,
        accept_audience_input,
        process_next_turn, append。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 epoch:
        CancellationEpoch。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """

        if self.asr_final == "":
            return

        pipeline = OrchestratorTurnPipeline(
            adapters=PipelineAdapters(
                mode_policy=ModePolicy.onsite_explainer(),
                llm=MockLLMAdapter(answer_chunks=("onsite ", "answer")),
                retrieval=RetrievalFixtureProvider(refs=()),
            ),
            config=PipelineConfig(
                queue_capacity=1,
                turn_id_prefix="turn",
                segment_id_prefix="seg",
            ),
        )

        assert pipeline.accept_audience_input(
            ASRAudienceEvent(
                text=self.asr_final,
                received_at_ms=1_000,
                segment_id="asr-onsite-001",
                seq=1,
            )
        )

        turn = pipeline.process_next_turn()

        assert turn is not None

        self.answers.append(turn.answer_text)

        cues = pipeline.complete_synthesis(
            MockSynthesisResult(
                turn_id=turn.turn_id,
                segment_id=turn.segment_id,
                audio=AudioMetadata(16_000, 1, "pcm_s16le", 20, 640),
                expression="smile",
                action="speak",
                scene="onsite",
                slide_page=1,
            ),
            rtp_stream_start_ms=0,
            stream_id="onsite-answer-001",
        )

        assert cues is not None

        output = self._output

        if output is not None:
            await output(
                stream,
                epoch,
                _rtp_packet(ssrc=TTS_SSRC, payload=SYNTHESIZED_L16_PAYLOAD),
            )

    def invalidate_stream(
        self, stream: StreamKey, next_epoch: CancellationEpoch
    ) -> None:
        """函数契约说明.

        功能: 执行 invalidate_stream
        的同步逻辑,并协调 tuple, cancel。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 next_epoch:
        CancellationEpoch。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = (stream, next_epoch)

        for task in tuple(self._tasks):
            _ = task.cancel()

    async def wait_quiescent(self) -> None:
        """函数契约说明.

        功能: 执行 wait_quiescent 的异步逻辑,并协调
        gather。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """

        if self._tasks:
            _ = await asyncio.gather(*self._tasks, return_exceptions=True)


def test_onsite_mic_rtp_is_ingested_and_replaced_with_synthesized_sound_rtp() -> None:
    """函数契约说明.

    功能: 验证 onsite mic rtp is ingested
    and replaced with synthesized sound
    rtp 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_replacement_proof())


async def _replacement_proof() -> None:
    # Given: authenticated Mic and Sound routes plus an onsite ASR/LLM/TTS bridge.

    """函数契约说明.

    功能: 执行 _replacement_proof 的异步逻辑,并协调
    FakeDatagramTransport,
    FakeOnsiteExplainerBridge,
    _onsite_hub, register_control。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    transport = FakeDatagramTransport()

    bridge = FakeOnsiteExplainerBridge(asr_final="Explain BitNet")

    hub = _onsite_hub(transport, bridge)

    hub.register_control(_source_registration(), MIC_PEER[0])

    hub.register_control(_sink_registration(), SOUND_PEER[0])

    mic_packet = _rtp_packet(ssrc=MIC_SSRC, payload=b"\x01\x02" * 320)

    # When: Mic sends a canonical RTP frame from its pinned UDP endpoint.

    delivered = hub.route_datagram(mic_packet, MIC_PEER)

    await hub.wait_for_onsite_jobs()

    # Then: only bridge-produced canonical L16 reaches Sound, never the Mic bytes.

    expected_packet = _rtp_packet(ssrc=TTS_SSRC, payload=SYNTHESIZED_L16_PAYLOAD)

    assert delivered is False

    assert bridge.mic_packets == [mic_packet]

    assert bridge.answers == ["onsite answer"]

    assert transport.sent == [(expected_packet, (SOUND_PEER[0], 5006))]

    assert transport.sent[0][0] != mic_packet

    assert transport.sent[0][0][:2] == b"\x80\x60"


def test_onsite_blank_asr_final_emits_no_sound_rtp() -> None:
    """函数契约说明.

    功能: 验证 onsite blank asr final emits
    no sound rtp 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_blank_asr_proof())


async def _blank_asr_proof() -> None:
    # Given: authenticated Mic and Sound routes with a blank ASR final.

    """函数契约说明.

    功能: 执行 _blank_asr_proof 的异步逻辑,并协调
    FakeDatagramTransport,
    FakeOnsiteExplainerBridge,
    _onsite_hub, register_control。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    transport = FakeDatagramTransport()

    bridge = FakeOnsiteExplainerBridge(asr_final="")

    hub = _onsite_hub(transport, bridge)

    hub.register_control(_source_registration(), MIC_PEER[0])

    hub.register_control(_sink_registration(), SOUND_PEER[0])

    mic_packet = _rtp_packet(ssrc=MIC_SSRC, payload=b"\x00\x00" * 320)

    # When: Mic sends the frame that produced no final ASR text.

    delivered = hub.route_datagram(mic_packet, MIC_PEER)

    await hub.wait_for_onsite_jobs()

    # Then: the frame is ingested but neither raw nor synthesized RTP reaches Sound.

    assert delivered is False

    assert bridge.mic_packets == [mic_packet]

    assert bridge.answers == []

    assert transport.sent == []


def _source_registration() -> str:
    """函数契约说明.

    功能: 执行 _source_registration
    的同步逻辑,并协调 _envelope, _codec,
    _endpoint。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `str`。
    """

    return _envelope(
        "media.rtp.source.register",
        "mic",
        {
            "stream_id": STREAM_ID,
            "ssrc": MIC_SSRC,
            "codec": _codec(),
            "rtp_endpoint": _endpoint(5004),
        },
    )


def _onsite_hub(
    transport: FakeDatagramTransport,
    bridge: FakeOnsiteExplainerBridge,
) -> RtpHub:
    """函数契约说明.

    功能: 执行 _onsite_hub 的同步逻辑,并协调 RtpHub。
    参数: transport:
    FakeDatagramTransport。 必填。 bridge:
    FakeOnsiteExplainerBridge。 必填。
    契约: 同步调用。 返回 `RtpHub`。
    """

    bridge_options = {"onsite_bridge": bridge}

    return RtpHub(transport, **bridge_options)


def _sink_registration() -> str:
    """函数契约说明.

    功能: 执行 _sink_registration 的同步逻辑,并协调
    _envelope, _codec, _endpoint。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `str`。
    """

    return _envelope(
        "media.rtp.sink.register",
        "sound",
        {"stream_id": STREAM_ID, "codec": _codec(), "rtp_endpoint": _endpoint(5006)},
    )


def _envelope(event_type: str, source: str, data: dict[str, JsonValue]) -> str:
    """函数契约说明.

    功能: 执行 _envelope 的同步逻辑,并协调 dumps。
    参数: event_type: str。 必填。 source:
    str。 必填。 data: dict[str, JsonValue]。
    必填。
    契约: 同步调用。 返回 `str`。
    """

    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": event_type,
            "event_id": f"event-{event_type}",
            "source": source,
            "time": "2026-07-08T00:00:00Z",
            "trace_id": "trace-onsite-001",
            "session_id": SESSION_ID,
            "seq": 1,
            "data": data,
        }
    )


def _codec() -> dict[str, JsonValue]:
    """函数契约说明.

    功能: 执行 _codec 的同步逻辑,并维持签名契约。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """

    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _endpoint(port: int) -> dict[str, JsonValue]:
    """函数契约说明.

    功能: 执行 _endpoint 的同步逻辑,并维持签名契约。
    参数: port: int。 必填。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """

    return {"host": "declared.example.test", "port": port}


def _rtp_packet(*, ssrc: int, payload: bytes) -> bytes:
    """函数契约说明.

    功能: 执行 _rtp_packet 的同步逻辑,并协调
    to_bytes。
    参数: ssrc: int。 必填。 payload: bytes。
    必填。
    契约: 同步调用。 返回 `bytes`。
    """

    return b"\x80\x60\x00\x01\x00\x00\x00\x01" + ssrc.to_bytes(4, "big") + payload
