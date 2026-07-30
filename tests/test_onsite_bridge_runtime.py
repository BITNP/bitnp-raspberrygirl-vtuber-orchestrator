"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.llm import MockLLMAdapter
from orchestrator.media_adapters import SynthesizedAudio
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import ASRAudienceEvent, PipelineConfig
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_hub import RtpHub
from orchestrator.transport_runtime import ControlHandler, TransportRuntime

if TYPE_CHECKING:
    from orchestrator.json_boundary import JsonValue
    from orchestrator.provider_streaming import ProviderCancellationHandle
    from orchestrator.transport_hub import RtpHub


@dataclass(slots=True)
class _Datagrams:
    """类契约说明.

    职责: 保存 _Datagrams
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


@dataclass(frozen=True, slots=True)
class _ControlServer:
    """类契约说明.

    职责: 保存 _ControlServer
    不可变数据结构,用类型标注表达字段契约。
    契约: 方法: close、wait_closed。
    """

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return

    async def wait_closed(self) -> None:
        """函数契约说明.

        功能: 执行 wait_closed
        的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """

        return


@dataclass(slots=True)
class _DatagramListener:
    """类契约说明.

    职责: 保存 _DatagramListener
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: transport、hub。 方法: listen。
    """

    transport: _Datagrams

    hub: RtpHub | None = None

    async def listen(self, _host: str, _port: int, hub: RtpHub) -> _Datagrams:
        """函数契约说明.

        功能: 执行 listen 的异步逻辑,并产出 hub。
        参数: self 表示当前实例。 _host: str。 必填。
        _port: int。 必填。 hub: RtpHub。 必填。
        契约: 异步调用。 返回 `_Datagrams`。
        """

        self.hub = hub

        return self.transport


@dataclass(slots=True)
class _DelayedAsr:
    """类契约说明.

    职责: 保存 _DelayedAsr
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    started、release、completed、cancelled。
    方法: transcribe、_cancel。
    """

    started: threading.Event = field(default_factory=threading.Event)

    release: threading.Event = field(default_factory=threading.Event)

    completed: threading.Event = field(default_factory=threading.Event)

    cancelled: threading.Event = field(default_factory=threading.Event)

    def transcribe(  # noqa: PLR0913
        self,
        *,
        audio: bytes,
        filename: str,
        received_at_ms: int,
        segment_id: str,
        seq: int,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> ASRAudienceEvent:
        """函数契约说明.

        功能: 执行 transcribe 的同步逻辑,并协调 set,
        wait, ASRAudienceEvent, bind。
        参数: self 表示当前实例。 audio: bytes。
        必填。 filename: str。 必填。
        received_at_ms: int。 必填。
        segment_id: str。 必填。 seq: int。
        必填。 cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回 `ASRAudienceEvent`。
        """

        _ = (audio, filename)

        if cancellation is not None:
            _ = cancellation.bind(self._cancel)

        _ = self.started.set()

        _ = self.release.wait()

        _ = self.completed.set()

        return ASRAudienceEvent("Explain BitNet", received_at_ms, segment_id, seq)

    def _cancel(self) -> None:
        """函数契约说明.

        功能: 执行 _cancel 的同步逻辑,并协调 set。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        _ = self.cancelled.set()

        _ = self.release.set()


@dataclass(frozen=True, slots=True)
class _Tts:
    """类契约说明.

    职责: 保存 _Tts 不可变数据结构,用类型标注表达字段契约。
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

        功能: 执行 synthesize 的同步逻辑,并协调
        SynthesizedAudio, _wav。
        参数: self 表示当前实例。 text: str。 必填。
        voice: str。 必填。 ref_audio: str。
        必填。 ref_text: str。 必填。
        cancellation:
        ProviderCancellationHandle |
        None。 可省略。
        契约: 同步调用。 返回 `SynthesizedAudio`。
        """

        _ = (text, voice, ref_audio, ref_text, cancellation)

        return SynthesizedAudio(_wav(b"\x10\x20" * 320), "audio/wav")


def test_runtime_processes_cancellation_while_provider_runs_and_drops_stale_rtp() -> (
    None
):
    """函数契约说明.

    功能: 验证 runtime processes
    cancellation while provider runs and
    drops stale rtp 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_cancellation_proof())


async def _cancellation_proof() -> None:
    # Given: a registered onsite route whose provider work waits on an explicit signal.

    """函数契约说明.

    功能: 执行 _cancellation_proof 的异步逻辑,并协调
    _Datagrams, _DatagramListener,
    _DelayedAsr, TransportRuntime。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    transport = _Datagrams()

    listener = _DatagramListener(transport)

    asr = _DelayedAsr()

    runtime = TransportRuntime(
        _loopback_config(),
        datagram_listener=listener.listen,
        control_listener=_control_listener,
        onsite_bridge=_bridge(asr),
    )

    await runtime.start()

    assert listener.hub is not None

    listener.hub.register_control(_source_registration(), "127.0.0.1")

    listener.hub.register_control(_sink_registration(), "127.0.0.1")

    # When: Mic work begins, control cancels its route, then the provider completes.

    assert runtime.route_datagram(_rtp_packet(), ("127.0.0.1", 41_000)) is False

    try:
        _ = await asyncio.to_thread(asr.started.wait)

        await runtime.cancel_stream("session-onsite-runtime", "stream-onsite-runtime")

    finally:
        asr.release.set()

    await runtime.wait_for_onsite_jobs()

    await runtime.close()

    # Then: the callback released the loop and no stale RTP reached Sound.

    assert transport.sent == []


def test_runtime_close_cancels_blocking_asr_and_drops_its_late_output() -> None:
    """函数契约说明.

    功能: 验证 runtime close cancels
    blocking asr and drops its late
    output 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_close_before_release_proof())


async def _close_before_release_proof() -> None:
    # Given: an onsite ASR worker blocked until its provider resource is cancelled.

    """函数契约说明.

    功能: 执行 _close_before_release_proof
    的异步逻辑,并协调 _Datagrams,
    _DatagramListener, _DelayedAsr,
    TransportRuntime。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    transport = _Datagrams()

    listener = _DatagramListener(transport)

    asr = _DelayedAsr()

    runtime = TransportRuntime(
        _loopback_config(),
        datagram_listener=listener.listen,
        control_listener=_control_listener,
        onsite_bridge=_bridge(asr),
    )

    await runtime.start()

    assert listener.hub is not None

    listener.hub.register_control(_source_registration(), "127.0.0.1")

    listener.hub.register_control(_sink_registration(), "127.0.0.1")

    assert runtime.route_datagram(_rtp_packet(), ("127.0.0.1", 41_000)) is False

    _ = await asyncio.to_thread(asr.started.wait)

    # When: runtime shutdown invalidates the route before the blocking worker releases.

    await runtime.close()

    # Then: shutdown cancels the provider and its result cannot emit stale RTP.

    assert transport.sent == []

    _ = await asyncio.to_thread(asr.completed.wait)

    assert asr.cancelled.is_set()

    assert transport.sent == []


def _source_registration() -> str:
    """函数契约说明.

    功能: 执行 _source_registration
    的同步逻辑,并协调 _registration, _codec。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `str`。
    """

    return _registration(
        "media.rtp.source.register",
        "mic",
        {
            "stream_id": "stream-onsite-runtime",
            "ssrc": 0x10203040,
            "codec": _codec(),
            "rtp_endpoint": {"host": "127.0.0.1", "port": 5004},
        },
    )


def _sink_registration() -> str:
    """函数契约说明.

    功能: 执行 _sink_registration 的同步逻辑,并协调
    _registration, _codec。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `str`。
    """

    return _registration(
        "media.rtp.sink.register",
        "sound",
        {
            "stream_id": "stream-onsite-runtime",
            "codec": _codec(),
            "rtp_endpoint": {"host": "127.0.0.1", "port": 5006},
        },
    )


def _registration(event_type: str, source: str, data: dict[str, JsonValue]) -> str:
    """函数契约说明.

    功能: 执行 _registration 的同步逻辑,并协调
    dumps。
    参数: event_type: str。 必填。 source:
    str。 必填。 data: dict[str, JsonValue]。
    必填。
    契约: 同步调用。 返回 `str`。
    """

    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": event_type,
            "event_id": event_type,
            "source": source,
            "time": "2026-07-28T00:00:00Z",
            "trace_id": "trace-onsite-runtime",
            "session_id": "session-onsite-runtime",
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


def _rtp_packet() -> bytes:
    """函数契约说明.

    功能: 执行 _rtp_packet 的同步逻辑,并维持签名契约。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `bytes`。
    """

    return b"\x80\x60\x00\x01\x00\x00\x00\x01\x10\x20\x30\x40" + (b"\x7f\xff" * 320)


def _bridge(asr: _DelayedAsr) -> OnsiteExplainerBridge:
    """函数契约说明.

    功能: 执行 _bridge 的同步逻辑,并协调
    OnsiteExplainerBridge, _Tts,
    OrchestratorTurnPipeline,
    PipelineAdapters。
    参数: asr: _DelayedAsr。 必填。
    契约: 同步调用。 返回
    `OnsiteExplainerBridge`。
    """

    return OnsiteExplainerBridge(
        asr=asr,
        tts=_Tts(),
        pipeline_factory=lambda: OrchestratorTurnPipeline(
            adapters=PipelineAdapters(
                mode_policy=AdaptiveAgentPolicy(),
                llm=MockLLMAdapter(("onsite answer",)),
                retrieval=RetrievalFixtureProvider(()),
            ),
            config=PipelineConfig(1, "turn-onsite", "segment-onsite"),
        ),
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
        frames_per_utterance=1,
        legacy_keyed_frames_per_utterance=1,
    )


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


def _loopback_config() -> TransportConfig:
    """函数契约说明.

    功能: 执行 _loopback_config 的同步逻辑,并协调
    TransportConfig。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `TransportConfig`。
    """

    return TransportConfig(
        "127.0.0.1",
        8765,
        "127.0.0.1",
        5004,
        "127.0.0.1",
        8765,
        5004,
        "ws",
        None,
        None,
        None,
    )


async def _control_listener(
    _config: TransportConfig, _handler: ControlHandler
) -> _ControlServer:
    """函数契约说明.

    功能: 执行 _control_listener 的异步逻辑,并协调
    _ControlServer。
    参数: _config: TransportConfig。 必填。
    _handler: ControlHandler。 必填。
    契约: 异步调用。 返回 `_ControlServer`。
    """

    return _ControlServer()
