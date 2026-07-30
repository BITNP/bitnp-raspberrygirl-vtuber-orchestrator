"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mic.stream_control import ControlContext, WebSocketStreamingControl
from mic.streaming import (
    AsyncioUdpSender,
    StreamingRuntimeConfig,
    StreamResources,
    StreamRuntime,
    load_streaming_runtime_config,
)
from sound.receive import AsyncioUdpBinder, ReceiveRuntime, WebsocketsControlConnector
from sound.receive_config import SoundReceiveConfig, load_runtime_config

from orchestrator.transport_config import (
    TransportConfig,
    load_transport_config_from_env,
)
from orchestrator.transport_runtime import TransportRuntime

if TYPE_CHECKING:
    from sound.rtp_playback import L16PlaybackFrame


@dataclass(slots=True)
class FakeCapture:
    """类契约说明.

    职责: 保存 FakeCapture
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: first_block、closed、reads。
    方法: open、read_block、aclose。
    """

    first_block: bytes

    closed: bool = False

    reads: int = 0

    async def open(self) -> None:
        """函数契约说明.

        功能: 执行 open 的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """

        return

    async def read_block(self) -> bytes | None:
        """函数契约说明.

        功能: 执行 read_block 的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `bytes | None`。
        """

        self.reads += 1

        if self.reads == 1:
            return self.first_block

        return None

    async def aclose(self) -> None:
        """函数契约说明.

        功能: 执行 aclose 的异步逻辑,并产出 closed。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """

        self.closed = True


@dataclass(slots=True)
class FakePortAudio:
    """类契约说明.

    职责: 保存 FakePortAudio
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: frames、frame_written、closed。
    方法: write、close_stream、close。
    """

    frames: list[L16PlaybackFrame] = field(default_factory=list)

    frame_written: asyncio.Event = field(default_factory=asyncio.Event)

    closed: bool = False

    def write(self, frame: L16PlaybackFrame) -> None:
        """函数契约说明.

        功能: 执行 write 的同步逻辑,并协调 append,
        set。
        参数: self 表示当前实例。 frame:
        L16PlaybackFrame。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.frames.append(frame)

        self.frame_written.set()

    def close_stream(self, stream_id: str) -> None:
        """函数契约说明.

        功能: 执行 close_stream 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 stream_id: str。
        必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = stream_id

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并产出 closed。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        self.closed = True


def test_mic_orchestrator_sound_loopback_forwards_rejects_cancels_and_closes() -> None:
    """函数契约说明.

    功能: 验证 mic orchestrator sound
    loopback forwards rejects cancels
    and closes 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_run_loopback_proof())


async def _run_loopback_proof() -> None:
    """函数契约说明.

    功能: 执行 _run_loopback_proof 的异步逻辑,并协调
    TransportRuntime, FakeCapture,
    FakePortAudio, _orchestrator_config。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    control_port = 37_561

    rtp_port = 37_562

    mic_port = 37_563

    sound_port = 37_564

    runtime = TransportRuntime(_orchestrator_config(control_port, rtp_port))

    capture = FakeCapture(b"\x01\x02" * 320)

    playback = FakePortAudio()

    sound_task: asyncio.Task[None] | None = None

    mic_task: asyncio.Task[None] | None = None

    try:
        await runtime.start()

        sound_task = asyncio.create_task(
            ReceiveRuntime(
                config=_sound_config(control_port, sound_port),
                udp_binder=AsyncioUdpBinder(),
                control_connector=WebsocketsControlConnector(),
                playback_sink=playback,
            ).run()
        )

        mic_config = _mic_config(control_port, rtp_port, mic_port)

        service_config = mic_config.service_config

        assert service_config is not None

        mic_task = asyncio.create_task(
            StreamRuntime(
                config=mic_config,
                resources=StreamResources(
                    capture=capture,
                    control=await WebSocketStreamingControl.open(
                        service_config,
                        ControlContext("trace-transport", "session-transport"),
                    ),
                    udp=AsyncioUdpSender(),
                ),
            ).run()
        )

        _ = await asyncio.wait_for(playback.frame_written.wait(), timeout=2)

        assert len(playback.frames) == 1

        await asyncio.wait_for(mic_task, timeout=2)

        playback.frame_written.clear()

        _send_rtp(rtp_port, _rtp_packet(0x4D494331))

        await _assert_no_additional_frame(playback)

        playback.frame_written.clear()

        _send_rtp(rtp_port, _rtp_packet(0x4D494332), source_port=mic_port)

        await _assert_no_additional_frame(playback)

        await runtime.cancel_stream("session-transport", "transport-stream")

        playback.frame_written.clear()

        _send_rtp(rtp_port, _rtp_packet(0x4D494331), source_port=mic_port)

        await _assert_no_additional_frame(playback)

        assert len(playback.frames) == 1

    finally:
        await _cancel(sound_task)

        await _cancel(mic_task)

        await runtime.close()

    assert capture.closed is True

    assert playback.closed is True

    assert runtime.readiness().ready is False


def _orchestrator_config(control_port: int, rtp_port: int) -> TransportConfig:
    """函数契约说明.

    功能: 执行 _orchestrator_config
    的同步逻辑,并协调
    load_transport_config_from_env, str。
    参数: control_port: int。 必填。 rtp_port:
    int。 必填。
    契约: 同步调用。 返回 `TransportConfig`。
    """

    return load_transport_config_from_env(
        {
            "ORCHESTRATOR_CONTROL_BIND_HOST": "127.0.0.1",
            "ORCHESTRATOR_CONTROL_BIND_PORT": str(control_port),
            "ORCHESTRATOR_RTP_BIND_HOST": "127.0.0.1",
            "ORCHESTRATOR_RTP_BIND_PORT": str(rtp_port),
            "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST": "127.0.0.1",
            "ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT": str(control_port),
            "ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT": str(rtp_port),
            "ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS": "true",
        }
    )


def _mic_config(
    control_port: int, rtp_port: int, mic_port: int
) -> StreamingRuntimeConfig:
    """函数契约说明.

    功能: 执行 _mic_config 的同步逻辑,并协调
    load_streaming_runtime_config, str。
    参数: control_port: int。 必填。 rtp_port:
    int。 必填。 mic_port: int。 必填。
    契约: 同步调用。 返回
    `StreamingRuntimeConfig`。
    """

    return load_streaming_runtime_config(
        {
            "ORCHESTRATOR_WS_URL": f"ws://127.0.0.1:{control_port}",
            "ORCHESTRATOR_RTP_HOST": "127.0.0.1",
            "ORCHESTRATOR_RTP_PORT": str(rtp_port),
            "MIC_RTP_BIND_HOST": "127.0.0.1",
            "MIC_RTP_BIND_PORT": str(mic_port),
            "MIC_ALLOW_LOOPBACK_WS": "true",
            "MIC_MAX_CAPTURE_BLOCKS": "1",
            "BITNP_MIC_RTP_STREAM_ID": "transport-stream",
            "BITNP_MIC_RTP_TIMESTAMP": "96000",
            "BITNP_TRACE_ID": "trace-transport",
            "BITNP_SESSION_ID": "session-transport",
        }
    )


def _sound_config(control_port: int, sound_port: int) -> SoundReceiveConfig:
    """函数契约说明.

    功能: 执行 _sound_config 的同步逻辑,并协调
    load_runtime_config, str。
    参数: control_port: int。 必填。
    sound_port: int。 必填。
    契约: 同步调用。 返回 `SoundReceiveConfig`。
    """

    return load_runtime_config(
        {
            "ORCHESTRATOR_WS_URL": f"ws://127.0.0.1:{control_port}",
            "SOUND_ALLOW_LOOPBACK_WS": "true",
            "SOUND_RTP_STREAM_ID": "transport-stream",
            "SOUND_RTP_BIND_HOST": "127.0.0.1",
            "SOUND_RTP_BIND_PORT": str(sound_port),
            "SOUND_RTP_ADVERTISED_HOST": "127.0.0.1",
            "SOUND_TRACE_ID": "trace-transport",
            "SOUND_SESSION_ID": "session-transport",
        }
    )


def _rtp_packet(ssrc: int) -> bytes:
    """函数契约说明.

    功能: 执行 _rtp_packet 的同步逻辑,并协调
    to_bytes。
    参数: ssrc: int。 必填。
    契约: 同步调用。 返回 `bytes`。
    """

    return (
        b"\x80\x60\x00\x00\x00\x01\x77\x00"
        + ssrc.to_bytes(4, "big")
        + (b"\x00\x01" * 320)
    )


def _send_rtp(rtp_port: int, packet: bytes, source_port: int | None = None) -> None:
    """函数契约说明.

    功能: 执行 _send_rtp 的同步逻辑,并协调 socket,
    sendto, bind。
    参数: rtp_port: int。 必填。 packet:
    bytes。 必填。 source_port: int | None。
    可省略。
    契约: 同步调用。 返回 `None`。
    """

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        if source_port is not None:
            sender.bind(("127.0.0.1", source_port))

        _ = sender.sendto(packet, ("127.0.0.1", rtp_port))


async def _assert_no_additional_frame(playback: FakePortAudio) -> None:
    """函数契约说明.

    功能: 执行 _assert_no_additional_frame
    的异步逻辑,并协调 len, wait_for, wait。
    参数: playback: FakePortAudio。 必填。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    initial_count = len(playback.frames)

    try:
        _ = await asyncio.wait_for(playback.frame_written.wait(), timeout=0.1)

    except TimeoutError:
        return

    assert len(playback.frames) == initial_count


async def _cancel(task: asyncio.Task[None] | None) -> None:
    """函数契约说明.

    功能: 执行 _cancel 的异步逻辑,并协调 cancel,
    done。
    参数: task: asyncio.Task[None] | None。
    必填。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    if task is None or task.done():
        return

    _ = task.cancel()

    try:
        await task

    except asyncio.CancelledError:
        return
