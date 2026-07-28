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
    first_block: bytes
    closed: bool = False
    reads: int = 0

    async def open(self) -> None:
        return None

    async def read_block(self) -> bytes | None:
        self.reads += 1
        if self.reads == 1:
            return self.first_block
        return None

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)
class FakePortAudio:
    frames: list[L16PlaybackFrame] = field(default_factory=list)
    frame_written: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False

    def write(self, frame: L16PlaybackFrame) -> None:
        self.frames.append(frame)
        self.frame_written.set()

    def close_stream(self, stream_id: str) -> None:
        _ = stream_id

    def close(self) -> None:
        self.closed = True


def test_mic_orchestrator_sound_loopback_forwards_rejects_cancels_and_closes() -> None:
    asyncio.run(_run_loopback_proof())


async def _run_loopback_proof() -> None:
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
    return (
        b"\x80\x60\x00\x00\x00\x01\x77\x00"
        + ssrc.to_bytes(4, "big")
        + (b"\x00\x01" * 320)
    )


def _send_rtp(rtp_port: int, packet: bytes, source_port: int | None = None) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        if source_port is not None:
            sender.bind(("127.0.0.1", source_port))
        _ = sender.sendto(packet, ("127.0.0.1", rtp_port))


async def _assert_no_additional_frame(playback: FakePortAudio) -> None:
    initial_count = len(playback.frames)
    try:
        _ = await asyncio.wait_for(playback.frame_written.wait(), timeout=0.1)
    except TimeoutError:
        return
    assert len(playback.frames) == initial_count


async def _cancel(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    _ = task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return
