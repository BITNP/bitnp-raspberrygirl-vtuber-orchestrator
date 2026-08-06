from __future__ import annotations

import asyncio
import io
import json
import threading
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.ids import ConnectionId, SessionId
from orchestrator.media_adapters import SynthesizedAudio
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.response_coordinator import run_blocking_provider
from orchestrator.scheduler_reflex import SchedulerOutputFence
from orchestrator.sessions import SessionScheduler
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    StreamFlush,
    StreamKey,
)
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_hub import RtpHub
from orchestrator.transport_runtime import ControlHandler, TransportRuntime
from orchestrator.tts_rtp import Pcm16leChunk

if TYPE_CHECKING:
    from collections.abc import Iterator

    from orchestrator.json_boundary import JsonValue
    from orchestrator.provider_streaming import ProviderCancellationHandle


@dataclass(slots=True)
class _Datagrams:
    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:

        self.sent.append((data, addr))

    def close(self) -> None:

        return


@dataclass(frozen=True, slots=True)
class _ControlServer:
    def close(self) -> None:

        return

    async def wait_closed(self) -> None:

        return


@dataclass(slots=True)
class _DatagramListener:
    transport: _Datagrams

    hub: RtpHub | None = None

    async def listen(self, _host: str, _port: int, hub: RtpHub) -> _Datagrams:

        self.hub = hub

        return self.transport


@dataclass(slots=True)
class _DelayedAsr:
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

        _ = (audio, filename)

        if cancellation is not None:
            _ = cancellation.bind(self._cancel)

        _ = self.started.set()

        _ = self.release.wait()

        _ = self.completed.set()

        return ASRAudienceEvent("Explain BitNet", received_at_ms, segment_id, seq)

    def _cancel(self) -> None:

        _ = self.cancelled.set()

        _ = self.release.set()


@dataclass(frozen=True, slots=True)
class _Tts:
    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:

        _ = (text, voice, ref_audio, ref_text, cancellation)

        return SynthesizedAudio(_wav(b"\x10\x20" * 320), "audio/wav")


@dataclass(slots=True)
class _StreamingTts:
    first_chunk_ready: threading.Event = field(default_factory=threading.Event)
    release_second_chunk: threading.Event = field(default_factory=threading.Event)
    third_chunk_requested: threading.Event = field(default_factory=threading.Event)

    capability: str = "streaming_sse"

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[Pcm16leChunk]:
        _ = (text, voice, ref_audio, ref_text, cancellation)
        self.first_chunk_ready.set()
        yield Pcm16leChunk(b"\x10\x20" * 320)
        _ = self.release_second_chunk.wait(timeout=1.0)
        yield Pcm16leChunk(b"\x30\x40" * 15_680)
        self.third_chunk_requested.set()
        yield Pcm16leChunk(b"\x50\x60" * 320)

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:
        _ = (text, voice, ref_audio, ref_text, cancellation)
        message = "streaming_sse must not fall back to full-clip synthesis"
        raise AssertionError(message)


@dataclass(frozen=True, slots=True)
class _ShortStreamingTts:
    capability: str = "streaming_sse"

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[Pcm16leChunk]:
        _ = (text, voice, ref_audio, ref_text, cancellation)
        yield Pcm16leChunk(b"\x10\x20" * 320)

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:
        _ = (text, voice, ref_audio, ref_text, cancellation)
        raise AssertionError


@dataclass(slots=True)
class _CancellableStreamingTts:
    next_started: threading.Event = field(default_factory=threading.Event)
    release_next: threading.Event = field(default_factory=threading.Event)

    capability: str = "streaming_sse"

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[Pcm16leChunk]:
        _ = (text, voice, ref_audio, ref_text)
        release = (
            (lambda: None)
            if cancellation is None
            else cancellation.bind(self.release_next.set)
        )
        try:
            self.next_started.set()
            _ = self.release_next.wait(timeout=1.0)
            if cancellation is None or not cancellation.cancelled:
                yield Pcm16leChunk(b"\x10\x20" * 320)
        finally:
            release()

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:
        _ = (text, voice, ref_audio, ref_text, cancellation)
        raise AssertionError


def test_runtime_rejects_mic_rtp_without_starting_a_provider() -> None:

    asyncio.run(_cancellation_proof())


async def _cancellation_proof() -> None:
    # Given: a registered onsite route and an ASR provider that would expose any
    # accidental UDP Mic ingress.

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

    # When: an untrusted peer sends a packet shaped like Mic RTP.

    assert runtime.route_datagram(_rtp_packet(), ("127.0.0.1", 41_000)) is False
    await runtime.wait_for_onsite_jobs()
    await runtime.close()

    # Then: Mic has no RTP ingress route; it must send authenticated asr.final
    # control instead.  No provider or downstream Sound output is started.

    assert transport.sent == []
    assert asr.started.is_set() is False
    assert transport.sent == []


def test_runtime_close_after_rejected_mic_rtp_has_no_provider_to_cancel() -> None:

    asyncio.run(_close_before_release_proof())


async def _close_before_release_proof() -> None:
    # Given: an onsite route whose ASR provider is reachable only through
    # authenticated control events.

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

    # When: runtime shutdown follows rejected Mic RTP.

    await runtime.close()

    # Then: no UDP packet can start the provider or emit stale RTP.

    assert transport.sent == []
    assert asr.started.is_set() is False
    assert asr.cancelled.is_set() is False


def test_mic_source_disconnect_does_not_turn_rtp_into_an_asr_utterance() -> None:

    asyncio.run(_source_disconnect_finalizes_proof())


def test_response_tts_replaces_only_after_sound_flush_ack() -> None:

    asyncio.run(_response_output_epoch_proof())


def test_response_tts_waits_for_one_second_startup_watermark() -> None:

    asyncio.run(_response_streaming_proof())


def test_response_tts_plays_short_stream_when_it_ends_below_watermark() -> None:
    asyncio.run(_short_response_streaming_proof())


def test_response_tts_cancellation_waits_for_active_generator_next() -> None:
    asyncio.run(_streaming_cancellation_proof())


async def _streaming_cancellation_proof() -> None:
    bridge = _bridge(_DelayedAsr())
    tts = _CancellableStreamingTts()
    bridge.tts = tts
    task = asyncio.create_task(
        bridge.speak_response(
            StreamKey("session-cancel", "stream-cancel"),
            "agent reply",
            CancellationEpoch(0),
            "turn-cancel",
            lambda: True,
        )
    )

    _ = await run_blocking_provider(tts.next_started.wait)
    _ = task.cancel()
    try:
        _ = await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError


async def _response_streaming_proof() -> None:
    # Given: an SSE provider that deliberately holds its second audio chunk.

    bridge = _bridge(_DelayedAsr())
    tts = _StreamingTts()
    bridge.tts = tts
    packets: list[bytes] = []
    timeline: list[str] = []
    started = False
    output_ready = asyncio.Event()
    first_output_times: list[float] = []

    async def output(
        _stream: StreamKey, _epoch: CancellationEpoch, packet: bytes
    ) -> None:
        if not first_output_times:
            first_output_times.append(asyncio.get_running_loop().time())
        packets.append(packet)
        timeline.append("output")
        output_ready.set()

    def output_started() -> bool:
        nonlocal started
        timeline.append("started")
        started = True
        return True

    bridge.set_output_callback(output)
    task = asyncio.create_task(
        bridge.speak_response(
            StreamKey("session-streaming", "stream-streaming"),
            "agent reply",
            CancellationEpoch(0),
            "turn-streaming",
            output_started,
        )
    )

    # When: the first full 20 ms PCM frame has arrived but the SSE response is
    # still blocked before the chunk that reaches the one-second watermark.

    _ = await run_blocking_provider(tts.first_chunk_ready.wait)
    await asyncio.sleep(0.25)

    # Then: no output is committed from the undersized startup buffer.

    assert output_ready.is_set() is False
    assert started is False
    assert packets == []
    assert timeline == []
    assert task.done() is False

    tts.release_second_chunk.set()
    async with asyncio.timeout(1.0):
        _ = await output_ready.wait()

    # Once exactly one second is buffered, commit only after its first RTP
    # frame reaches the output adapter.

    assert started is True
    assert 1 <= len(packets) < 50
    assert timeline[:2] == ["output", "started"]
    assert task.done() is False
    assert await run_blocking_provider(tts.third_chunk_requested.wait, 0.2)

    assert await task is True
    finished_at = asyncio.get_running_loop().time()
    assert len(first_output_times) == 1
    assert finished_at - first_output_times[0] >= 0.95
    assert len(packets) == 51


async def _short_response_streaming_proof() -> None:
    bridge = _bridge(_DelayedAsr())
    bridge.tts = _ShortStreamingTts()
    packets: list[bytes] = []
    finished: list[CancellationEpoch] = []

    async def output(
        _stream: StreamKey, _epoch: CancellationEpoch, packet: bytes
    ) -> None:
        packets.append(packet)

    async def output_finished(
        _stream: StreamKey, epoch: CancellationEpoch
    ) -> None:
        finished.append(epoch)

    bridge.set_output_callback(output)
    bridge.set_output_finished_callback(output_finished)

    emitted = await bridge.speak_response(
        StreamKey("session-short", "stream-short"),
        "short reply",
        CancellationEpoch(0),
        "turn-short",
        lambda: True,
    )

    assert emitted is True
    assert len(packets) == 1
    assert finished == [CancellationEpoch(0)]


async def _response_output_epoch_proof() -> None:
    # Given: a completed output lease at generation zero while Mic input remains
    # at epoch zero for the next ASR final.

    transport = _Datagrams()
    bridge = _bridge(_DelayedAsr())
    hub = RtpHub(transport, onsite_bridge=bridge)
    scheduler = SessionScheduler(
        session_id=SessionId("session-onsite-runtime"),
        turn_id_prefix="turn-response-output",
    )
    fence = SchedulerOutputFence(scheduler)
    hub.set_output_fence(fence)
    command_epochs: list[int] = []

    async def record_command(_stream: StreamKey, epoch: int) -> None:
        command_epochs.append(epoch)

    hub.set_output_command_callback(record_command)

    async def acknowledge_flush(flush: StreamFlush) -> None:
        assert fence.acknowledge(FlushAcknowledgement.from_flush(flush))

    async def admit_flush(_flush: StreamFlush) -> bool:
        return True

    hub.set_replacement_callbacks(acknowledge_flush, admit_flush)
    hub.register_control(_source_registration(), "127.0.0.1")
    hub.register_control(_sink_registration(), "127.0.0.1")
    stream = StreamKey("session-onsite-runtime", "stream-onsite-runtime")

    assert hub.authorize_onsite_output(stream, CancellationEpoch(0)) is True
    turn_id = scheduler.snapshot.active_turn_id
    assert turn_id is not None
    revision = scheduler.snapshot.revision

    # When: a new accepted Brain reply is synthesized from the unchanged Mic
    # input epoch.

    emitted = await bridge.speak_response(
        stream, "agent reply", CancellationEpoch(0), turn_id, lambda: True
    )
    await asyncio.sleep(0)

    # Then: the old lease stays live until the flush callback acknowledges it;
    # this current turn then receives the next output epoch without a new turn.

    assert emitted is True
    assert scheduler.snapshot.revision == revision
    assert scheduler.snapshot.active_turn_id == turn_id
    assert command_epochs == [0]
    assert len(transport.sent) == 1
    packet, endpoint = transport.sent[0]
    assert endpoint == ("127.0.0.1", 5006)
    assert int.from_bytes(packet[8:12], "big") == hub.output_ssrc(stream, 1)


async def _source_disconnect_finalizes_proof() -> None:
    # Given: a non-legacy onsite route with a peer attempting forbidden RTP ingress.

    transport = _Datagrams()

    listener = _DatagramListener(transport)

    asr = _DelayedAsr()

    runtime = TransportRuntime(
        _loopback_config(),
        datagram_listener=listener.listen,
        control_listener=_control_listener,
        onsite_bridge=_bridge(asr, legacy_keyed=False),
    )

    await runtime.start()

    assert listener.hub is not None

    mic_owner = ConnectionId("mic-control")

    listener.hub.register_control(_source_registration(), "127.0.0.1", mic_owner)

    listener.hub.register_control(_sink_registration(), "127.0.0.1")

    assert runtime.route_datagram(_rtp_packet(), ("127.0.0.1", 41_000)) is False

    # When: the registered Mic control connection closes.

    listener.hub.remove_connection(mic_owner)
    await runtime.wait_for_onsite_jobs()
    await runtime.close()

    # Then: UDP did not create an endpoint or provider work before disconnect.

    assert asr.started.is_set() is False
    assert asr.completed.is_set() is False
    assert transport.sent == []


def _source_registration() -> str:

    return _registration(
        "mic.input.register",
        "mic",
        {"stream_id": "stream-onsite-runtime"},
    )


def _sink_registration() -> str:

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

    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _rtp_packet() -> bytes:

    return b"\x80\x60\x00\x01\x00\x00\x00\x01\x10\x20\x30\x40" + (b"\x7f\xff" * 320)


def _bridge(asr: _DelayedAsr, *, legacy_keyed: bool = True) -> OnsiteExplainerBridge:
    _ = asr, legacy_keyed

    return OnsiteExplainerBridge(
        tts=_Tts(),
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
    )


def _wav(payload: bytes) -> bytes:

    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(2)

        audio.setframerate(16_000)

        audio.writeframes(payload)

    return output.getvalue()


def _loopback_config() -> TransportConfig:

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

    return _ControlServer()
