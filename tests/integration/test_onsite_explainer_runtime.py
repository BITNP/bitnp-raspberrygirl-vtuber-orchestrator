
from __future__ import annotations

import asyncio
import io
import json
import wave
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import ssl
    from collections.abc import Callable

    from orchestrator.json_boundary import JsonValue
    from orchestrator.provider_streaming import ProviderCancellationHandle


from sound.receive import ReceiveRuntime
from sound.receive_config import SoundReceiveConfig
from sound.rtp_playback import L16PlaybackFrame, StreamId

from orchestrator.ids import SessionId
from orchestrator.llm import MockLLMAdapter
from orchestrator.media_adapters import SynthesizedAudio
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import ASRAudienceEvent, PipelineConfig
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.scheduler_reflex import SchedulerOutputFence
from orchestrator.sessions import SessionScheduler
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    FlushRequestId,
    GeneratedSsrc,
    SegmentId,
    StreamKey,
)
from orchestrator.transport_hub import RtpHub
from orchestrator.tts_rtp import Pcm16leChunk

SESSION_ID: Final = "session-onsite-runtime-001"

STREAM_ID: Final = "mic-onsite-runtime-001"

MIC_PEER: Final = ("192.0.2.40", 42_040)

SOUND_PEER: Final = ("192.0.2.41", 42_041)

MIC_SSRC: Final = 0x1020_3040

SOUND_BIND_HOST: Final = "127.0.0.1"


@dataclass(slots=True)
class _Datagrams:

    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:

        self.sent.append((data, addr))

    def close(self) -> None:

        return


@dataclass(slots=True)
class _SoundBinding:

    handler: Callable[[bytes], None] | None = None

    @property
    def port(self) -> int:

        return 50_006

    def set_packet_handler(self, handler: Callable[[bytes], None]) -> None:

        self.handler = handler

    def deliver(self, packet: bytes) -> None:

        assert self.handler is not None

        self.handler(packet)

    def close(self) -> None:

        return


@dataclass(frozen=True, slots=True)
class _SoundBinder:

    binding: _SoundBinding

    async def bind(self, host: str, port: int) -> _SoundBinding:

        assert (host, port) == (SOUND_BIND_HOST, 50_006)

        return self.binding


@dataclass(slots=True)
class _SoundControl:

    command: str

    generated_packet: bytes

    binding: _SoundBinding

    delivered: bool = False

    async def send(self, message: str) -> None:

        _ = message

    async def recv(self) -> str | None:

        if not self.delivered:
            self.delivered = True

            return self.command

        self.binding.deliver(self.generated_packet)

        return None

    async def close(self) -> None:

        return


@dataclass(frozen=True, slots=True)
class _SoundConnector:

    control: _SoundControl

    async def connect(
        self, url: str, headers: dict[str, str], ssl_context: ssl.SSLContext | None
    ) -> _SoundControl:

        assert url == "wss://orchestrator.example.test/control"

        assert headers == {}

        assert ssl_context is None

        return self.control


@dataclass(slots=True)
class _SoundSink:

    frames: list[L16PlaybackFrame] = field(default_factory=list)

    def write(self, frame: L16PlaybackFrame) -> None:

        self.frames.append(frame)

    def close_stream(self, stream_id: str) -> None:

        _ = stream_id

    def close(self) -> None:

        return


@dataclass(frozen=True, slots=True)
class _Asr:

    text: str

    expected_audio_bytes: int = 1_280

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

        _ = cancellation

        assert filename == "onsite-l16.wav"

        assert len(_wav_payload(audio)) == self.expected_audio_bytes

        return ASRAudienceEvent(self.text, received_at_ms, segment_id, seq)


@dataclass(frozen=True, slots=True)
class _Tts:

    def stream_pcm16le(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> tuple[Pcm16leChunk, ...]:

        _ = cancellation

        assert text == "onsite answer"

        assert voice == "raspberry"

        assert ref_audio == "file:///voice.wav"

        assert ref_text == "reference"

        return (Pcm16leChunk(b"\x10\x20" * 319 + b"\x10"), Pcm16leChunk(b"\x20"))

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:

        _ = cancellation

        assert text == "onsite answer"

        assert voice == "raspberry"

        assert ref_audio == "file:///voice.wav"

        assert ref_text == "reference"

        return SynthesizedAudio(_wav(b"\x10\x20" * 320), "audio/wav")


def test_onsite_runtime_composes_asr_pipeline_tts_and_replaces_mic_rtp() -> None:

    asyncio.run(_runtime_composition_proof())


def test_onsite_runtime_baseline_preserves_current_generated_output() -> None:

    asyncio.run(_runtime_composition_proof())


def test_onsite_runtime_scheduler_fences_barge_in_until_exact_flush_ack() -> None:

    asyncio.run(_scheduler_barge_in_proof())


async def _scheduler_barge_in_proof() -> None:
    # Given: generated output for a scheduler-owned active turn on a live route.


    transport = _Datagrams()

    scheduler = SessionScheduler(
        session_id=SessionId(SESSION_ID), turn_id_prefix="turn-onsite-reflex"
    )

    fence = SchedulerOutputFence(scheduler)

    hub = RtpHub(transport)

    hub.set_output_fence(fence)

    hub.register_control(
        _registration("media.rtp.source.register", "mic", _source()), MIC_PEER[0]
    )

    hub.register_control(
        _registration("media.rtp.sink.register", "sound", _sink()), SOUND_PEER[0]
    )

    stream = StreamKey(SESSION_ID, STREAM_ID)

    correlation = hub.correlation(stream)

    assert correlation is not None

    first = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-first"),
        target_generated_ssrc=GeneratedSsrc(0x1111_1111),
        correlation=correlation,
    )

    old_packet = _rtp(int(first.target_generated_ssrc), b"\x10\x20" * 320)

    await hub.deliver_generated_rtp(stream, first.cancellation_epoch, old_packet)

    # When: a later meaningful utterance interrupts the active output.

    replacement, flush = fence.interrupt(
        stream=stream,
        segment_id=SegmentId("segment-replacement"),
        correlation=correlation,
    )

    replacement_packet = _rtp(int(replacement.target_generated_ssrc), b"\x30\x40" * 320)

    await hub.deliver_generated_rtp(stream, first.cancellation_epoch, old_packet)

    await hub.deliver_generated_rtp(
        stream, replacement.cancellation_epoch, replacement_packet
    )

    # Then: old and pre-ack replacement RTP are fenced until Sound confirms this flush.

    assert transport.sent == [(old_packet, (SOUND_PEER[0], 5006))]

    acknowledgement = FlushAcknowledgement.from_flush(flush)

    assert (
        fence.acknowledge(
            replace(acknowledgement, request_id=FlushRequestId("stale-flush"))
        )
        is False
    )

    assert fence.can_emit(stream, replacement.cancellation_epoch) is False

    assert fence.acknowledge(FlushAcknowledgement.from_flush(flush)) is True

    assert fence.acknowledge(FlushAcknowledgement.from_flush(flush)) is False

    await hub.deliver_generated_rtp(
        stream, replacement.cancellation_epoch, replacement_packet
    )

    assert transport.sent == [
        (old_packet, (SOUND_PEER[0], 5006)),
        (replacement_packet, (SOUND_PEER[0], 5006)),
    ]


def test_generated_rtp_rejects_epoch_41_after_epoch_42_is_active() -> None:
    asyncio.run(_stale_epoch_gate_proof())


async def _stale_epoch_gate_proof() -> None:
    # Given: a registered generated-audio route whose current cancellation epoch is 42.
    transport = _Datagrams()
    hub = RtpHub(transport)
    hub.register_control(
        _registration("media.rtp.source.register", "mic", _source()), MIC_PEER[0]
    )
    hub.register_control(
        _registration("media.rtp.sink.register", "sound", _sink()), SOUND_PEER[0]
    )
    stream = StreamKey(SESSION_ID, STREAM_ID)
    correlation = hub.correlation(stream)
    assert correlation is not None
    scheduler = SessionScheduler(
        session_id=SessionId(SESSION_ID), turn_id_prefix="turn-stale-epoch"
    )
    fence = SchedulerOutputFence(scheduler)
    hub.set_output_fence(fence)
    for epoch in range(41):
        _ = fence.activate(
            stream=stream,
            segment_id=SegmentId(f"segment-{epoch}"),
            target_generated_ssrc=GeneratedSsrc(0x4242_4242),
            correlation=correlation,
        )
    active = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-41"),
        target_generated_ssrc=GeneratedSsrc(0x4141_4141),
        correlation=correlation,
    )
    assert active.cancellation_epoch == CancellationEpoch(41)
    stale_packet = _rtp(0x4141_4141, b"\x10\x20" * 320)
    active_packet = _rtp(0x4242_4242, b"\x30\x40" * 320)

    await hub.deliver_generated_rtp(stream, active.cancellation_epoch, stale_packet)
    replacement, flush = fence.interrupt(
        stream=stream,
        segment_id=SegmentId("segment-42"),
        correlation=correlation,
    )

    # When: a retired packet arrives after a newer generated-audio epoch is active.
    await hub.deliver_generated_rtp(stream, CancellationEpoch(41), stale_packet)
    await hub.deliver_generated_rtp(
        stream, replacement.cancellation_epoch, active_packet
    )

    assert fence.acknowledge(FlushAcknowledgement.from_flush(flush)) is True
    await hub.deliver_generated_rtp(stream, CancellationEpoch(41), stale_packet)
    await hub.deliver_generated_rtp(
        stream, replacement.cancellation_epoch, active_packet
    )

    # Then: stale audio remains gated and cannot resume after epoch 42.
    assert transport.sent == [
        (stale_packet, (SOUND_PEER[0], 5006)),
        (active_packet, (SOUND_PEER[0], 5006)),
    ]


async def _runtime_composition_proof() -> None:
    # Given: the real bridge with deterministic provider boundaries and pinned routes.


    transport = _Datagrams()

    hub = RtpHub(transport, onsite_bridge=_bridge("Explain BitNet", 19_840))

    hub.register_control(
        _registration("media.rtp.source.register", "mic", _source()), MIC_PEER[0]
    )

    hub.register_control(
        _registration("media.rtp.sink.register", "sound", _sink()), SOUND_PEER[0]
    )

    speech_packet = _rtp(MIC_SSRC, b"\x03\xe8" * 320, 1, 320)

    # When: speech is followed by the 600 ms silence endpoint.

    first_delivered = hub.route_datagram(speech_packet, MIC_PEER)

    delivered = all(
        hub.route_datagram(
            _rtp(MIC_SSRC, b"\x00\x00" * 320, sequence, sequence * 320),
            MIC_PEER,
        )
        is False
        for sequence in range(2, 32)
    )

    await hub.wait_for_onsite_jobs()

    # Then: Sound receives generated canonical RTP, never the microphone packet.

    assert first_delivered is False

    assert delivered is True

    assert len(transport.sent) == 1

    generated, endpoint = transport.sent[0]

    assert endpoint == (SOUND_PEER[0], 5006)

    assert generated != speech_packet

    assert generated[:2] == b"\x80\x60"

    assert int.from_bytes(generated[8:12]) != MIC_SSRC

    assert generated[12:] == b"\x20\x10" * 320


def test_onsite_runtime_drops_blank_asr_without_raw_rtp() -> None:

    asyncio.run(_blank_asr_proof())


async def _blank_asr_proof() -> None:
    # Given: a real bridge whose ASR boundary emits a blank final.


    transport = _Datagrams()

    hub = RtpHub(transport, onsite_bridge=_bridge(""))

    hub.register_control(
        _registration("media.rtp.source.register", "mic", _source()), MIC_PEER[0]
    )

    hub.register_control(
        _registration("media.rtp.sink.register", "sound", _sink()), SOUND_PEER[0]
    )

    # When: the pinned Mic sends its canonical frame.

    delivered = hub.route_datagram(_rtp(MIC_SSRC, b"\x00\x00" * 320), MIC_PEER)

    await hub.wait_for_onsite_jobs()

    # Then: no raw or generated packet is routed to Sound.

    assert delivered is False

    assert transport.sent == []


def test_generated_onsite_command_ssrc_unlocks_real_sound_runtime_playback() -> None:

    asyncio.run(_sound_runtime_playback_proof())


async def _sound_runtime_playback_proof() -> None:
    # Given: an onsite-generated RTP packet and a Sound command bound to its SSRC.


    mic_packet = _rtp(MIC_SSRC, b"\x01\x02" * 320)

    bridge = _bridge("Explain BitNet")

    assert await bridge.ingest_mic_rtp(mic_packet) is None

    generated = await bridge.ingest_mic_rtp(mic_packet)

    assert isinstance(generated, tuple)

    generated_packet = generated[0]

    generated_ssrc = int.from_bytes(generated_packet[8:12], "big")

    binding = _SoundBinding()

    command = _sound_command(generated_ssrc)

    control = _SoundControl(command, generated_packet, binding)

    sink = _SoundSink()

    runtime = ReceiveRuntime(
        config=SoundReceiveConfig(
            orchestrator_ws_url="wss://orchestrator.example.test/control",
            trusted_lan_token=None,
            stream_id="onsite-answer",
            rtp_host=SOUND_BIND_HOST,
            rtp_port=50_006,
            advertised_rtp_host="sound.example.test",
        ),
        udp_binder=_SoundBinder(binding),
        control_connector=_SoundConnector(control),
        playback_sink=sink,
    )

    # When: Sound processes the Orchestrator command then receives the generated RTP.

    await runtime.run()

    # Then: the command SSRC admits the exact generated L16 payload to real playback.

    assert sink.frames == [
        L16PlaybackFrame(
            stream_id=StreamId("onsite-answer"),
            sample_rate=16_000,
            channels=1,
            payload=b"\x20\x10" * 320,
        )
    ]


def _bridge(text: str, expected_audio_bytes: int = 1_280) -> OnsiteExplainerBridge:

    return OnsiteExplainerBridge(
        asr=_Asr(text, expected_audio_bytes),
        tts=_Tts(),
        pipeline_factory=lambda: OrchestratorTurnPipeline(
            adapters=PipelineAdapters(
                mode_policy=AdaptiveAgentPolicy(),
                llm=MockLLMAdapter(("onsite ", "answer")),
                retrieval=RetrievalFixtureProvider(()),
            ),
            config=PipelineConfig(1, "turn-onsite", "segment-onsite"),
        ),
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
        frames_per_utterance=2,
    )


def _source() -> dict[str, JsonValue]:

    return {
        "stream_id": STREAM_ID,
        "ssrc": MIC_SSRC,
        "codec": _codec(),
        "rtp_endpoint": {"host": "declared", "port": 5004},
    }


def _sink() -> dict[str, JsonValue]:

    return {
        "stream_id": STREAM_ID,
        "codec": _codec(),
        "rtp_endpoint": {"host": "declared", "port": 5006},
    }


def _codec() -> dict[str, JsonValue]:

    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _sound_command(ssrc: int) -> str:

    return _registration(
        "media.stream.command",
        "orchestrator",
        {
            "command_id": "onsite-command-001",
            "stream_id": "onsite-answer",
            "start_rtp_timestamp": 96_000,
            "ssrc": ssrc,
            "codec": _codec(),
            "rtp_endpoint": {"host": "sound.example.test", "port": 50_006},
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
            "trace_id": "trace-onsite",
            "session_id": SESSION_ID,
            "seq": 1,
            "data": data,
        }
    )


def _rtp(ssrc: int, payload: bytes, sequence: int = 1, timestamp: int = 1) -> bytes:

    return (
        b"\x80\x60"
        + sequence.to_bytes(2, "big")
        + timestamp.to_bytes(4, "big")
        + ssrc.to_bytes(4, "big")
        + payload
    )


def _wav(payload: bytes) -> bytes:

    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(2)

        audio.setframerate(16_000)

        audio.writeframes(payload)

    return output.getvalue()


def _wav_payload(data: bytes) -> bytes:

    with wave.open(io.BytesIO(data), "rb") as audio:
        return audio.readframes(audio.getnframes())
