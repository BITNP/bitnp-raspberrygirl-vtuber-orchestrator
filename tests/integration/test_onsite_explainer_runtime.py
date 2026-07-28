from __future__ import annotations

import asyncio
import io
import json
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.json_boundary import JsonValue

from sound.receive import ReceiveRuntime
from sound.receive_config import SoundReceiveConfig
from sound.rtp_playback import L16PlaybackFrame, StreamId

from orchestrator.llm import MockLLMAdapter
from orchestrator.media_adapters import SynthesizedAudio
from orchestrator.modes import ModePolicy
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import ASRAudienceEvent, PipelineConfig
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.transport_hub import RtpHub

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
        return None


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
        return None


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
        return None


@dataclass(frozen=True, slots=True)
class _SoundConnector:
    control: _SoundControl

    async def connect(self, url: str, headers: dict[str, str]) -> _SoundControl:
        assert url == "wss://orchestrator.example.test/control"
        assert headers == {}
        return self.control


@dataclass(slots=True)
class _SoundSink:
    frames: list[L16PlaybackFrame] = field(default_factory=list)

    def write(self, frame: L16PlaybackFrame) -> None:
        self.frames.append(frame)

    def close_stream(self, stream_id: str) -> None:
        _ = stream_id

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _Asr:
    text: str

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        received_at_ms: int,
        segment_id: str,
        seq: int,
    ) -> ASRAudienceEvent | None:
        assert filename == "onsite-l16.wav"
        assert len(_wav_payload(audio)) == 1_280
        return ASRAudienceEvent(self.text, received_at_ms, segment_id, seq)


@dataclass(frozen=True, slots=True)
class _Tts:
    def synthesize(
        self, *, text: str, voice: str, ref_audio: str, ref_text: str
    ) -> SynthesizedAudio:
        assert text == "onsite answer"
        assert voice == "raspberry"
        assert ref_audio == "file:///voice.wav"
        assert ref_text == "reference"
        return SynthesizedAudio(_wav(b"\x10\x20" * 320), "audio/wav")


def test_onsite_runtime_composes_asr_pipeline_tts_and_replaces_mic_rtp() -> None:
    asyncio.run(_runtime_composition_proof())


async def _runtime_composition_proof() -> None:
    # Given: the real bridge with deterministic provider boundaries and pinned routes.
    transport = _Datagrams()
    hub = RtpHub(transport, onsite_bridge=_bridge("Explain BitNet"))
    hub.register_control(
        _registration("media.rtp.source.register", "mic", _source()), MIC_PEER[0]
    )
    hub.register_control(
        _registration("media.rtp.sink.register", "sound", _sink()), SOUND_PEER[0]
    )
    mic_packet = _rtp(MIC_SSRC, b"\x01\x02" * 320)

    # When: two registered Mic frames complete the deterministic utterance.
    first_delivered = hub.route_datagram(mic_packet, MIC_PEER)
    delivered = hub.route_datagram(mic_packet, MIC_PEER)
    await hub.wait_for_onsite_jobs()

    # Then: Sound receives generated canonical RTP, never the microphone packet.
    assert first_delivered is False
    assert delivered is False
    assert len(transport.sent) == 1
    generated, endpoint = transport.sent[0]
    assert endpoint == (SOUND_PEER[0], 5006)
    assert generated != mic_packet
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


def _bridge(text: str) -> OnsiteExplainerBridge:
    return OnsiteExplainerBridge(
        asr=_Asr(text),
        tts=_Tts(),
        pipeline=OrchestratorTurnPipeline(
            adapters=PipelineAdapters(
                mode_policy=ModePolicy.onsite_explainer(),
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


def _rtp(ssrc: int, payload: bytes) -> bytes:
    return b"\x80\x60\x00\x01\x00\x00\x00\x01" + ssrc.to_bytes(4, "big") + payload


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
