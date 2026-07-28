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
    from orchestrator.json_boundary import JsonValue

SESSION_ID: Final = "session-onsite-001"
STREAM_ID: Final = "mic-onsite-001"
MIC_PEER: Final = ("192.0.2.20", 42_000)
SOUND_PEER: Final = ("192.0.2.21", 42_001)
MIC_SSRC: Final = 0x0102_0304
TTS_SSRC: Final = 0x0506_0708
SYNTHESIZED_L16_PAYLOAD: Final = b"\x10\x20" * 320


@dataclass(slots=True)  # noqa: MUTABLE_OK - records datagrams delivered to Sound.
class FakeDatagramTransport:
    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        self.sent.append((data, addr))

    def close(self) -> None:
        return None


@dataclass(slots=True)  # noqa: MUTABLE_OK - records the bridge's externally visible work.
class FakeOnsiteExplainerBridge:
    asr_final: str
    mic_packets: list[bytes] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)

    async def ingest_mic_rtp(self, packet: bytes) -> bytes | None:
        self.mic_packets.append(packet)
        if self.asr_final == "":
            return None
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
        return _rtp_packet(ssrc=TTS_SSRC, payload=SYNTHESIZED_L16_PAYLOAD)


def test_onsite_mic_rtp_is_ingested_and_replaced_with_synthesized_sound_rtp() -> None:
    asyncio.run(_replacement_proof())


async def _replacement_proof() -> None:
    # Given: authenticated Mic and Sound routes plus an onsite ASR/LLM/TTS bridge.
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
    asyncio.run(_blank_asr_proof())


async def _blank_asr_proof() -> None:
    # Given: authenticated Mic and Sound routes with a blank ASR final.
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
    bridge_options = {"onsite_bridge": bridge}
    return RtpHub(transport, **bridge_options)


def _sink_registration() -> str:
    return _envelope(
        "media.rtp.sink.register",
        "sound",
        {"stream_id": STREAM_ID, "codec": _codec(), "rtp_endpoint": _endpoint(5006)},
    )


def _envelope(event_type: str, source: str, data: dict[str, JsonValue]) -> str:
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
    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _endpoint(port: int) -> dict[str, JsonValue]:
    return {"host": "declared.example.test", "port": port}


def _rtp_packet(*, ssrc: int, payload: bytes) -> bytes:
    return b"\x80\x60\x00\x01\x00\x00\x00\x01" + ssrc.to_bytes(4, "big") + payload
