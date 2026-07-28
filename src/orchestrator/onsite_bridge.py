"""Onsite L16 RTP composition through ASR, turn handling, and TTS."""

from __future__ import annotations

import asyncio
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.llm import AdapterConfigError
from orchestrator.media_adapters import (
    MediaAdapterConfigError,
    OpenAICompatibleASRAdapter,
    VllmOmniTTSAdapter,
)
from orchestrator.modes import ModePolicy
from orchestrator.onsite_bridge_contracts import (
    AsrAdapter,
    OnsiteBridgeConfigError,
    OnsiteBridgeMediaError,
    TtsAdapter,
    generated_ssrc,
    l16_from_wav,
    pad_l16_frames,
    wav_from_l16,
)
from orchestrator.openai_llm_runtime import OpenAICompatibleLLMRuntimeAdapter
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    AudioMetadata,
    MockSynthesisResult,
    PipelineConfig,
    TurnResult,
)
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.transport_hub import (
    L16_FRAME_BYTES,
    RTP_HEADER_BYTES,
    RTP_PAYLOAD_TYPE,
    RTP_V2_HEADER,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

__all__ = (
    "OnsiteBridgeConfigError",
    "OnsiteBridgeMediaError",
    "OnsiteExplainerBridge",
    "build_onsite_bridge",
    "generated_ssrc",
)

_SAMPLE_RATE = 16_000
_SAMPLES_PER_FRAME = 320
_START_TIMESTAMP = 96_000
_RTP_HEADER_PREFIX = bytes((RTP_V2_HEADER, RTP_PAYLOAD_TYPE))
@dataclass(slots=True)
class OnsiteExplainerBridge:
    """Converts one authenticated Mic L16 frame into generated Sound RTP."""

    asr: AsrAdapter
    tts: TtsAdapter
    pipeline: OrchestratorTurnPipeline
    voice: str
    ref_audio: str
    ref_text: str
    frames_per_utterance: int = 50
    _frames: list[bytes] = field(default_factory=list)
    _utterance_sequence: int = 0
    _rtp_sequence: int = 0
    _rtp_timestamp: int = _START_TIMESTAMP
    _processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def ingest_mic_rtp(self, packet: bytes) -> bytes | tuple[bytes, ...] | None:
        """Schedule provider work after collecting one canonical input frame."""
        self._frames.append(packet[RTP_HEADER_BYTES:])
        if len(self._frames) < self.frames_per_utterance:
            return None
        utterance = b"".join(self._frames)
        self._frames.clear()
        self._utterance_sequence += 1
        utterance_sequence = self._utterance_sequence
        async with self._processing_lock:
            return await asyncio.to_thread(
                self._process_utterance,
                utterance,
                utterance_sequence,
                int.from_bytes(packet[8:12], "big"),
            )

    def _process_utterance(
        self, utterance: bytes, utterance_sequence: int, mic_ssrc: int
    ) -> bytes | tuple[bytes, ...] | None:
        """Run the synchronous provider pipeline away from the event loop."""
        event = self._transcribe(utterance, utterance_sequence)
        generated: bytes | tuple[bytes, ...] | None = None
        if event is not None and event.text.strip() != "":
            turn = self._answer_for(event)
            if turn is not None and turn.answer_text.strip() != "":
                audio = self._synthesize(turn.answer_text)
                if audio is not None and len(audio) > 0:
                    _ = self.pipeline.complete_synthesis(
                        MockSynthesisResult(
                            turn_id=turn.turn_id,
                            segment_id=turn.segment_id,
                            audio=AudioMetadata(
                                _SAMPLE_RATE,
                                1,
                                "pcm_s16le",
                                len(audio) // 32,
                                len(audio),
                            ),
                            expression="smile",
                            action="speak",
                            scene="onsite",
                            slide_page=1,
                        ),
                        rtp_stream_start_ms=self._rtp_timestamp // 16,
                        stream_id="onsite-answer",
                    )
                    generated = self._packets(
                        audio,
                        mic_ssrc,
                    )
        return generated

    def _transcribe(self, utterance: bytes, sequence: int) -> ASRAudienceEvent | None:
        try:
            return self.asr.transcribe(
                audio=wav_from_l16(utterance),
                filename="onsite-l16.wav",
                received_at_ms=sequence * 20,
                segment_id=f"asr-onsite-{sequence:04d}",
                seq=sequence,
            )
        except (MediaAdapterConfigError, OSError):
            return None

    def _answer_for(self, event: ASRAudienceEvent) -> TurnResult | None:
        if not self.pipeline.accept_audience_input(event):
            return None
        try:
            return self.pipeline.process_next_turn()
        except (AdapterConfigError, OSError):
            return None

    def _synthesize(self, text: str) -> bytes | None:
        try:
            response = self.tts.synthesize(
                text=text,
                voice=self.voice,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
            )
            return l16_from_wav(response)
        except (MediaAdapterConfigError, OnsiteBridgeMediaError, OSError, wave.Error):
            return None

    def _packets(self, audio: bytes, mic_ssrc: int) -> tuple[bytes, ...]:
        payload = pad_l16_frames(audio)
        ssrc = generated_ssrc(mic_ssrc)
        packets: list[bytes] = []
        for offset in range(0, len(payload), L16_FRAME_BYTES):
            header = (
                _RTP_HEADER_PREFIX
                + self._rtp_sequence.to_bytes(2, "big")
                + self._rtp_timestamp.to_bytes(4, "big")
                + ssrc.to_bytes(4, "big")
            )
            packets.append(header + payload[offset : offset + L16_FRAME_BYTES])
            self._rtp_sequence = (self._rtp_sequence + 1) % 65_536
            self._rtp_timestamp = (self._rtp_timestamp + _SAMPLES_PER_FRAME) % (2**32)
        return tuple(packets)


def build_onsite_bridge(
    config: OrchestratorConfig,
    *,
    voice: str,
    ref_audio: str,
    ref_text: str,
) -> OnsiteExplainerBridge:
    """Compose configured providers with the onsite policy and turn pipeline."""
    if config.asr_provider != "openai_compatible" or config.tts_provider != "vllm_omni":
        raise OnsiteBridgeConfigError(field_name="asr_provider_or_tts_provider")
    if config.asr_endpoint is None or config.asr_model is None:
        raise OnsiteBridgeConfigError(field_name="asr_endpoint_or_asr_model")
    if config.tts_endpoint is None or config.tts_model is None:
        raise OnsiteBridgeConfigError(field_name="tts_endpoint_or_tts_model")
    if (
        config.llm_provider != "openai_compatible"
        or config.llm_endpoint is None
        or config.llm_model is None
        or config.llm_api_key is None
    ):
        raise OnsiteBridgeConfigError(field_name="llm_provider_or_llm_configuration")
    if voice.strip() == "" or ref_audio.strip() == "" or ref_text.strip() == "":
        raise OnsiteBridgeConfigError(field_name="voice_reference")
    return OnsiteExplainerBridge(
        asr=OpenAICompatibleASRAdapter(
            config.asr_endpoint, config.asr_model, config.asr_api_key
        ),
        tts=VllmOmniTTSAdapter(
            config.tts_endpoint,
            config.tts_model,
            config.tts_api_key,
        ),
        pipeline=OrchestratorTurnPipeline(
            adapters=PipelineAdapters(
                mode_policy=ModePolicy.onsite_explainer(),
                llm=OpenAICompatibleLLMRuntimeAdapter(
                    config.llm_endpoint,
                    config.llm_model,
                    config.llm_api_key,
                ),
                retrieval=RetrievalFixtureProvider(()),
            ),
            config=PipelineConfig(1, "turn-onsite", "segment-onsite"),
        ),
        voice=voice,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
