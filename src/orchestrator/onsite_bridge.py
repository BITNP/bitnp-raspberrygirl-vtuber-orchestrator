
from __future__ import annotations

import asyncio
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, override

from orchestrator.funasr_adapter import FunASRWebSocketAdapter
from orchestrator.llm import AdapterConfigError, CancellationToken
from orchestrator.media_adapters import (
    MediaAdapterConfigError,
    OpenAICompatibleASRAdapter,
    VllmOmniTTSAdapter,
)
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.onsite_bridge_contracts import (
    AsrAdapter,
    OnsiteBridgeConfigError,
    OnsiteBridgeMediaError,
    TtsAdapter,
    l16_from_wav,
    pcm16le_from_l16,
    wav_from_l16,
)
from orchestrator.onsite_stream_actor import OnsiteStages, OnsiteStreamActor
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
from orchestrator.streaming_contracts import CancellationEpoch, StreamKey
from orchestrator.streaming_endpoint import (
    EndpointedUtterance,
    PartialUtterance,
    StreamEndpointer,
)
from orchestrator.transport_hub import (
    RTP_HEADER_BYTES,
    RTP_PAYLOAD_TYPE,
    RTP_V2_HEADER,
)
from orchestrator.tts_rtp import (
    Pcm16leChunk,
    StreamingTtsAdapter,
    TtsPcmRtpPacketizer,
    generated_ssrc,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig
    from orchestrator.observability import OnsiteObservability


class PipelineFactory(Protocol):

    def __call__(self) -> OrchestratorTurnPipeline:
        ...


type OnsiteOutput = Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]]


async def _discard_output(
    stream: StreamKey, epoch: CancellationEpoch, packet: bytes
) -> None:
    _ = (stream, epoch, packet)


@dataclass(frozen=True, slots=True)
class _LegacyTurn:

    utterance: bytes

    sequence: int

    segment_id: str | None

    stream: StreamKey | None

    epoch: CancellationEpoch | None

    pipeline: OrchestratorTurnPipeline


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

    asr: AsrAdapter

    tts: TtsAdapter

    pipeline_factory: PipelineFactory

    voice: str

    ref_audio: str

    ref_text: str

    frames_per_utterance: int = 50

    legacy_keyed_frames_per_utterance: int | None = None

    _frames: list[bytes] = field(default_factory=list)

    _utterance_sequence: int = 0

    _input_actors: dict[StreamKey, StreamEndpointer] = field(default_factory=dict)

    _legacy_keyed_frames: dict[StreamKey, list[bytes]] = field(default_factory=dict)

    _legacy_keyed_sequences: dict[StreamKey, int] = field(default_factory=dict)

    _processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    _stream_actors: dict[StreamKey, OnsiteStreamActor] = field(default_factory=dict)

    _closing_actors: dict[StreamKey, OnsiteStreamActor] = field(default_factory=dict)

    output: OnsiteOutput = field(default=_discard_output)

    observability: OnsiteObservability | None = None

    def set_output_callback(self, callback: OnsiteOutput) -> None:
        self.output = callback

    def set_observability(self, observability: OnsiteObservability) -> None:
        self.observability = observability

    def submit_mic_rtp(
        self, stream: StreamKey, packet: bytes, epoch: CancellationEpoch
    ) -> None:
        endpointer = self._input_actors.setdefault(stream, StreamEndpointer(stream))

        endpoint: EndpointedUtterance | None = None

        for event in endpointer.push(packet):
            match event:
                case PartialUtterance() as partial:
                    self._record_partial(partial, epoch)

                case EndpointedUtterance() as final:
                    endpoint = final

        if endpoint is None and self.legacy_keyed_frames_per_utterance == 1:
            endpoint = endpointer.disconnect()

        if endpoint is None:
            return

        actor = self._stream_actors.get(stream)

        if actor is None:
            stages = _BridgeStages(self, self.pipeline_factory())

            actor = OnsiteStreamActor(
                stream, epoch, stages, observability=self.observability
            )

            self._stream_actors[stream] = actor

        actor.submit(endpoint, epoch)

    def _record_partial(
        self, partial: PartialUtterance, epoch: CancellationEpoch
    ) -> None:
        observability = self.observability

        if observability is not None:
            correlation = observability.correlation(
                partial.stream, str(partial.turn_id), str(partial.segment_id), epoch
            )

            if correlation is not None:
                observability.record("asr_partial", correlation)

    def invalidate_stream(
        self, stream: StreamKey, next_epoch: CancellationEpoch
    ) -> None:
        _ = self._input_actors.pop(stream, None)

        actor = self._stream_actors.pop(stream, None)

        if actor is not None:
            actor.invalidate(next_epoch)

            self._closing_actors[stream] = actor

    async def aclose_stream(self, stream: StreamKey) -> None:
        actor = self._stream_actors.pop(stream, None)

        if actor is not None:
            actor.invalidate(CancellationEpoch(int(actor.epoch) + 1))

            self._closing_actors[stream] = actor

        closing = self._closing_actors.pop(stream, None)

        if closing is not None:
            await closing.wait_quiescent()

    async def wait_quiescent(self) -> None:
        for actor in tuple(self._stream_actors.values()):
            await actor.wait_quiescent()

        for stream, actor in tuple(self._closing_actors.items()):
            await actor.wait_quiescent()

            del self._closing_actors[stream]

    async def ingest_mic_rtp(
        self, stream: StreamKey | bytes, packet: bytes = b""
    ) -> bytes | tuple[bytes, ...] | None:
        if isinstance(stream, bytes):
            return await self._ingest_legacy_batch(stream)

        if self.legacy_keyed_frames_per_utterance is not None:
            return await self._ingest_legacy_keyed_batch(stream, packet)

        async with self._processing_lock:
            return await asyncio.to_thread(
                self._process_utterance,
                _LegacyTurn(
                    packet[RTP_HEADER_BYTES:],
                    1,
                    None,
                    stream,
                    CancellationEpoch(0),
                    self.pipeline_factory(),
                ),
            )

    def disconnect_stream(self, stream: StreamKey) -> None:
        actor = self._stream_actors.get(stream)

        next_epoch = (
            CancellationEpoch(0)
            if actor is None
            else CancellationEpoch(int(actor.epoch) + 1)
        )

        self.invalidate_stream(stream, next_epoch)

        _ = self._legacy_keyed_frames.pop(stream, None)

        _ = self._legacy_keyed_sequences.pop(stream, None)

    async def _ingest_legacy_keyed_batch(
        self, stream: StreamKey, packet: bytes
    ) -> bytes | tuple[bytes, ...] | None:
        frames = self._legacy_keyed_frames.setdefault(stream, [])

        frames.append(packet[RTP_HEADER_BYTES:])

        frame_limit = self.legacy_keyed_frames_per_utterance

        if frame_limit is None or len(frames) < frame_limit:
            return None

        utterance = b"".join(frames)

        frames.clear()

        sequence = self._legacy_keyed_sequences.get(stream, 0) + 1

        self._legacy_keyed_sequences[stream] = sequence

        async with self._processing_lock:
            return await asyncio.to_thread(
                self._process_utterance,
                _LegacyTurn(
                    utterance,
                    sequence,
                    None,
                    stream,
                    CancellationEpoch(0),
                    self.pipeline_factory(),
                ),
            )

    async def _ingest_legacy_batch(
        self, packet: bytes
    ) -> bytes | tuple[bytes, ...] | None:
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
                _LegacyTurn(
                    utterance,
                    utterance_sequence,
                    None,
                    None,
                    None,
                    self.pipeline_factory(),
                ),
            )

    def _process_utterance(self, work: _LegacyTurn) -> bytes | tuple[bytes, ...] | None:
        cancellation = CancellationToken()

        event = self.transcribe_endpoint(
            work.utterance, work.sequence, work.segment_id, cancellation
        )

        generated: bytes | tuple[bytes, ...] | None = None

        if event is not None and event.text.strip() != "":
            turn = self.answer(work.pipeline, event, cancellation)

            if turn is not None and turn.answer_text.strip() != "":
                chunks = self.synthesize(turn.answer_text, cancellation)

                if chunks:
                    self.complete(work.pipeline, turn, chunks)

                    resolved_stream = StreamKey("legacy", "onsite-answer")

                    if work.stream is not None:
                        resolved_stream = work.stream

                    resolved_epoch = (
                        CancellationEpoch(work.sequence)
                        if work.epoch is None
                        else work.epoch
                    )

                    generated = self._packets(chunks, resolved_stream, resolved_epoch)

        return generated

    def transcribe_endpoint(
        self,
        utterance: bytes,
        sequence: int,
        segment_id: str | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ASRAudienceEvent | None:
        audio = wav_from_l16(utterance)

        filename = "onsite-l16.wav"

        if isinstance(self.asr, FunASRWebSocketAdapter):
            audio = pcm16le_from_l16(utterance)

            filename = "onsite-l16.pcm"

        try:
            return self.asr.transcribe(
                audio=audio,
                filename=filename,
                received_at_ms=sequence * 20,
                segment_id=segment_id or f"asr-onsite-{sequence:04d}",
                seq=sequence,
                cancellation=cancellation,
            )

        except (MediaAdapterConfigError, OSError):
            return None

    def answer(
        self,
        pipeline: OrchestratorTurnPipeline,
        event: ASRAudienceEvent,
        cancellation: CancellationToken,
    ) -> TurnResult | None:
        if not pipeline.accept_audience_input(event):
            return None

        try:
            return pipeline.process_next_turn(cancellation)

        except (AdapterConfigError, OSError):
            return None

    def synthesize(
        self, text: str, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None:
        try:
            if isinstance(self.tts, StreamingTtsAdapter):
                return self.tts.stream_pcm16le(
                    text=text,
                    voice=self.voice,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                    cancellation=cancellation,
                )

            response = self.tts.synthesize(
                text=text,
                voice=self.voice,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                cancellation=cancellation,
            )

            l16 = l16_from_wav(response)

            pcm16le = b"".join(
                l16[offset : offset + 2][::-1] for offset in range(0, len(l16), 2)
            )

            return (Pcm16leChunk(pcm16le),)

        except (MediaAdapterConfigError, OnsiteBridgeMediaError, OSError, wave.Error):
            return None

    def complete(
        self,
        pipeline: OrchestratorTurnPipeline,
        turn: TurnResult,
        chunks: tuple[Pcm16leChunk, ...],
    ) -> None:
        _ = pipeline.complete_synthesis(
            MockSynthesisResult(
                turn_id=turn.turn_id,
                segment_id=turn.segment_id,
                audio=AudioMetadata(
                    _SAMPLE_RATE,
                    1,
                    "pcm_s16le",
                    sum(len(chunk.data) for chunk in chunks) // 32,
                    sum(len(chunk.data) for chunk in chunks),
                ),
                expression="smile",
                action="speak",
                scene="onsite",
                slide_page=1,
            ),
            rtp_stream_start_ms=_START_TIMESTAMP // 16,
            stream_id="onsite-answer",
        )

    def _packets(
        self,
        chunks: tuple[Pcm16leChunk, ...],
        stream: StreamKey,
        cancellation_epoch: CancellationEpoch,
    ) -> tuple[bytes, ...]:
        packetizer = TtsPcmRtpPacketizer(
            stream=stream,
            cancellation_epoch=cancellation_epoch,
        )

        packets = tuple(packet for chunk in chunks for packet in packetizer.push(chunk))

        return packets + packetizer.finish()


@dataclass(frozen=True, slots=True)
class _BridgeStages(OnsiteStages):

    bridge: OnsiteExplainerBridge

    pipeline: OrchestratorTurnPipeline

    @override
    def transcribe(
        self, endpoint: EndpointedUtterance, cancellation: CancellationToken
    ) -> ASRAudienceEvent | None:
        return self.bridge.transcribe_endpoint(
            endpoint.payload,
            int(endpoint.cancellation_epoch) + 1,
            str(endpoint.segment_id),
            cancellation,
        )

    @override
    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult | None:
        return self.bridge.answer(self.pipeline, event, cancellation)

    @override
    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None:
        return self.bridge.synthesize(turn.answer_text, cancellation)

    @override
    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:
        self.bridge.complete(self.pipeline, turn, chunks)

    @override
    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        await self.bridge.output(stream, epoch, packet)


def build_onsite_bridge(
    config: OrchestratorConfig,
    *,
    voice: str,
    ref_audio: str,
    ref_text: str,
) -> OnsiteExplainerBridge:
    if (
        config.asr_provider not in {"openai_compatible", "funasr"}
        or config.tts_provider != "vllm_omni"
    ):
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

    adapters = PipelineAdapters(
        mode_policy=AdaptiveAgentPolicy(),
        llm=OpenAICompatibleLLMRuntimeAdapter(
            config.llm_endpoint,
            config.llm_model,
            config.llm_api_key,
            ca_path=config.tls_ca_path,
        ),
        retrieval=RetrievalFixtureProvider(()),
    )

    pipeline_config = PipelineConfig(1, "turn-onsite", "segment-onsite")

    def pipeline_factory() -> OrchestratorTurnPipeline:
        return OrchestratorTurnPipeline(adapters=adapters, config=pipeline_config)

    asr = OpenAICompatibleASRAdapter(
        config.asr_endpoint,
        config.asr_model,
        config.asr_api_key,
        ca_path=config.tls_ca_path,
    )

    if config.asr_provider == "funasr":
        asr = FunASRWebSocketAdapter(
            config.asr_endpoint, config.asr_model, config.tls_ca_path
        )

    return OnsiteExplainerBridge(
        asr=asr,
        tts=VllmOmniTTSAdapter(
            config.tts_endpoint,
            config.tts_model,
            config.tts_api_key,
            ca_path=config.tls_ca_path,
        ),
        pipeline_factory=pipeline_factory,
        voice=voice,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
