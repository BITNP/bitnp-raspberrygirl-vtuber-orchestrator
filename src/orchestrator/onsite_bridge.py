from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, override

from orchestrator.asr_semantic_gate import (
    AsrGateDecision,
    AsrGateRequest,
    AsrSemanticGate,
)
from orchestrator.funasr_adapter import FunASRWebSocketAdapter
from orchestrator.llm import (
    AdapterConfigError,
    CancellationToken,
    LLMFinal,
    LLMPrompt,
    LLMRequest,
)
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
from orchestrator.provider_streaming import ProviderDeadlines
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.streaming_contracts import CancellationEpoch, SegmentId, StreamKey
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
    from collections.abc import Iterator

    from orchestrator.config import OrchestratorConfig
    from orchestrator.observability import OnsiteObservability


class PipelineFactory(Protocol):
    def __call__(self) -> OrchestratorTurnPipeline: ...


type OnsiteOutput = Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]]

type OnsiteOutputFinished = Callable[[StreamKey, CancellationEpoch], Awaitable[None]]

type OnsiteOutputAuthorization = Callable[[StreamKey, CancellationEpoch], bool]

type OnsiteReplacement = Callable[
    [StreamKey, SegmentId], Awaitable[CancellationEpoch | None]
]


async def _discard_output(
    stream: StreamKey, epoch: CancellationEpoch, packet: bytes
) -> None:
    _ = (stream, epoch, packet)


async def _discard_finished(stream: StreamKey, epoch: CancellationEpoch) -> None:
    _ = (stream, epoch)


def _allow_output(stream: StreamKey, epoch: CancellationEpoch) -> bool:
    _ = (stream, epoch)

    return True


async def _discard_replacement(
    stream: StreamKey, segment_id: SegmentId
) -> CancellationEpoch | None:
    _ = (stream, segment_id)
    return None


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

_LOGGER = logging.getLogger(__name__)
_RTP_DIAGNOSTIC_INTERVAL = 100


def _build_asr_gate(llm: OpenAICompatibleLLMRuntimeAdapter) -> AsrSemanticGate:
    def provider(request: AsrGateRequest) -> str:
        user = (
            f"转写: {request.transcript}\n"
            f"正在播放: {request.is_playing}\n"
            f"当前回答摘录: {request.active_answer_excerpt}"
        )
        for event in llm.stream(LLMRequest(LLMPrompt(request.instruction, user))):
            if isinstance(event, LLMFinal):
                return event.text
        message = "ASR Gate 未返回最终结果"
        raise TimeoutError(message)

    return AsrSemanticGate(provider)


@dataclass(slots=True)
class OnsiteExplainerBridge:
    asr: AsrAdapter

    tts: TtsAdapter

    pipeline_factory: PipelineFactory

    voice: str

    ref_audio: str

    ref_text: str

    asr_gate: AsrSemanticGate | None = None

    frames_per_utterance: int = 50

    legacy_keyed_frames_per_utterance: int | None = None

    _frames: list[bytes] = field(default_factory=list)

    _utterance_sequence: int = 0

    _input_actors: dict[StreamKey, StreamEndpointer] = field(default_factory=dict)

    _legacy_keyed_frames: dict[StreamKey, list[bytes]] = field(default_factory=dict)

    _legacy_keyed_sequences: dict[StreamKey, int] = field(default_factory=dict)

    _rtp_diagnostic_counts: dict[StreamKey, int] = field(default_factory=dict)

    _processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    _stream_actors: dict[StreamKey, OnsiteStreamActor] = field(default_factory=dict)

    _closing_actors: dict[StreamKey, OnsiteStreamActor] = field(default_factory=dict)

    output: OnsiteOutput = field(default=_discard_output)

    output_finished: OnsiteOutputFinished = field(default=_discard_finished)

    begin_replacement: OnsiteReplacement = field(default=_discard_replacement)

    authorize_output: OnsiteOutputAuthorization = field(default=_allow_output)

    observability: OnsiteObservability | None = None

    def set_output_callback(self, callback: OnsiteOutput) -> None:
        self.output = callback

    def set_output_finished_callback(self, callback: OnsiteOutputFinished) -> None:
        self.output_finished = callback

    def set_output_authorizer(self, callback: OnsiteOutputAuthorization) -> None:
        """Install the transport's scheduler-owned output admission callback."""
        self.authorize_output = callback

    def set_replacement_callback(self, callback: OnsiteReplacement) -> None:
        self.begin_replacement = callback

    def set_observability(self, observability: OnsiteObservability) -> None:
        self.observability = observability

    def submit_mic_rtp(
        self, stream: StreamKey, packet: bytes, epoch: CancellationEpoch
    ) -> None:
        frame_count = self._rtp_diagnostic_counts.get(stream, 0) + 1
        self._rtp_diagnostic_counts[stream] = frame_count
        if frame_count % _RTP_DIAGNOSTIC_INTERVAL == 0:
            _LOGGER.debug(
                "onsite_rtp_ingress stream=%s frames=%d epoch=%d",
                stream.stream_id,
                frame_count,
                epoch,
            )
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

        _LOGGER.debug(
            "onsite_endpoint stream=%s reason=%s pcm_bytes=%d epoch=%d",
            stream.stream_id,
            endpoint.reason,
            len(endpoint.payload),
            epoch,
        )
        self._submit_endpoint(stream, endpoint, epoch)

    def _submit_endpoint(
        self,
        stream: StreamKey,
        endpoint: EndpointedUtterance,
        epoch: CancellationEpoch,
    ) -> None:
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
        endpointer = self._input_actors.pop(stream, None)

        if endpointer is not None:
            endpoint = endpointer.disconnect()

            if endpoint is not None:
                self._submit_endpoint(stream, endpoint, endpoint.cancellation_epoch)

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

        _LOGGER.debug(
            "onsite_asr_request segment=%s audio_bytes=%d format=%s",
            segment_id,
            len(audio),
            filename,
        )

        started_at = perf_counter()
        try:
            event = self.asr.transcribe(
                audio=audio,
                filename=filename,
                received_at_ms=sequence * 20,
                segment_id=segment_id or f"asr-onsite-{sequence:04d}",
                seq=sequence,
                cancellation=cancellation,
            )
        except (MediaAdapterConfigError, OSError):
            _LOGGER.exception("onsite_asr_failed segment=%s", segment_id)
            return None
        else:
            if event is None:
                _LOGGER.debug("onsite_asr_empty segment=%s", segment_id)
                return None
            _LOGGER.debug(
                "onsite_asr_final segment=%s chars=%d latency_ms=%.1f",
                event.segment_id,
                len(event.text),
                (perf_counter() - started_at) * 1_000,
            )
            return event

    def answer(
        self,
        pipeline: OrchestratorTurnPipeline,
        event: ASRAudienceEvent,
        cancellation: CancellationToken,
    ) -> TurnResult | None:
        if not pipeline.accept_audience_input(event):
            return None

        _LOGGER.debug(
            "onsite_llm_request segment=%s transcript_chars=%d",
            event.segment_id,
            len(event.text),
        )

        started_at = perf_counter()
        try:
            turn = pipeline.process_next_turn(cancellation)
        except (AdapterConfigError, OSError):
            _LOGGER.exception("onsite_llm_failed")
            return None
        else:
            if turn is not None:
                _LOGGER.debug(
                    "onsite_llm_final turn=%s chars=%d latency_ms=%.1f",
                    turn.turn_id,
                    len(turn.answer_text),
                    (perf_counter() - started_at) * 1_000,
                )
            return turn

    def gate(
        self,
        event: ASRAudienceEvent,
        *,
        active_answer_excerpt: str,
        is_playing: bool,
    ) -> AsrGateDecision:
        gate = self.asr_gate
        if gate is None:
            return AsrGateDecision.DISCARD if is_playing else AsrGateDecision.ACCEPT
        decision = gate.evaluate(
            event.text,
            active_answer_excerpt=active_answer_excerpt,
            is_playing=is_playing,
        )
        _LOGGER.info(
            "asr_gate segment=%s decision=%s playing=%s transcript_chars=%d",
            event.segment_id,
            decision,
            is_playing,
            len(event.text),
        )
        return decision

    async def prepare_replacement(
        self, stream: StreamKey, segment_id: SegmentId
    ) -> CancellationEpoch | None:
        return await self.begin_replacement(stream, segment_id)

    def synthesize(
        self, text: str, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None:
        started_at = perf_counter()
        _LOGGER.debug(
            "onsite_tts_request text_chars=%d voice=%s", len(text), self.voice
        )
        try:
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

            chunks = (Pcm16leChunk(pcm16le),)
        except (MediaAdapterConfigError, OnsiteBridgeMediaError, OSError, wave.Error):
            _LOGGER.exception("onsite_tts_failed")
            return None
        else:
            _LOGGER.debug(
                "onsite_tts_complete pcm_bytes=%d latency_ms=%.1f",
                len(pcm16le),
                (perf_counter() - started_at) * 1_000,
            )
            return chunks

    def stream_synthesize(
        self, text: str, cancellation: CancellationToken
    ) -> Iterator[Pcm16leChunk] | None:
        if (
            getattr(self.tts, "capability", "final_only") != "streaming_sse"
            or not isinstance(self.tts, StreamingTtsAdapter)
        ):
            return None
        return self.tts.stream_pcm16le(
            text=text,
            voice=self.voice,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            cancellation=cancellation,
        )

    def complete(
        self,
        pipeline: OrchestratorTurnPipeline,
        turn: TurnResult,
        chunks: tuple[Pcm16leChunk, ...],
    ) -> None:
        self.complete_stream(
            pipeline,
            turn,
            sum(len(chunk.data) for chunk in chunks),
        )

    def complete_stream(
        self,
        pipeline: OrchestratorTurnPipeline,
        turn: TurnResult,
        pcm_bytes: int,
    ) -> None:
        _ = pipeline.complete_synthesis(
            MockSynthesisResult(
                turn_id=turn.turn_id,
                segment_id=turn.segment_id,
                audio=AudioMetadata(
                    _SAMPLE_RATE,
                    1,
                    "pcm_s16le",
                    pcm_bytes // 32,
                    pcm_bytes,
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

    def gate(
        self,
        event: ASRAudienceEvent,
        *,
        active_answer_excerpt: str,
        is_playing: bool,
    ) -> AsrGateDecision:
        return self.bridge.gate(
            event,
            active_answer_excerpt=active_answer_excerpt,
            is_playing=is_playing,
        )

    def authorize_output(self, stream: StreamKey, epoch: CancellationEpoch) -> bool:
        return self.bridge.authorize_output(stream, epoch)

    async def prepare_replacement(
        self, stream: StreamKey, segment_id: SegmentId
    ) -> CancellationEpoch | None:
        return await self.bridge.prepare_replacement(stream, segment_id)

    @override
    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None:
        return self.bridge.synthesize(turn.answer_text, cancellation)

    def stream_synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> Iterator[Pcm16leChunk] | None:
        return self.bridge.stream_synthesize(turn.answer_text, cancellation)

    @override
    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:
        self.bridge.complete(self.pipeline, turn, chunks)

    def complete_stream(self, turn: TurnResult, pcm_bytes: int) -> None:
        self.bridge.complete_stream(self.pipeline, turn, pcm_bytes)

    @override
    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        await self.bridge.output(stream, epoch, packet)

    async def finish_output(self, stream: StreamKey, epoch: CancellationEpoch) -> None:
        await self.bridge.output_finished(stream, epoch)


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

    llm = OpenAICompatibleLLMRuntimeAdapter(
        config.llm_endpoint,
        config.llm_model,
        config.llm_api_key,
        timeout_seconds=120.0,
        deadlines=ProviderDeadlines(read_seconds=60.0, total_seconds=120.0),
        ca_path=config.tls_ca_path,
    )

    adapters = PipelineAdapters(
        mode_policy=AdaptiveAgentPolicy(),
        llm=llm,
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
            capability=config.tts_mode,
            ca_path=config.tls_ca_path,
        ),
        pipeline_factory=pipeline_factory,
        voice=voice,
        ref_audio=ref_audio,
        ref_text=ref_text,
        asr_gate=_build_asr_gate(llm),
    )
