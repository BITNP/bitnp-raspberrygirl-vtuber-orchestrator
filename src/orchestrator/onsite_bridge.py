from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast, override

from orchestrator.asr_semantic_gate import (
    AsrGateDecision,
    AsrSemanticGate,
    AsyncAsrSemanticGate,
)
from orchestrator.funasr_adapter import FunASRWebSocketAdapter
from orchestrator.llm import (
    AdapterConfigError,
    CancellationToken,
)
from orchestrator.media_adapters import (
    MediaAdapterConfigError,
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
from orchestrator.openai_llm_runtime import AsyncOpenAICompatibleLLMRuntime
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
    L16_FRAME_BYTES,
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
    from orchestrator.llm import LLMAdapter
    from orchestrator.observability import OnsiteObservability


class PipelineFactory(Protocol):
    def __call__(self) -> OrchestratorTurnPipeline: ...


type OnsiteOutput = Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]]

type OnsiteOutputFinished = Callable[[StreamKey, CancellationEpoch], Awaitable[None]]

type OnsiteOutputAuthorization = Callable[[StreamKey, CancellationEpoch], bool]

type OnsiteResponseOutputPreparation = Callable[
    [StreamKey, CancellationEpoch, str], Awaitable[CancellationEpoch | None]
]

type OnsiteReplacement = Callable[
    [StreamKey, SegmentId], Awaitable[CancellationEpoch | None]
]

type OnsiteAsrFinal = Callable[[StreamKey, ASRAudienceEvent], Awaitable[bool]]

type ResponseOutputStarted = Callable[[], bool]


async def _discard_output(
    stream: StreamKey, epoch: CancellationEpoch, packet: bytes
) -> None:
    _ = (stream, epoch, packet)


async def _discard_finished(stream: StreamKey, epoch: CancellationEpoch) -> None:
    _ = (stream, epoch)


def _allow_output(stream: StreamKey, epoch: CancellationEpoch) -> bool:
    _ = (stream, epoch)

    return True


async def _prepare_reuse_requested_output_epoch(
    stream: StreamKey, epoch: CancellationEpoch, turn_id: str
) -> CancellationEpoch:
    _ = (stream, turn_id)
    return epoch


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
@dataclass(frozen=True, slots=True)
class _MicOwnedAsrAdapter:
    """Fail closed if retired local-ASR code is invoked accidentally."""

    def transcribe(self, **_kwargs: object) -> ASRAudienceEvent | None:
        return None


@dataclass(slots=True)
class OnsiteExplainerBridge:
    asr: AsrAdapter

    tts: TtsAdapter

    pipeline_factory: PipelineFactory

    voice: str

    ref_audio: str

    ref_text: str

    asr_gate: AsrSemanticGate | AsyncAsrSemanticGate | None = None

    llm: AsyncOpenAICompatibleLLMRuntime | None = None

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

    prepare_response_output: OnsiteResponseOutputPreparation = field(
        default=_prepare_reuse_requested_output_epoch
    )

    observability: OnsiteObservability | None = None

    asr_final_handler: OnsiteAsrFinal | None = None

    def set_output_callback(self, callback: OnsiteOutput) -> None:
        self.output = callback

    def set_output_finished_callback(self, callback: OnsiteOutputFinished) -> None:
        self.output_finished = callback

    def set_output_authorizer(self, callback: OnsiteOutputAuthorization) -> None:
        """Install the transport's scheduler-owned output admission callback."""
        self.authorize_output = callback

    def set_response_output_preparer(
        self, callback: OnsiteResponseOutputPreparation
    ) -> None:
        """Install the async first-frame/cutover admission callback."""
        self.prepare_response_output = callback

    def set_replacement_callback(self, callback: OnsiteReplacement) -> None:
        self.begin_replacement = callback

    def set_asr_final_handler(self, handler: OnsiteAsrFinal) -> None:
        """Route finalized speech to the session-owned Agent Pipeline."""
        self.asr_final_handler = handler

    def set_observability(self, observability: OnsiteObservability) -> None:
        self.observability = observability

    def submit_mic_rtp(
        self, stream: StreamKey, packet: bytes, epoch: CancellationEpoch
    ) -> None:
        # Mic RTP ingress is retired. Keep the legacy implementation below for
        # controlled compatibility rollouts, but fail closed in this build.
        if not self._accepts_retired_mic_rtp(stream):
            return
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

    def _accepts_retired_mic_rtp(self, stream: StreamKey) -> bool:
        _ = stream
        return False

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
            return await self._process_utterance_async(
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
            return await self._process_utterance_async(
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
            return await self._process_utterance_async(
                _LegacyTurn(
                    utterance,
                    utterance_sequence,
                    None,
                    None,
                    None,
                    self.pipeline_factory(),
                ),
            )

    async def _process_utterance_async(
        self, work: _LegacyTurn
    ) -> bytes | tuple[bytes, ...] | None:
        cancellation = CancellationToken()

        event = await asyncio.to_thread(
            self.transcribe_endpoint,
            work.utterance,
            work.sequence,
            work.segment_id,
            cancellation,
        )

        generated: bytes | tuple[bytes, ...] | None = None

        if event is not None and event.text.strip() != "":
            turn = await self.answer_async(work.pipeline, event, cancellation)

            if turn is not None and turn.answer_text.strip() != "":
                chunks = await asyncio.to_thread(
                    self.synthesize, turn.answer_text, cancellation
                )

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
        for attempt in range(2):
            try:
                event = self.asr.transcribe(
                    audio=audio,
                    filename=filename,
                    received_at_ms=sequence * 20,
                    segment_id=segment_id or f"asr-onsite-{sequence:04d}",
                    seq=sequence,
                    cancellation=cancellation,
                )
            except MediaAdapterConfigError:
                _LOGGER.exception("onsite_asr_permanent_failure segment=%s", segment_id)
                return None
            except OSError:
                if cancellation is not None and cancellation.cancelled:
                    return None
                if attempt == 1:
                    _LOGGER.exception("onsite_asr_failed segment=%s", segment_id)
                    return None
                _LOGGER.warning(
                    "onsite_asr_retry segment=%s attempt=%d", segment_id, attempt + 1
                )
                continue
            if event is None:
                _LOGGER.debug("onsite_asr_empty segment=%s", segment_id)
                return None
            _LOGGER.debug(
                "onsite_asr_final segment=%s text=%r latency_ms=%.1f",
                event.segment_id,
                event.text,
                (perf_counter() - started_at) * 1_000,
            )
            return event
        return None

    async def answer_async(
        self,
        pipeline: OrchestratorTurnPipeline,
        event: ASRAudienceEvent,
        cancellation: CancellationToken,
    ) -> TurnResult | None:
        if not pipeline.accept_audience_input(event):
            return None
        _LOGGER.debug(
            "onsite_llm_request segment=%s transcript=%r",
            event.segment_id,
            event.text,
        )
        started_at = perf_counter()
        for attempt in range(2):
            try:
                turn = await pipeline.process_next_turn_async(cancellation)
            except AdapterConfigError:
                _LOGGER.exception("onsite_llm_permanent_failure")
                return None
            except OSError:
                if cancellation.cancelled:
                    return None
                if attempt == 1:
                    _LOGGER.exception("onsite_llm_failed")
                    return None
                _LOGGER.warning("onsite_llm_retry attempt=%d", attempt + 1)
                continue
            if turn is not None:
                _LOGGER.debug(
                    "onsite_llm_final turn=%s answer=%r latency_ms=%.1f",
                    turn.turn_id,
                    turn.answer_text,
                    (perf_counter() - started_at) * 1_000,
                )
            return turn
        return None

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
        if isinstance(gate, AsyncAsrSemanticGate):
            # Legacy batch ingress is synchronous; it must never start a live
            # network request on its event-loop caller.
            return AsrGateDecision.DISCARD
        decision = gate.evaluate(
            event.text,
            active_answer_excerpt=active_answer_excerpt,
            is_playing=is_playing,
        )
        _LOGGER.info(
            "asr_gate segment=%s decision=%s playing=%s transcript=%r",
            event.segment_id,
            decision,
            is_playing,
            event.text,
        )
        return decision

    async def gate_async(
        self,
        event: ASRAudienceEvent,
        *,
        active_answer_excerpt: str,
        is_playing: bool,
    ) -> AsrGateDecision:
        gate = self.asr_gate
        if gate is None:
            return AsrGateDecision.DISCARD if is_playing else AsrGateDecision.ACCEPT
        if isinstance(gate, AsyncAsrSemanticGate):
            return await gate.evaluate(
                event.text,
                active_answer_excerpt=active_answer_excerpt,
                is_playing=is_playing,
            )
        return gate.evaluate(
            event.text,
            active_answer_excerpt=active_answer_excerpt,
            is_playing=is_playing,
        )

    async def aclose(self) -> None:
        if self.llm is not None:
            await self.llm.aclose()

    async def prepare_replacement(
        self, stream: StreamKey, segment_id: SegmentId
    ) -> CancellationEpoch | None:
        return await self.begin_replacement(stream, segment_id)

    def synthesize(
        self, text: str, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None:
        started_at = perf_counter()
        _LOGGER.debug("onsite_tts_request text=%r voice=%s", text, self.voice)
        for attempt in range(2):
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
            except (MediaAdapterConfigError, OnsiteBridgeMediaError, wave.Error):
                _LOGGER.exception("onsite_tts_permanent_failure")
                return None
            except OSError:
                if cancellation.cancelled:
                    return None
                if attempt == 1:
                    _LOGGER.exception("onsite_tts_failed")
                    return None
                _LOGGER.warning("onsite_tts_retry attempt=%d", attempt + 1)
                continue
            _LOGGER.debug(
                "onsite_tts_complete pcm_bytes=%d latency_ms=%.1f",
                len(pcm16le),
                (perf_counter() - started_at) * 1_000,
            )
            return chunks
        return None

    async def speak_response(
        self,
        stream: StreamKey,
        text: str,
        epoch: CancellationEpoch,
        turn_id: str,
        output_started: ResponseOutputStarted,
    ) -> bool:
        """Synthesize one accepted Brain reply and deliver paced RTP to Sound.

        ``output_started`` commits the scheduler task only after the first
        valid RTP frame has crossed the output adapter.  This keeps context and
        caption effects behind the same first-frame boundary as playback.
        """
        cancellation = CancellationToken()
        try:
            stream_synthesis = await asyncio.to_thread(
                self.stream_synthesize, text, cancellation
            )
            if stream_synthesis is not None:
                return await self._speak_streaming_response(
                    stream,
                    epoch,
                    turn_id,
                    stream_synthesis,
                    cancellation,
                    output_started,
                )
            chunks = await asyncio.to_thread(self.synthesize, text, cancellation)
            output_epoch = await self.prepare_response_output(stream, epoch, turn_id)
            if not chunks or output_epoch is None:
                return False
            packetizer = TtsPcmRtpPacketizer(
                stream=stream, cancellation_epoch=output_epoch
            )
            packets = tuple(
                packet for chunk in chunks for packet in packetizer.push(chunk)
            )
            packets += packetizer.finish()
            if not packets:
                return False
            loop = asyncio.get_running_loop()
            deadline = loop.time()
            for index, packet in enumerate(packets):
                await self.output(stream, output_epoch, packet)
                if index == 0 and not output_started():
                    return False
                deadline += 0.02
                delay = deadline - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            await self.output_finished(stream, output_epoch)
        except asyncio.CancelledError:
            _ = cancellation.cancel(reason="response_tts_cancelled")
            raise
        else:
            return True

    async def _speak_streaming_response(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
        self,
        stream: StreamKey,
        epoch: CancellationEpoch,
        turn_id: str,
        synthesis: Iterator[Pcm16leChunk],
        cancellation: CancellationToken,
        output_started: ResponseOutputStarted,
    ) -> bool:
        """Start RTP playback as soon as SSE has produced one valid frame."""
        packetizer: TtsPcmRtpPacketizer | None = None
        output_epoch: CancellationEpoch | None = None
        committed = False
        loop = asyncio.get_running_loop()
        deadline = loop.time()

        async def emit(packets: tuple[bytes, ...]) -> bool:
            nonlocal committed, deadline
            if not packets:
                return True
            if output_epoch is None:
                return False
            for _index, packet in enumerate(packets):
                await self.output(stream, output_epoch, packet)
                if not committed:
                    committed = output_started()
                    if not committed:
                        return False
                deadline += 0.02
                delay = deadline - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            return True

        buffered = bytearray()
        pending_next: asyncio.Task[Pcm16leChunk | None] | None = None
        try:
            while not cancellation.cancelled:
                pending_next = asyncio.create_task(
                    asyncio.to_thread(next, synthesis, None)
                )
                chunk = await asyncio.shield(pending_next)
                pending_next = None
                if chunk is None:
                    break
                buffered.extend(chunk.data)
                if len(buffered) < L16_FRAME_BYTES:
                    continue
                if packetizer is None:
                    output_epoch = await self.prepare_response_output(
                        stream, epoch, turn_id
                    )
                    if output_epoch is None:
                        return False
                    packetizer = TtsPcmRtpPacketizer(stream, output_epoch)
                packets = packetizer.push(Pcm16leChunk(bytes(buffered)))
                buffered.clear()
                if not await emit(packets):
                    return False
            if cancellation.cancelled:
                return False
            if packetizer is None:
                if not buffered:
                    return False
                output_epoch = await self.prepare_response_output(
                    stream, epoch, turn_id
                )
                if output_epoch is None:
                    return False
                packetizer = TtsPcmRtpPacketizer(stream, output_epoch)
            packets = (
                packetizer.push(Pcm16leChunk(bytes(buffered))) + packetizer.finish()
            )
            if not await emit(packets):
                return False
        except asyncio.CancelledError:
            _ = cancellation.cancel(reason="response_tts_cancelled")
            if pending_next is not None and not pending_next.done():
                with suppress(asyncio.CancelledError, OSError, ValueError):
                    _ = await asyncio.shield(pending_next)
            raise
        finally:
            close = cast(
                "Callable[[], object] | None", getattr(synthesis, "close", None)
            )
            if close is not None:
                _ = await asyncio.to_thread(close)
        if not committed or output_epoch is None:
            return False
        await self.output_finished(stream, output_epoch)
        return True

    def stream_synthesize(
        self, text: str, cancellation: CancellationToken
    ) -> Iterator[Pcm16leChunk] | None:
        if getattr(
            self.tts, "capability", "final_only"
        ) != "streaming_sse" or not isinstance(self.tts, StreamingTtsAdapter):
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
    async def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult | None:
        return await self.bridge.answer_async(self.pipeline, event, cancellation)

    async def on_asr_final(self, stream: StreamKey, event: ASRAudienceEvent) -> bool:
        handler = self.bridge.asr_final_handler
        if handler is None:
            return False
        return await handler(stream, event)

    async def gate(
        self,
        event: ASRAudienceEvent,
        *,
        active_answer_excerpt: str,
        is_playing: bool,
    ) -> AsrGateDecision:
        return await self.bridge.gate_async(
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
    if config.tts_provider != "vllm_omni":
        raise OnsiteBridgeConfigError(field_name="tts_provider")

    if config.tts_endpoint is None or config.tts_model is None:
        raise OnsiteBridgeConfigError(field_name="tts_endpoint_or_tts_model")

    if (
        config.llm_provider != "openai_compatible"
        or config.llm_endpoint is None
        or config.llm_model is None
        or config.llm_api_key is None
        or config.llm_reasoning_dialect is None
    ):
        raise OnsiteBridgeConfigError(field_name="llm_provider_or_llm_configuration")

    if voice.strip() == "" or ref_audio.strip() == "" or ref_text.strip() == "":
        raise OnsiteBridgeConfigError(field_name="voice_reference")

    llm = AsyncOpenAICompatibleLLMRuntime(
        config.llm_endpoint,
        config.llm_model,
        config.llm_api_key,
        config.llm_reasoning_dialect,
        gate_model=config.llm_gate_model,
        brain_model=config.llm_brain_model,
        maintenance_model=config.llm_maintenance_model,
        timeout_seconds=120.0,
        deadlines=ProviderDeadlines(read_seconds=60.0, total_seconds=120.0),
        ca_path=config.tls_ca_path,
    )

    adapters = PipelineAdapters(
        mode_policy=AdaptiveAgentPolicy(),
        llm=cast("LLMAdapter", cast("object", llm)),
        retrieval=RetrievalFixtureProvider(()),
    )

    pipeline_config = PipelineConfig(1, "turn-onsite", "segment-onsite")

    def pipeline_factory() -> OrchestratorTurnPipeline:
        return OrchestratorTurnPipeline(adapters=adapters, config=pipeline_config)

    return OnsiteExplainerBridge(
        asr=_MicOwnedAsrAdapter(),
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
        asr_gate=None,
        llm=llm,
    )
