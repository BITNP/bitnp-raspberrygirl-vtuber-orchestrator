from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

from orchestrator.agent_state import AgentStateReducer, GateOutcome, StateEffect
from orchestrator.asr_semantic_gate import AsrGateDecision
from orchestrator.llm import CancellationToken
from orchestrator.observability import (
    OnsiteObservability,
    OnsiteStage,
    StageCorrelation,
    StageDetails,
)
from orchestrator.streaming_contracts import CancellationEpoch, SegmentId, StreamKey
from orchestrator.streaming_pipeline_actors import (
    ENDPOINTED_UTTERANCE_CAPACITY,
    TTS_CHUNK_CAPACITY,
    PipelineDropCounts,
)
from orchestrator.tts_rtp import Pcm16leChunk, TtsPcmRtpPacketizer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from orchestrator.pipeline_contracts import ASRAudienceEvent, TurnResult
    from orchestrator.streaming_endpoint import EndpointedUtterance


class OnsiteStages(Protocol):
    def transcribe(
        self, endpoint: EndpointedUtterance, cancellation: CancellationToken
    ) -> ASRAudienceEvent | None: ...

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult | None | Awaitable[TurnResult | None]: ...

    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None: ...

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None: ...

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None: ...


_RTP_FRAME_DURATION_SECONDS = 0.020

_PCM_FRAME_BYTES = 640

_LOGGER = logging.getLogger(__name__)

type GateCallable = (
    Callable[..., Awaitable[AsrGateDecision]] | Callable[..., AsrGateDecision]
)


def _next_chunk(stream: Iterator[Pcm16leChunk]) -> Pcm16leChunk | None:
    try:
        return next(stream)
    except StopIteration:
        return None


def _run_answer_and_synthesize(
    actor: OnsiteStreamActor,
    event: ASRAudienceEvent,
    correlation: StageCorrelation,
    cancellation: CancellationToken,
) -> _SynthesizedAnswer | None:
    """Keep legacy synchronous mock/media stages off the RTP event loop."""
    return asyncio.run(
        actor.answer_and_synthesize(
            event, correlation, cancellation
        )
    )


@dataclass(frozen=True, slots=True)
class _EndpointItem:
    epoch: CancellationEpoch

    endpoint: EndpointedUtterance


@dataclass(frozen=True, slots=True)
class _AnswerItem:
    epoch: CancellationEpoch

    event: ASRAudienceEvent

    correlation: StageCorrelation

    interrupts: bool = False

    state_epoch: int = 0


@dataclass(frozen=True, slots=True)
class _ChunkItem:
    epoch: CancellationEpoch

    chunk: Pcm16leChunk | None

    correlation: StageCorrelation

    packetizer: TtsPcmRtpPacketizer

    answer_excerpt: str

    state_epoch: int = 0


@dataclass(frozen=True, slots=True)
class _SynthesizedAnswer:
    chunks: tuple[Pcm16leChunk, ...] | None

    answer_text: str

    segment_id: SegmentId

    turn: TurnResult

    stream: Iterator[Pcm16leChunk] | None = None


@dataclass(slots=True)
class OnsiteStreamActor:
    stream: StreamKey

    epoch: CancellationEpoch

    stages: OnsiteStages

    observability: OnsiteObservability | None = None

    _endpoints: deque[_EndpointItem] = field(default_factory=deque)

    _answers: deque[_AnswerItem] = field(default_factory=deque)

    _chunks: deque[_ChunkItem] = field(default_factory=deque)

    _endpoint_wake: asyncio.Event = field(default_factory=asyncio.Event)

    _answer_wake: asyncio.Event = field(default_factory=asyncio.Event)

    _chunk_wake: asyncio.Event = field(default_factory=asyncio.Event)

    # TTS media is ordered data, not a latest-wins control message.  Producers
    # wait for RTP egress capacity instead of dropping audible PCM.
    _chunk_space: asyncio.Event = field(default_factory=asyncio.Event)

    _endpoint_task: asyncio.Task[None] | None = None

    _answer_task: asyncio.Task[None] | None = None

    _chunk_task: asyncio.Task[None] | None = None

    _drops: PipelineDropCounts = field(default_factory=PipelineDropCounts)

    _closed: bool = False

    _latest_correlation: StageCorrelation | None = None

    _active_cancellations: set[CancellationToken] = field(default_factory=set)

    # The answer lane is deliberately single-flight.  Retain its cancellation
    # handle separately so a newer *recognized* utterance cannot wait behind
    # an obsolete remote LLM/TTS request.  Endpoint arrivals alone are not
    # enough: they include VAD noise and blank ASR results.
    _active_answer_cancellation: CancellationToken | None = None

    _authorized_output_epochs: set[CancellationEpoch] = field(default_factory=set)

    _active_answer_excerpt: str = ""

    _is_playing: bool = False

    _output_epoch: CancellationEpoch | None = None

    _state: AgentStateReducer = field(default_factory=AgentStateReducer)

    def __post_init__(self) -> None:
        self._chunk_space.set()

    @property
    def drop_counts(self) -> PipelineDropCounts:
        return self._drops

    def submit(self, endpoint: EndpointedUtterance, epoch: CancellationEpoch) -> None:
        if self._closed or epoch != self.epoch:
            return

        self._record("endpoint", endpoint, epoch)

        self._latest_correlation = self._correlation(endpoint, epoch)

        self._append_endpoint(_EndpointItem(epoch, endpoint))

    def invalidate(self, next_epoch: CancellationEpoch) -> None:
        self.epoch = next_epoch

        self._closed = True

        correlation = self._latest_correlation

        if correlation is not None:
            self._record_correlation("cancellation", correlation, None)

        self._endpoints.clear()

        self._answers.clear()

        self._chunks.clear()
        self._chunk_space.set()

        for cancellation in tuple(self._active_cancellations):
            _ = cancellation.cancel(reason="stream_invalidated")

        if self._chunk_task is not None:
            _ = self._chunk_task.cancel()

    async def aclose(self) -> None:
        self.invalidate(CancellationEpoch(int(self.epoch) + 1))

        await self.wait_quiescent()

    async def wait_quiescent(self) -> None:
        while True:
            tasks = tuple(
                task
                for task in (self._endpoint_task, self._answer_task, self._chunk_task)
                if task is not None and not task.done()
            )

            if not tasks:
                return

            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task

    def _append_endpoint(self, item: _EndpointItem) -> None:
        correlation = self._correlation(item.endpoint, item.epoch)

        if len(self._endpoints) == ENDPOINTED_UTTERANCE_CAPACITY:
            _ = self._endpoints.popleft()

            self._drops = PipelineDropCounts(
                endpointed_utterances=self._drops.endpointed_utterances + 1,
                answer_turns=self._drops.answer_turns,
                tts_chunks=self._drops.tts_chunks,
            )

            self._record_details(
                "drop",
                correlation,
                StageDetails(drop_count=self._drops.endpointed_utterances),
            )

        self._endpoints.append(item)

        self._record_details(
            "queue",
            correlation,
            StageDetails(
                queue_name="endpointed_utterances", queue_depth=len(self._endpoints)
            ),
        )

        self._endpoint_wake.set()

        task = self._endpoint_task

        if task is None or task.done():
            self._endpoint_task = asyncio.create_task(self._run_endpoints())

    def _append_answer(self, item: _AnswerItem) -> None:
        active = self._active_answer_cancellation

        if active is not None:
            _ = active.cancel(reason="superseded_asr_final")

        # A later ASR final represents the user's latest completed input.  Do
        # not let an already queued answer begin after the active provider has
        # released its cancellation resource.
        dropped_answers = len(self._answers)
        self._answers.clear()

        if dropped_answers > 0:
            self._drops = PipelineDropCounts(
                endpointed_utterances=self._drops.endpointed_utterances,
                answer_turns=self._drops.answer_turns + dropped_answers,
                tts_chunks=self._drops.tts_chunks,
            )

            self._record_details(
                "drop",
                item.correlation,
                StageDetails(drop_count=self._drops.answer_turns),
            )

        self._answers.append(item)

        self._record_details(
            "queue",
            item.correlation,
            StageDetails(queue_name="answer_turns", queue_depth=len(self._answers)),
        )

        self._answer_wake.set()

        task = self._answer_task

        if task is None or task.done():
            self._answer_task = asyncio.create_task(self._run_answers())

    async def _append_chunk(self, item: _ChunkItem) -> bool:
        while len(self._chunks) >= TTS_CHUNK_CAPACITY:
            self._chunk_space.clear()
            _ = await self._chunk_space.wait()
            if self._closed:
                return False

        self._chunks.append(item)

        self._record_details(
            "queue",
            item.correlation,
            StageDetails(queue_name="tts_chunks", queue_depth=len(self._chunks)),
        )

        self._chunk_wake.set()

        task = self._chunk_task

        if task is None or task.done():
            self._chunk_task = asyncio.create_task(self._run_chunks())
        return True

    async def _run_endpoints(self) -> None:
        while self._endpoints:
            item = self._endpoints.popleft()

            started_at = time.perf_counter()

            cancellation = self._new_cancellation()

            try:
                event = await asyncio.to_thread(
                    self.stages.transcribe, item.endpoint, cancellation
                )

            finally:
                self._active_cancellations.discard(cancellation)

            latency_ms = (time.perf_counter() - started_at) * 1_000

            if item.epoch == self.epoch and event is not None:
                correlation = self._correlation(item.endpoint, item.epoch)
                self._record_correlation("asr_final", correlation, latency_ms)
                final_handler = cast(
                    "Callable[[StreamKey, ASRAudienceEvent], Awaitable[bool]] | None",
                    getattr(self.stages, "on_asr_final", None),
                )
                if final_handler is not None:
                    handled = await final_handler(self.stream, event)
                    if handled:
                        continue
                gate = cast(
                    "GateCallable | None",
                    getattr(self.stages, "gate", None),
                )
                if gate is None:
                    decision = AsrGateDecision.ACCEPT
                elif inspect.iscoroutinefunction(gate):
                    async_gate = cast(
                        "Callable[..., Awaitable[AsrGateDecision]]", gate
                    )
                    decision = await async_gate(
                        event,
                        active_answer_excerpt=self._active_answer_excerpt,
                        is_playing=self._is_playing,
                    )
                else:
                    # Test and mock stages can remain synchronous; only the
                    # live SDK path is required to stay on the event loop.
                    decision = await asyncio.to_thread(self._gate, event)
                if decision is AsrGateDecision.DISCARD:
                    self._record_details(
                        "drop", correlation, StageDetails(drop_count=1)
                    )
                    continue
                transition = self._state.gate(
                    GateOutcome.INTERRUPT
                    if decision is AsrGateDecision.INTERRUPT
                    else GateOutcome.ACCEPT
                )
                if StateEffect.START_REASONING not in transition.effects:
                    continue
                self._append_answer(
                    _AnswerItem(
                        item.epoch,
                        event,
                        correlation,
                        decision is AsrGateDecision.INTERRUPT,
                        transition.state.epoch,
                    )
                )

    async def _run_answers(self) -> None:  # noqa: C901, PLR0912
        while self._answers:
            item = self._answers.popleft()

            cancellation = self._new_cancellation()

            self._active_answer_cancellation = cancellation

            try:
                if inspect.iscoroutinefunction(self.stages.answer):
                    chunks = await self.answer_and_synthesize(
                        item.event, item.correlation, cancellation
                    )
                else:
                    chunks = await asyncio.to_thread(
                        _run_answer_and_synthesize,
                        self,
                        item.event,
                        item.correlation,
                        cancellation,
                    )

            finally:
                self._active_cancellations.discard(cancellation)

                if self._active_answer_cancellation is cancellation:
                    self._active_answer_cancellation = None

            if self._closed or cancellation.cancelled or chunks is None:
                continue

            transition = self._state.reasoning_complete(
                item.state_epoch, has_text=chunks.answer_text.strip() != ""
            )
            if StateEffect.START_TTS not in transition.effects:
                continue

            if chunks.stream is not None:
                await self._consume_stream(item, chunks, cancellation)
                continue

            output_epoch = item.epoch
            if item.interrupts:
                ready = self._state.audio_ready(item.state_epoch)
                if StateEffect.FLUSH_SOUND not in ready.effects:
                    continue
                replacement_epoch = await self._prepare_replacement(chunks.segment_id)
                if replacement_epoch is None:
                    _ = self._state.failed(item.state_epoch, audio_started=False)
                    continue
                acknowledged = self._state.flush_acknowledged(item.state_epoch)
                if StateEffect.EMIT_AUDIO not in acknowledged.effects:
                    continue
                output_epoch = replacement_epoch
            elif (
                StateEffect.EMIT_AUDIO
                not in self._state.audio_ready(item.state_epoch).effects
            ):
                continue

            # A packetizer has exactly one lifetime: one synthesized response.
            # Reusing it after finish would retain RTP sequence/timestamp state and
            # can never safely represent a later turn.
            packetizer = TtsPcmRtpPacketizer(self.stream, output_epoch)

            resolved_chunks = chunks.chunks
            if resolved_chunks is None:
                continue
            for chunk in resolved_chunks:
                appended = await self._append_chunk(
                    _ChunkItem(
                        output_epoch,
                        chunk,
                        item.correlation,
                        packetizer,
                        chunks.answer_text[:240],
                        item.state_epoch,
                    )
                )
                if not appended:
                    return

            _ = await self._append_chunk(
                _ChunkItem(
                    output_epoch,
                    None,
                    item.correlation,
                    packetizer,
                    chunks.answer_text[:240],
                    item.state_epoch,
                )
            )

    async def answer_and_synthesize(
        self,
        event: ASRAudienceEvent,
        correlation: StageCorrelation,
        cancellation: CancellationToken,
    ) -> _SynthesizedAnswer | None:
        answer_started_at = time.perf_counter()

        turn = self.stages.answer(event, cancellation)
        if inspect.isawaitable(turn):
            turn = await turn

        self._record_correlation(
            "answer", correlation, (time.perf_counter() - answer_started_at) * 1_000
        )

        if turn is None or turn.answer_text.strip() == "":
            return None

        streamer = cast(
            "Callable[[TurnResult, CancellationToken], Iterator[Pcm16leChunk] | None] | None",  # noqa: E501
            getattr(self.stages, "stream_synthesize", None),
        )
        if streamer is not None:
            stream = await asyncio.to_thread(streamer, turn, cancellation)
            if stream is not None:
                return _SynthesizedAnswer(
                    None,
                    turn.answer_text,
                    SegmentId(str(turn.segment_id)),
                    turn,
                    stream,
                )

        tts_started_at = time.perf_counter()

        # TTS adapters use synchronous HTTP clients.  The answer stage is async
        # in production, so calling the adapter directly here would otherwise
        # stall the transport loop and starve websocket keepalive ping/pong.
        chunks = await asyncio.to_thread(self.stages.synthesize, turn, cancellation)

        self._record_correlation(
            "tts", correlation, (time.perf_counter() - tts_started_at) * 1_000
        )

        if chunks is None or not chunks:
            return None

        self.stages.complete(turn, chunks)

        return _SynthesizedAnswer(
            chunks, turn.answer_text, SegmentId(str(turn.segment_id)), turn
        )

    async def _consume_stream(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        item: _AnswerItem,
        answer: _SynthesizedAnswer,
        cancellation: CancellationToken,
    ) -> None:
        stream = answer.stream
        if stream is None:
            return
        buffered: list[Pcm16leChunk] = []
        buffered_bytes = 0
        output_epoch = item.epoch
        packetizer: TtsPcmRtpPacketizer | None = None
        total_bytes = 0
        try:
            while not cancellation.cancelled:
                chunk = await asyncio.to_thread(_next_chunk, stream)
                if chunk is None:
                    break
                total_bytes += len(chunk.data)
                if packetizer is None:
                    buffered.append(chunk)
                    buffered_bytes += len(chunk.data)
                    if buffered_bytes < _PCM_FRAME_BYTES:
                        continue
                    if item.interrupts:
                        ready = self._state.audio_ready(item.state_epoch)
                        if StateEffect.FLUSH_SOUND not in ready.effects:
                            return
                        replacement_epoch = await self._prepare_replacement(
                            answer.segment_id
                        )
                        if replacement_epoch is None:
                            _ = self._state.failed(
                                item.state_epoch, audio_started=False
                            )
                            return
                        acknowledged = self._state.flush_acknowledged(item.state_epoch)
                        if StateEffect.EMIT_AUDIO not in acknowledged.effects:
                            return
                        output_epoch = replacement_epoch
                    elif (
                        StateEffect.EMIT_AUDIO
                        not in self._state.audio_ready(item.state_epoch).effects
                    ):
                        return
                    packetizer = TtsPcmRtpPacketizer(self.stream, output_epoch)
                    for buffered_chunk in buffered:
                        if not await self._append_chunk(
                            _ChunkItem(
                                output_epoch,
                                buffered_chunk,
                                item.correlation,
                                packetizer,
                                answer.answer_text[:240],
                                item.state_epoch,
                            )
                        ):
                            return
                    buffered.clear()
                    continue
                if not await self._append_chunk(
                    _ChunkItem(
                        output_epoch,
                        chunk,
                        item.correlation,
                        packetizer,
                        answer.answer_text[:240],
                        item.state_epoch,
                    )
                ):
                    return
        except (OSError, ValueError):
            # Provider failures are terminal for this turn.  Do not synthesize
            # a recovery response or convert the failed stream into a normal
            # completion; retain only the structured error log.
            _LOGGER.exception(
                "onsite_tts_stream_failure stream=%s segment=%s",
                self.stream,
                answer.segment_id,
            )
            self._chunks = deque(
                queued
                for queued in self._chunks
                if queued.correlation != item.correlation
            )
            self._chunk_space.set()
            self._record_correlation("tts_failure", item.correlation, None)
            _ = self._state.failed(
                item.state_epoch, audio_started=packetizer is not None
            )
            return
        if cancellation.cancelled:
            return
        if packetizer is None:
            if item.interrupts:
                ready = self._state.audio_ready(item.state_epoch)
                if StateEffect.FLUSH_SOUND not in ready.effects:
                    return
                replacement_epoch = await self._prepare_replacement(answer.segment_id)
                if replacement_epoch is None:
                    _ = self._state.failed(item.state_epoch, audio_started=False)
                    return
                acknowledged = self._state.flush_acknowledged(item.state_epoch)
                if StateEffect.EMIT_AUDIO not in acknowledged.effects:
                    return
                output_epoch = replacement_epoch
            elif (
                StateEffect.EMIT_AUDIO
                not in self._state.audio_ready(item.state_epoch).effects
            ):
                return
            packetizer = TtsPcmRtpPacketizer(self.stream, output_epoch)
            for buffered_chunk in buffered:
                if not await self._append_chunk(
                    _ChunkItem(
                        output_epoch,
                        buffered_chunk,
                        item.correlation,
                        packetizer,
                        answer.answer_text[:240],
                        item.state_epoch,
                    )
                ):
                    return
        if not await self._append_chunk(
            _ChunkItem(
                output_epoch,
                None,
                item.correlation,
                packetizer,
                answer.answer_text[:240],
                item.state_epoch,
            )
        ):
            return
        completer = cast(
            "Callable[[TurnResult, int], None] | None",
            getattr(self.stages, "complete_stream", None),
        )
        if completer is not None:
            completer(answer.turn, total_bytes)

    async def _prepare_replacement(
        self, segment_id: SegmentId
    ) -> CancellationEpoch | None:
        callback = cast(
            "Callable[[StreamKey, SegmentId], Awaitable[CancellationEpoch | None]] | None",  # noqa: E501
            getattr(self.stages, "prepare_replacement", None),
        )
        if callback is None or self._output_epoch is None:
            return None
        return await callback(self.stream, segment_id)

    def _new_cancellation(self) -> CancellationToken:
        cancellation = CancellationToken()

        self._active_cancellations.add(cancellation)

        return cancellation

    async def _run_chunks(self) -> None:
        loop = asyncio.get_running_loop()
        next_packet_deadline = loop.time()

        while self._chunks:
            item = self._chunks.popleft()
            self._chunk_space.set()

            if self._closed:
                continue

            packets = (
                item.packetizer.finish()
                if item.chunk is None
                else item.packetizer.push(item.chunk)
            )

            for packet in packets:
                if not self._closed:
                    # Anchor every frame to one monotonic media timeline.  A
                    # relative 20 ms sleep after each send includes Python and
                    # UDP handling time, so a long answer slowly runs behind
                    # the 16 kHz playback clock and drains Sound's reserve.
                    wait_seconds = next_packet_deadline - loop.time()
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
                    if not self._authorize_output(item.epoch):
                        continue
                    await self.stages.output(self.stream, item.epoch, packet)
                    self._is_playing = True
                    self._active_answer_excerpt = item.answer_excerpt
                    self._output_epoch = item.epoch

                    self._record_correlation("rtp_egress", item.correlation, None)

                    # RTP is a real-time transport boundary, not a bulk UDP
                    # transfer.  Sending an entire synthesized answer in one
                    # event-loop turn overflows Sound's socket/playback queues
                    # and truncates the audible tail.  Cancellation cancels the
                    # chunk task, so this pacing never delays barge-in.
                    next_packet_deadline += _RTP_FRAME_DURATION_SECONDS

            if item.chunk is None and not self._closed:
                finisher = getattr(self.stages, "finish_output", None)
                if finisher is not None:
                    await finisher(self.stream, item.epoch)
                _ = self._state.audio_finished(item.state_epoch)
                self._active_answer_excerpt = ""
                self._is_playing = False
                if self._output_epoch == item.epoch:
                    self._output_epoch = None

    def _gate(self, event: ASRAudienceEvent) -> AsrGateDecision:
        gate = cast(
            "Callable[..., AsrGateDecision] | None",
            getattr(self.stages, "gate", None),
        )
        if gate is None:
            # Test-only/legacy stages do not have a semantic gate.  Production
            # composition always installs one; while audio is active we still
            # fail closed to avoid accidental barge-in.
            return (
                AsrGateDecision.DISCARD if self._is_playing else AsrGateDecision.ACCEPT
            )
        try:
            return gate(
                event,
                active_answer_excerpt=self._active_answer_excerpt,
                is_playing=self._is_playing,
            )
        except (OSError, TimeoutError, ValueError):
            return AsrGateDecision.DISCARD

    def _authorize_output(self, epoch: CancellationEpoch) -> bool:
        if epoch in self._authorized_output_epochs:
            return True
        authorizer = cast(
            "Callable[[StreamKey, CancellationEpoch], bool] | None",
            getattr(self.stages, "authorize_output", None),
        )
        if authorizer is not None and not authorizer(self.stream, epoch):
            return False
        self._authorized_output_epochs.add(epoch)
        return True

    def _correlation(
        self, endpoint: EndpointedUtterance, epoch: CancellationEpoch
    ) -> StageCorrelation:
        observability = self.observability

        if observability is None:
            return StageCorrelation("", "", 0)

        correlation = observability.correlation(
            endpoint.stream, str(endpoint.turn_id), str(endpoint.segment_id), epoch
        )

        if correlation is None:
            message = "missing source envelope correlation"

            raise RuntimeError(message)

        return correlation

    def _record(
        self,
        stage: OnsiteStage,
        endpoint: EndpointedUtterance,
        epoch: CancellationEpoch,
    ) -> None:
        self._record_correlation(stage, self._correlation(endpoint, epoch), None)

    def _record_correlation(
        self,
        stage: OnsiteStage,
        correlation: StageCorrelation,
        latency_ms: float | None,
    ) -> None:
        observability = self.observability

        if observability is not None:
            observability.record(
                stage, correlation, StageDetails(latency_ms=latency_ms)
            )

    def _record_details(
        self, stage: OnsiteStage, correlation: StageCorrelation, details: StageDetails
    ) -> None:
        observability = self.observability

        if observability is not None:
            observability.record(stage, correlation, details)
