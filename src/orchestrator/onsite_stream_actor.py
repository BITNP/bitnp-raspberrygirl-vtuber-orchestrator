
from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

from orchestrator.llm import CancellationToken
from orchestrator.observability import (
    OnsiteObservability,
    OnsiteStage,
    StageCorrelation,
    StageDetails,
)
from orchestrator.streaming_contracts import CancellationEpoch, StreamKey
from orchestrator.streaming_pipeline_actors import (
    ANSWER_TURN_CAPACITY,
    ENDPOINTED_UTTERANCE_CAPACITY,
    TTS_CHUNK_CAPACITY,
    PipelineDropCounts,
)
from orchestrator.tts_rtp import Pcm16leChunk, TtsPcmRtpPacketizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.pipeline_contracts import ASRAudienceEvent, TurnResult
    from orchestrator.streaming_endpoint import EndpointedUtterance


class OnsiteStages(Protocol):

    def transcribe(
        self, endpoint: EndpointedUtterance, cancellation: CancellationToken
    ) -> ASRAudienceEvent | None:
        ...

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult | None:
        ...

    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None:
        ...

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:
        ...

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        ...

    async def finish_output(self, stream: StreamKey, epoch: CancellationEpoch) -> None:
        ...


_RTP_FRAME_DURATION_SECONDS = 0.020


@dataclass(frozen=True, slots=True)
class _EndpointItem:

    epoch: CancellationEpoch

    endpoint: EndpointedUtterance


@dataclass(frozen=True, slots=True)
class _AnswerItem:

    epoch: CancellationEpoch

    event: ASRAudienceEvent

    correlation: StageCorrelation


@dataclass(frozen=True, slots=True)
class _ChunkItem:

    epoch: CancellationEpoch

    chunk: Pcm16leChunk | None

    correlation: StageCorrelation

    packetizer: TtsPcmRtpPacketizer


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

    _endpoint_task: asyncio.Task[None] | None = None

    _answer_task: asyncio.Task[None] | None = None

    _chunk_task: asyncio.Task[None] | None = None

    _drops: PipelineDropCounts = field(default_factory=PipelineDropCounts)

    _closed: bool = False

    _latest_correlation: StageCorrelation | None = None

    _active_cancellations: set[CancellationToken] = field(default_factory=set)

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
        if len(self._answers) == ANSWER_TURN_CAPACITY:
            _ = self._answers.popleft()

            self._drops = PipelineDropCounts(
                endpointed_utterances=self._drops.endpointed_utterances,
                answer_turns=self._drops.answer_turns + 1,
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

    def _append_chunk(self, item: _ChunkItem) -> None:
        if len(self._chunks) == TTS_CHUNK_CAPACITY:
            _ = self._chunks.popleft()

            self._drops = PipelineDropCounts(
                endpointed_utterances=self._drops.endpointed_utterances,
                answer_turns=self._drops.answer_turns,
                tts_chunks=self._drops.tts_chunks + 1,
            )

            self._record_details(
                "drop",
                item.correlation,
                StageDetails(drop_count=self._drops.tts_chunks),
            )

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

                authorizer = cast(
                    "Callable[[StreamKey, CancellationEpoch], bool] | None",
                    getattr(self.stages, "authorize_output", None),
                )

                if authorizer is not None:
                    authorized = authorizer(self.stream, item.epoch)

                    if authorized is False:
                        continue

                self._record_correlation("asr_final", correlation, latency_ms)

                self._append_answer(_AnswerItem(item.epoch, event, correlation))

    async def _run_answers(self) -> None:
        while self._answers:
            item = self._answers.popleft()

            cancellation = self._new_cancellation()

            try:
                chunks = await asyncio.to_thread(
                    self._answer_and_synthesize,
                    item.event,
                    item.correlation,
                    cancellation,
                )

            finally:
                self._active_cancellations.discard(cancellation)

            if self._closed or chunks is None:
                continue

            # A packetizer has exactly one lifetime: one synthesized response.
            # Reusing it after finish would retain RTP sequence/timestamp state and
            # can never safely represent a later turn.
            packetizer = TtsPcmRtpPacketizer(self.stream, item.epoch)

            for chunk in chunks:
                self._append_chunk(
                    _ChunkItem(item.epoch, chunk, item.correlation, packetizer)
                )

            self._append_chunk(
                _ChunkItem(item.epoch, None, item.correlation, packetizer)
            )

    def _answer_and_synthesize(
        self,
        event: ASRAudienceEvent,
        correlation: StageCorrelation,
        cancellation: CancellationToken,
    ) -> tuple[Pcm16leChunk, ...] | None:
        answer_started_at = time.perf_counter()

        turn = self.stages.answer(event, cancellation)

        self._record_correlation(
            "answer", correlation, (time.perf_counter() - answer_started_at) * 1_000
        )

        if turn is None or turn.answer_text.strip() == "":
            return None

        tts_started_at = time.perf_counter()

        chunks = self.stages.synthesize(turn, cancellation)

        self._record_correlation(
            "tts", correlation, (time.perf_counter() - tts_started_at) * 1_000
        )

        if chunks is None or not chunks:
            return None

        self.stages.complete(turn, chunks)

        return chunks

    def _new_cancellation(self) -> CancellationToken:
        cancellation = CancellationToken()

        self._active_cancellations.add(cancellation)

        return cancellation

    async def _run_chunks(self) -> None:
        while self._chunks:
            item = self._chunks.popleft()

            if self._closed:
                continue

            packets = (
                item.packetizer.finish()
                if item.chunk is None
                else item.packetizer.push(item.chunk)
            )

            for packet in packets:
                if not self._closed:
                    await self.stages.output(self.stream, item.epoch, packet)

                    self._record_correlation("rtp_egress", item.correlation, None)

                    # RTP is a real-time transport boundary, not a bulk UDP
                    # transfer.  Sending an entire synthesized answer in one
                    # event-loop turn overflows Sound's socket/playback queues
                    # and truncates the audible tail.  Cancellation cancels the
                    # chunk task, so this pacing never delays barge-in.
                    await asyncio.sleep(_RTP_FRAME_DURATION_SECONDS)

            if item.chunk is None and not self._closed:
                finisher = getattr(self.stages, "finish_output", None)
                if finisher is not None:
                    await finisher(self.stream, item.epoch)

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
