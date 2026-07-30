"""模块契约说明.

职责: 提供 orchestrator.onsite_stream_actor
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

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
    from orchestrator.pipeline_contracts import ASRAudienceEvent, TurnResult
    from orchestrator.streaming_endpoint import EndpointedUtterance


class OnsiteStages(Protocol):
    """类契约说明.

    职责: 声明 OnsiteStages
    协议接口,约束实现方必须提供的行为。
    契约: 方法: transcribe、answer、synthesize
    、complete、output。
    """

    def transcribe(
        self, endpoint: EndpointedUtterance, cancellation: CancellationToken
    ) -> ASRAudienceEvent | None:
        """函数契约说明.

        功能: 执行 transcribe 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 endpoint:
        EndpointedUtterance。 必填。
        cancellation: CancellationToken。
        必填。
        契约: 同步调用。 返回 `ASRAudienceEvent |
        None`。
        """
        ...

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult | None:
        """函数契约说明.

        功能: 执行 answer 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 event:
        ASRAudienceEvent。 必填。
        cancellation: CancellationToken。
        必填。
        契约: 同步调用。 返回 `TurnResult |
        None`。
        """
        ...

    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None:
        """函数契约说明.

        功能: 执行 synthesize 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。 cancellation:
        CancellationToken。 必填。
        契约: 同步调用。 返回
        `tuple[Pcm16leChunk, ...] |
        None`。
        """
        ...

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:
        """函数契约说明.

        功能: 执行 complete 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。 chunks:
        tuple[Pcm16leChunk, ...]。 必填。
        契约: 同步调用。 返回 `None`。
        """
        ...

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        """函数契约说明.

        功能: 执行 output 的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 epoch:
        CancellationEpoch。 必填。 packet:
        bytes。 必填。
        契约: 异步调用。 返回 `None`。
        """
        ...


@dataclass(frozen=True, slots=True)
class _EndpointItem:
    """类契约说明.

    职责: 保存 _EndpointItem
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: epoch、endpoint。
    """

    epoch: CancellationEpoch

    endpoint: EndpointedUtterance


@dataclass(frozen=True, slots=True)
class _AnswerItem:
    """类契约说明.

    职责: 保存 _AnswerItem
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: epoch、event、correlation。
    """

    epoch: CancellationEpoch

    event: ASRAudienceEvent

    correlation: StageCorrelation


@dataclass(frozen=True, slots=True)
class _ChunkItem:
    """类契约说明.

    职责: 保存 _ChunkItem
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: epoch、chunk、correlation。
    """

    epoch: CancellationEpoch

    chunk: Pcm16leChunk | None

    correlation: StageCorrelation


@dataclass(slots=True)
class OnsiteStreamActor:
    """类契约说明.

    职责: 保存 OnsiteStreamActor
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、epoch、stages、observab
    ility、_endpoints、_answers。 方法: __pos
    t_init__、drop_counts、submit、invalida
    te、aclose、wait_quiescent。
    """

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

    _packetizer: TtsPcmRtpPacketizer = field(init=False)

    _latest_correlation: StageCorrelation | None = None

    _active_cancellations: set[CancellationToken] = field(default_factory=set)

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化 OnsiteStreamActor
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self._packetizer = TtsPcmRtpPacketizer(self.stream, self.epoch)

    @property
    def drop_counts(self) -> PipelineDropCounts:
        """函数契约说明.

        功能: 执行 drop_counts
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `PipelineDropCounts`。
        """
        return self._drops

    def submit(self, endpoint: EndpointedUtterance, epoch: CancellationEpoch) -> None:
        """函数契约说明.

        功能: 执行 submit 的同步逻辑,并协调 _record,
        _correlation, _append_endpoint,
        _EndpointItem。
        参数: self 表示当前实例。 endpoint:
        EndpointedUtterance。 必填。 epoch:
        CancellationEpoch。 必填。
        契约: 同步调用。 返回 `None`。
        """
        if self._closed or epoch != self.epoch:
            return

        self._record("endpoint", endpoint, epoch)

        self._latest_correlation = self._correlation(endpoint, epoch)

        self._append_endpoint(_EndpointItem(epoch, endpoint))

    def invalidate(self, next_epoch: CancellationEpoch) -> None:
        """函数契约说明.

        功能: 执行 invalidate 的同步逻辑,并协调
        cancel, clear, tuple,
        _record_correlation。
        参数: self 表示当前实例。 next_epoch:
        CancellationEpoch。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self.epoch = next_epoch

        self._closed = True

        self._packetizer.cancel()

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
        """函数契约说明.

        功能: 执行 aclose 的异步逻辑,并协调
        invalidate, CancellationEpoch,
        wait_quiescent, int。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        self.invalidate(CancellationEpoch(int(self.epoch) + 1))

        await self.wait_quiescent()

    async def wait_quiescent(self) -> None:
        """函数契约说明.

        功能: 执行 wait_quiescent 的异步逻辑,并协调
        tuple, suppress, done。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
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
        """函数契约说明.

        功能: 执行 _append_endpoint
        的同步逻辑,并协调 _correlation, append,
        _record_details, set。
        参数: self 表示当前实例。 item:
        _EndpointItem。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 _append_answer 的同步逻辑,并协调
        append, _record_details, set,
        len。
        参数: self 表示当前实例。 item:
        _AnswerItem。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 _append_chunk 的同步逻辑,并协调
        append, _record_details, set,
        len。
        参数: self 表示当前实例。 item:
        _ChunkItem。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 _run_endpoints 的异步逻辑,并协调
        popleft, perf_counter,
        _new_cancellation, discard。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
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

                self._append_answer(_AnswerItem(item.epoch, event, correlation))

    async def _run_answers(self) -> None:
        """函数契约说明.

        功能: 执行 _run_answers 的异步逻辑,并协调
        popleft, _new_cancellation,
        _append_chunk, discard。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
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

            if item.epoch != self.epoch or chunks is None:
                continue

            for chunk in chunks:
                self._append_chunk(_ChunkItem(item.epoch, chunk, item.correlation))

            self._append_chunk(_ChunkItem(item.epoch, None, item.correlation))

    def _answer_and_synthesize(
        self,
        event: ASRAudienceEvent,
        correlation: StageCorrelation,
        cancellation: CancellationToken,
    ) -> tuple[Pcm16leChunk, ...] | None:
        """函数契约说明.

        功能: 执行 _answer_and_synthesize
        的同步逻辑,并协调 perf_counter, answer,
        _record_correlation, synthesize。
        参数: self 表示当前实例。 event:
        ASRAudienceEvent。 必填。
        correlation: StageCorrelation。
        必填。 cancellation:
        CancellationToken。 必填。
        契约: 同步调用。 返回
        `tuple[Pcm16leChunk, ...] |
        None`。
        """
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
        """函数契约说明.

        功能: 执行 _new_cancellation
        的同步逻辑,并协调 CancellationToken,
        add。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `CancellationToken`。
        """
        cancellation = CancellationToken()

        self._active_cancellations.add(cancellation)

        return cancellation

    async def _run_chunks(self) -> None:
        """函数契约说明.

        功能: 执行 _run_chunks 的异步逻辑,并协调
        popleft, finish, push,
        _record_correlation。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        while self._chunks:
            item = self._chunks.popleft()

            if item.epoch != self.epoch:
                continue

            packets = (
                self._packetizer.finish()
                if item.chunk is None
                else self._packetizer.push(item.chunk)
            )

            for packet in packets:
                if item.epoch == self.epoch:
                    await self.stages.output(self.stream, item.epoch, packet)

                    self._record_correlation("rtp_egress", item.correlation, None)

    def _correlation(
        self, endpoint: EndpointedUtterance, epoch: CancellationEpoch
    ) -> StageCorrelation:
        """函数契约说明.

        功能: 执行 _correlation 的同步逻辑,并协调
        correlation, StageCorrelation,
        str, RuntimeError。
        参数: self 表示当前实例。 endpoint:
        EndpointedUtterance。 必填。 epoch:
        CancellationEpoch。 必填。
        契约: 同步调用。 返回 `StageCorrelation`。
        可能抛出 RuntimeError。
        """
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
        """函数契约说明.

        功能: 执行 _record 的同步逻辑,并协调
        _record_correlation,
        _correlation。
        参数: self 表示当前实例。 stage:
        OnsiteStage。 必填。 endpoint:
        EndpointedUtterance。 必填。 epoch:
        CancellationEpoch。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._record_correlation(stage, self._correlation(endpoint, epoch), None)

    def _record_correlation(
        self,
        stage: OnsiteStage,
        correlation: StageCorrelation,
        latency_ms: float | None,
    ) -> None:
        """函数契约说明.

        功能: 执行 _record_correlation
        的同步逻辑,并协调 record, StageDetails。
        参数: self 表示当前实例。 stage:
        OnsiteStage。 必填。 correlation:
        StageCorrelation。 必填。
        latency_ms: float | None。 必填。
        契约: 同步调用。 返回 `None`。
        """
        observability = self.observability

        if observability is not None:
            observability.record(
                stage, correlation, StageDetails(latency_ms=latency_ms)
            )

    def _record_details(
        self, stage: OnsiteStage, correlation: StageCorrelation, details: StageDetails
    ) -> None:
        """函数契约说明.

        功能: 执行 _record_details 的同步逻辑,并协调
        record。
        参数: self 表示当前实例。 stage:
        OnsiteStage。 必填。 correlation:
        StageCorrelation。 必填。 details:
        StageDetails。 必填。
        契约: 同步调用。 返回 `None`。
        """
        observability = self.observability

        if observability is not None:
            observability.record(stage, correlation, details)
