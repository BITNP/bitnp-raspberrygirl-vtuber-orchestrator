"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.ids import SegmentId as PipelineSegmentId
from orchestrator.ids import TurnId as PipelineTurnId
from orchestrator.onsite_stream_actor import OnsiteStreamActor
from orchestrator.pipeline_contracts import ASRAudienceEvent, TurnResult
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    SegmentId,
    StreamKey,
    TurnId,
)
from orchestrator.streaming_endpoint import EndpointedUtterance, EndpointReason
from orchestrator.tts_rtp import Pcm16leChunk

if TYPE_CHECKING:
    from orchestrator.llm import CancellationToken


@dataclass(slots=True)
class _Stages:
    """类契约说明.

    职责: 保存 _Stages 不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    asr_started、release_asr、outputs。 方法:
    transcribe、answer、synthesize、complet
    e、output。
    """

    asr_started: asyncio.Event = field(default_factory=asyncio.Event)

    release_asr: asyncio.Event = field(default_factory=asyncio.Event)

    outputs: list[tuple[StreamKey, CancellationEpoch, bytes]] = field(
        default_factory=list
    )

    def transcribe(
        self, endpoint: EndpointedUtterance, cancellation: CancellationToken
    ) -> ASRAudienceEvent:
        """函数契约说明.

        功能: 执行 transcribe 的同步逻辑,并协调 set,
        ASRAudienceEvent。
        参数: self 表示当前实例。 endpoint:
        EndpointedUtterance。 必填。
        cancellation: CancellationToken。
        必填。
        契约: 同步调用。 返回 `ASRAudienceEvent`。
        """

        _ = (endpoint, cancellation)

        _ = self.asr_started.set()

        return ASRAudienceEvent("question", 0, "segment", 1)

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult:
        """函数契约说明.

        功能: 执行 answer 的同步逻辑,并协调
        TurnResult, PipelineTurnId,
        PipelineSegmentId。
        参数: self 表示当前实例。 event:
        ASRAudienceEvent。 必填。
        cancellation: CancellationToken。
        必填。
        契约: 同步调用。 返回 `TurnResult`。
        """

        _ = (event, cancellation)

        return TurnResult(
            PipelineTurnId("turn"),
            PipelineSegmentId("segment"),
            "answer",
            used_fallback=False,
        )

    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...]:
        """函数契约说明.

        功能: 执行 synthesize 的同步逻辑,并协调
        Pcm16leChunk。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。 cancellation:
        CancellationToken。 必填。
        契约: 同步调用。 返回
        `tuple[Pcm16leChunk, ...]`。
        """

        _ = (turn, cancellation)

        return (Pcm16leChunk(b"\x10\x20" * 320),)

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:
        """函数契约说明.

        功能: 执行 complete 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。 chunks:
        tuple[Pcm16leChunk, ...]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = (turn, chunks)

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        """函数契约说明.

        功能: 执行 output 的异步逻辑,并协调 append。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 epoch:
        CancellationEpoch。 必填。 packet:
        bytes。 必填。
        契约: 异步调用。 返回 `None`。
        """

        self.outputs.append((stream, epoch, packet))


def test_actor_tags_generated_rtp_with_admission_epoch() -> None:
    """函数契约说明.

    功能: 验证 actor tags generated rtp with
    admission epoch 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_epoch_proof())


async def _epoch_proof() -> None:
    # Given: one actor with pure deterministic provider stages.

    """函数契约说明.

    功能: 执行 _epoch_proof 的异步逻辑,并协调
    _Stages, StreamKey,
    OnsiteStreamActor, submit。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    stages = _Stages()

    stream = StreamKey("session", "stream")

    actor = OnsiteStreamActor(stream, CancellationEpoch(7), stages)

    # When: an endpointed utterance enters the staged pipeline.

    actor.submit(_endpoint(stream), CancellationEpoch(7))

    await actor.wait_quiescent()

    # Then: every emitted packet carries the admission epoch.

    assert [epoch for _, epoch, _ in stages.outputs] == [CancellationEpoch(7)]


def test_actor_invalidation_clears_queued_work_and_suppresses_output() -> None:
    """函数契约说明.

    功能: 验证 actor invalidation clears
    queued work and suppresses output
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_invalidation_proof())


async def _invalidation_proof() -> None:
    # Given: an actor whose first provider stage has accepted work.

    """函数契约说明.

    功能: 执行 _invalidation_proof 的异步逻辑,并协调
    _Stages, StreamKey,
    OnsiteStreamActor, submit。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    stages = _Stages()

    stream = StreamKey("session", "stream")

    actor = OnsiteStreamActor(stream, CancellationEpoch(0), stages)

    actor.submit(_endpoint(stream), CancellationEpoch(0))

    _ = await stages.asr_started.wait()

    # When: the authenticated route invalidates its current epoch.

    actor.invalidate(CancellationEpoch(1))

    await actor.wait_quiescent()

    # Then: no packet from the retired epoch can reach the callback.

    assert stages.outputs == []


@dataclass(slots=True)
class _BlockingStages:
    """类契约说明.

    职责: 保存 _BlockingStages
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: asr_started、provider_cancell
    ed、provider_completed、release、output
    s。 方法: transcribe、answer、synthesize、
    complete、output、_cancel_provider。
    """

    asr_started: threading.Event = field(default_factory=threading.Event)

    provider_cancelled: threading.Event = field(default_factory=threading.Event)

    provider_completed: threading.Event = field(default_factory=threading.Event)

    release: threading.Event = field(default_factory=threading.Event)

    outputs: list[tuple[StreamKey, CancellationEpoch, bytes]] = field(
        default_factory=list
    )

    def transcribe(
        self,
        endpoint: EndpointedUtterance,
        cancellation: CancellationToken | None = None,
    ) -> ASRAudienceEvent:
        """函数契约说明.

        功能: 执行 transcribe 的同步逻辑,并协调 set,
        wait, ASRAudienceEvent, bind。
        参数: self 表示当前实例。 endpoint:
        EndpointedUtterance。 必填。
        cancellation: CancellationToken
        | None。 可省略。
        契约: 同步调用。 返回 `ASRAudienceEvent`。
        """

        _ = endpoint

        release = None

        if cancellation is not None:
            release = cancellation.bind(self._cancel_provider)

        _ = self.asr_started.set()

        _ = self.release.wait()

        if release is not None:
            release()

        _ = self.provider_completed.set()

        return ASRAudienceEvent("stale question", 0, "segment", 1)

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult:
        """函数契约说明.

        功能: 执行 answer 的同步逻辑,并协调
        TurnResult, PipelineTurnId,
        PipelineSegmentId。
        参数: self 表示当前实例。 event:
        ASRAudienceEvent。 必填。
        cancellation: CancellationToken。
        必填。
        契约: 同步调用。 返回 `TurnResult`。
        """

        _ = (event, cancellation)

        return TurnResult(
            PipelineTurnId("stale-turn"),
            PipelineSegmentId("stale-segment"),
            "stale answer",
            used_fallback=False,
        )

    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...]:
        """函数契约说明.

        功能: 执行 synthesize 的同步逻辑,并协调
        Pcm16leChunk。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。 cancellation:
        CancellationToken。 必填。
        契约: 同步调用。 返回
        `tuple[Pcm16leChunk, ...]`。
        """

        _ = (turn, cancellation)

        return (Pcm16leChunk(b"\x10\x20" * 320),)

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:
        """函数契约说明.

        功能: 执行 complete 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。 chunks:
        tuple[Pcm16leChunk, ...]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = (turn, chunks)

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        """函数契约说明.

        功能: 执行 output 的异步逻辑,并协调 append。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 epoch:
        CancellationEpoch。 必填。 packet:
        bytes。 必填。
        契约: 异步调用。 返回 `None`。
        """

        self.outputs.append((stream, epoch, packet))

    def _cancel_provider(self) -> None:
        """函数契约说明.

        功能: 执行 _cancel_provider
        的同步逻辑,并协调 set。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        _ = self.provider_cancelled.set()

        _ = self.release.set()


@dataclass(slots=True)
class _BackpressureStages:
    """类契约说明.

    职责: 保存 _BackpressureStages
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: asr_started、all_asr_finished
    、answer_started、all_answers_finished
    、release_answer、output_started。 方法:
    transcribe、answer、synthesize、complet
    e、output。
    """

    asr_started: threading.Event = field(default_factory=threading.Event)

    all_asr_finished: threading.Event = field(default_factory=threading.Event)

    answer_started: threading.Event = field(default_factory=threading.Event)

    all_answers_finished: threading.Event = field(default_factory=threading.Event)

    release_answer: threading.Event = field(default_factory=threading.Event)

    output_started: asyncio.Event = field(default_factory=asyncio.Event)

    release_output: asyncio.Event = field(default_factory=asyncio.Event)

    asr_count: int = 0

    answer_count: int = 0

    def transcribe(
        self, endpoint: EndpointedUtterance, cancellation: CancellationToken
    ) -> ASRAudienceEvent:
        """函数契约说明.

        功能: 执行 transcribe 的同步逻辑,并协调 set,
        ASRAudienceEvent。
        参数: self 表示当前实例。 endpoint:
        EndpointedUtterance。 必填。
        cancellation: CancellationToken。
        必填。
        契约: 同步调用。 返回 `ASRAudienceEvent`。
        """

        _ = (endpoint, cancellation)

        self.asr_count += 1

        _ = self.asr_started.set()

        if self.asr_count == 4:
            _ = self.all_asr_finished.set()

        return ASRAudienceEvent("question", 0, "segment", self.asr_count)

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult:
        """函数契约说明.

        功能: 执行 answer 的同步逻辑,并协调 set,
        wait, TurnResult,
        PipelineTurnId。
        参数: self 表示当前实例。 event:
        ASRAudienceEvent。 必填。
        cancellation: CancellationToken。
        必填。
        契约: 同步调用。 返回 `TurnResult`。
        """

        _ = (event, cancellation)

        _ = self.answer_started.set()

        _ = self.release_answer.wait()

        self.answer_count += 1

        if self.answer_count == 3:
            _ = self.all_answers_finished.set()

        return TurnResult(
            PipelineTurnId(f"turn-{self.answer_count}"),
            PipelineSegmentId(f"segment-{self.answer_count}"),
            "answer",
            used_fallback=False,
        )

    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...]:
        """函数契约说明.

        功能: 执行 synthesize 的同步逻辑,并协调
        tuple, Pcm16leChunk, range。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。 cancellation:
        CancellationToken。 必填。
        契约: 同步调用。 返回
        `tuple[Pcm16leChunk, ...]`。
        """

        _ = (turn, cancellation)

        return tuple(Pcm16leChunk(b"\x10\x20") for _ in range(17))

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:
        """函数契约说明.

        功能: 执行 complete 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。 chunks:
        tuple[Pcm16leChunk, ...]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = (turn, chunks)

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        """函数契约说明.

        功能: 执行 output 的异步逻辑,并协调 set,
        wait。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 epoch:
        CancellationEpoch。 必填。 packet:
        bytes。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """

        _ = (stream, epoch, packet)

        _ = self.output_started.set()

        _ = await self.release_output.wait()


def test_actor_counts_newest_wins_drops_for_all_stage_mailboxes() -> None:
    """函数契约说明.

    功能: 验证 actor counts newest wins
    drops for all stage mailboxes
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_stage_mailbox_drop_count_proof())


async def _stage_mailbox_drop_count_proof() -> None:
    """函数契约说明.

    功能: 执行
    _stage_mailbox_drop_count_proof
    的异步逻辑,并协调 _BackpressureStages,
    StreamKey, OnsiteStreamActor,
    submit。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    stages = _BackpressureStages()

    stream = StreamKey("session", "stream")

    actor = OnsiteStreamActor(stream, CancellationEpoch(0), stages)

    actor.submit(_endpoint(stream), CancellationEpoch(0))

    _ = await asyncio.to_thread(stages.answer_started.wait)

    for _ in range(3):
        actor.submit(_endpoint(stream), CancellationEpoch(0))

    _ = await asyncio.to_thread(stages.all_asr_finished.wait)

    assert actor.drop_counts.answer_turns == 1

    _ = stages.release_answer.set()

    _ = await asyncio.to_thread(stages.all_answers_finished.wait)

    _ = await stages.output_started.wait()

    assert actor.drop_counts.tts_chunks > 0

    _ = stages.release_output.set()

    await actor.wait_quiescent()


def test_actor_invalidation_cancels_provider_and_waits() -> None:
    """函数契约说明.

    功能: 验证 actor invalidation cancels
    provider and waits 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_blocking_provider_invalidation_proof())


async def _blocking_provider_invalidation_proof() -> None:
    # Given: an ASR provider that can settle only when its cancellation resource closes.

    """函数契约说明.

    功能: 执行 _blocking_provider_invalidati
    on_proof 的异步逻辑,并协调 _BlockingStages,
    StreamKey, OnsiteStreamActor,
    submit。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    stages = _BlockingStages()

    stream = StreamKey("session", "stream")

    actor = OnsiteStreamActor(stream, CancellationEpoch(0), stages)

    actor.submit(_endpoint(stream), CancellationEpoch(0))

    _ = await asyncio.to_thread(stages.asr_started.wait)

    for _ in range(5):
        actor.submit(_endpoint(stream), CancellationEpoch(0))

    try:
        # When: route invalidation retires the admitted epoch while ASR is blocked.

        actor.invalidate(CancellationEpoch(1))

        await actor.wait_quiescent()

        # Then: the owned provider resource is cancelled, settles, and cannot emit

        # stale work.

        assert stages.provider_cancelled.is_set()

        assert stages.provider_completed.is_set()

        assert stages.outputs == []

        assert actor.drop_counts.endpointed_utterances == 1

    finally:
        _ = stages.release.set()

        await actor.wait_quiescent()


def _endpoint(stream: StreamKey) -> EndpointedUtterance:
    """函数契约说明.

    功能: 执行 _endpoint 的同步逻辑,并协调
    EndpointedUtterance, TurnId,
    SegmentId, CancellationEpoch。
    参数: stream: StreamKey。 必填。
    契约: 同步调用。 返回 `EndpointedUtterance`。
    """

    return EndpointedUtterance(
        stream=stream,
        payload=b"\x01\x02" * 320,
        reason=EndpointReason.FORCED,
        turn_id=TurnId("turn"),
        segment_id=SegmentId("segment"),
        cancellation_epoch=CancellationEpoch(0),
    )
