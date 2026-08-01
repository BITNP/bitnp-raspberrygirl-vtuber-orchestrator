
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from orchestrator.asr_semantic_gate import AsrGateDecision
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
    from collections.abc import Iterator

    from orchestrator.llm import CancellationToken


@dataclass(slots=True)
class _Stages:

    asr_started: asyncio.Event = field(default_factory=asyncio.Event)

    release_asr: asyncio.Event = field(default_factory=asyncio.Event)

    outputs: list[tuple[StreamKey, CancellationEpoch, bytes]] = field(
        default_factory=list
    )

    def transcribe(
        self, endpoint: EndpointedUtterance, cancellation: CancellationToken
    ) -> ASRAudienceEvent:

        _ = (endpoint, cancellation)

        _ = self.asr_started.set()

        return ASRAudienceEvent("question", 0, "segment", 1)

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult:

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

        _ = (turn, cancellation)

        return (Pcm16leChunk(b"\x10\x20" * 320),)

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:

        _ = (turn, chunks)

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:

        self.outputs.append((stream, epoch, packet))


@dataclass(slots=True)
class _DiscardingGateStages(_Stages):

    gate_calls: list[tuple[str, str, bool]] = field(default_factory=list)

    answer_calls: int = 0

    def gate(
        self,
        event: ASRAudienceEvent,
        *,
        active_answer_excerpt: str,
        is_playing: bool,
    ) -> AsrGateDecision:
        self.gate_calls.append((event.text, active_answer_excerpt, is_playing))
        return AsrGateDecision.DISCARD

    @override
    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult:
        _ = (event, cancellation)
        self.answer_calls += 1
        message = "discarded ASR must not reach the LLM lane"
        raise AssertionError(message)


@dataclass(slots=True)
class _StreamingStages(_Stages):

    streamed: list[int] = field(default_factory=list)

    completed_pcm_bytes: int | None = None

    first_packet: asyncio.Event = field(default_factory=asyncio.Event)

    release_second_chunk: threading.Event = field(default_factory=threading.Event)

    def stream_synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> Iterator[Pcm16leChunk]:
        _ = (turn, cancellation)

        def chunks() -> Iterator[Pcm16leChunk]:
            self.streamed.append(1)
            yield Pcm16leChunk(b"\x10\x20" * 320)
            _ = self.release_second_chunk.wait()
            self.streamed.append(2)
            yield Pcm16leChunk(b"\x30\x40" * 320)

        return chunks()

    @override
    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...]:
        _ = (turn, cancellation)
        message = "streaming TTS must not call final synthesis"
        raise AssertionError(message)

    def complete_stream(self, turn: TurnResult, pcm_bytes: int) -> None:
        _ = turn
        self.completed_pcm_bytes = pcm_bytes

    @override
    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        self.outputs.append((stream, epoch, packet))
        self.first_packet.set()


def test_actor_tags_generated_rtp_with_admission_epoch() -> None:

    asyncio.run(_epoch_proof())


def test_actor_drops_asr_final_when_semantic_gate_rejects_it() -> None:
    asyncio.run(_semantic_gate_proof())


def test_actor_emits_streaming_tts_chunks_without_materializing_clip() -> None:
    asyncio.run(_streaming_tts_proof())


async def _streaming_tts_proof() -> None:
    stages = _StreamingStages()
    stream = StreamKey("session", "stream")
    actor = OnsiteStreamActor(stream, CancellationEpoch(0), stages)

    actor.submit(_endpoint(stream), CancellationEpoch(0))

    _ = await stages.first_packet.wait()
    assert stages.streamed == [1]
    _ = stages.release_second_chunk.set()
    await actor.wait_quiescent()

    assert stages.streamed == [1, 2]
    assert stages.completed_pcm_bytes == 1_280
    assert [epoch for _, epoch, _ in stages.outputs] == [
        CancellationEpoch(0),
        CancellationEpoch(0),
    ]


async def _semantic_gate_proof() -> None:
    stages = _DiscardingGateStages()
    stream = StreamKey("session", "stream")
    actor = OnsiteStreamActor(stream, CancellationEpoch(0), stages)

    actor.submit(_endpoint(stream), CancellationEpoch(0))

    await actor.wait_quiescent()

    assert stages.gate_calls == [("question", "", False)]
    assert stages.answer_calls == 0
    assert stages.outputs == []


async def _epoch_proof() -> None:
    # Given: one actor with pure deterministic provider stages.


    stages = _Stages()

    stream = StreamKey("session", "stream")

    actor = OnsiteStreamActor(stream, CancellationEpoch(7), stages)

    # When: an endpointed utterance enters the staged pipeline.

    actor.submit(_endpoint(stream), CancellationEpoch(7))

    await actor.wait_quiescent()

    # Then: every emitted packet carries the admission epoch.

    assert [epoch for _, epoch, _ in stages.outputs] == [CancellationEpoch(7)]


def test_actor_invalidation_clears_queued_work_and_suppresses_output() -> None:

    asyncio.run(_invalidation_proof())


async def _invalidation_proof() -> None:
    # Given: an actor whose first provider stage has accepted work.


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

        _ = (turn, cancellation)

        return (Pcm16leChunk(b"\x10\x20" * 320),)

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:

        _ = (turn, chunks)

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:

        self.outputs.append((stream, epoch, packet))

    def _cancel_provider(self) -> None:

        _ = self.provider_cancelled.set()

        _ = self.release.set()


@dataclass(slots=True)
class _BackpressureStages:

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

        _ = (endpoint, cancellation)

        self.asr_count += 1

        _ = self.asr_started.set()

        if self.asr_count == 4:
            _ = self.all_asr_finished.set()

        return ASRAudienceEvent("question", 0, "segment", self.asr_count)

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult:

        _ = (event, cancellation)

        _ = self.answer_started.set()

        _ = self.release_answer.wait()

        self.answer_count += 1

        # Newer recognized ASR finals cancel the active answer and retain only
        # the latest queued answer, so the stale first answer never emits.
        if self.answer_count == 2:
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

        _ = (turn, cancellation)

        return tuple(Pcm16leChunk(b"\x10\x20") for _ in range(17))

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:

        _ = (turn, chunks)

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:

        _ = (stream, epoch, packet)

        _ = self.output_started.set()

        _ = await self.release_output.wait()


def test_actor_counts_newest_wins_drops_for_all_stage_mailboxes() -> None:

    asyncio.run(_stage_mailbox_drop_count_proof())


async def _stage_mailbox_drop_count_proof() -> None:

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

    asyncio.run(_blocking_provider_invalidation_proof())


async def _blocking_provider_invalidation_proof() -> None:
    # Given: an ASR provider that can settle only when its cancellation resource closes.


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

    return EndpointedUtterance(
        stream=stream,
        payload=b"\x01\x02" * 320,
        reason=EndpointReason.FORCED,
        turn_id=TurnId("turn"),
        segment_id=SegmentId("segment"),
        cancellation_epoch=CancellationEpoch(0),
    )
