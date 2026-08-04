import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from typing import cast

from orchestrator.agent_pipeline import (
    AgentPipeline,
    AsyncAgentPipeline,
    BrainStateSnapshot,
    FrontendOperation,
    GateDecision,
    MediaOperation,
)
from orchestrator.caption_timeline import CaptionTimelineCommand
from orchestrator.ids import SegmentId, SessionId, TraceId, TurnId
from orchestrator.intent_router import IntentRouter, IntentSpec
from orchestrator.interactions import CommentProposal
from orchestrator.response_contracts import ResponseProposal
from orchestrator.response_coordinator import AsyncResponseCoordinator
from orchestrator.response_execution_mode import ResponseExecutionMode
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence, StateRevision
from orchestrator.state_snapshots import MemoryRevision
from orchestrator.task_reducer import TaskEffect, TaskResult
from orchestrator.task_registry import (
    IdempotencyKey,
    SchedulerTaskConfig,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRequest,
    TaskState,
)
from orchestrator.transient_context import (
    ContextComposition,
    ContextProvenance,
    ContextSequence,
    ContextSourceId,
    FinalizedInput,
)


def test_runtime_rejects_stale_snapshot_before_registry_or_lane_mutation() -> None:
    # Given: a live runtime and a request captured before its active revision.
    runtime = _runtime()
    request = replace(_request(runtime), snapshot_revision=StateRevision(0))

    # When: stale work reaches the only runtime admission surface.
    outcome = runtime.schedule_task(request, _correlation("task", 2))

    # Then: it is neither retained nor available to a worker.
    assert outcome.accepted is False
    assert runtime.task_registry.records == ()
    assert runtime.next_task(now_ms=0) is None


def test_runtime_rejects_inactive_turn_before_registry_or_lane_mutation() -> None:
    # Given: a live runtime and a request for another turn.
    runtime = _runtime()
    request = replace(_request(runtime), turn_id=TurnId("inactive-turn"))

    # When: inactive work reaches runtime admission.
    outcome = runtime.schedule_task(request, _correlation("task", 2))

    # Then: it is neither retained nor available to a worker.
    assert outcome.accepted is False
    assert runtime.task_registry.records == ()
    assert runtime.next_task(now_ms=0) is None


def test_runtime_rejects_explicitly_stale_data_snapshot_before_enqueue() -> None:
    # Given: a live runtime and a caller-supplied stale data revision.
    runtime = _runtime()
    stale_snapshot = replace(
        runtime.interaction_ingress.data.task_snapshot,
        memory_revision=MemoryRevision(1),
    )
    request = replace(_request(runtime), data_snapshot=stale_snapshot)

    # When: the stale data request reaches runtime admission.
    outcome = runtime.schedule_task(request, _correlation("task", 2))

    # Then: it cannot create registry or lane state.
    assert outcome.accepted is False
    assert runtime.task_registry.records == ()
    assert runtime.next_task(now_ms=0) is None


def test_runtime_rejects_stale_result_after_newer_turn_without_commit() -> None:
    # Given: admitted work for a current runtime turn.
    runtime = _runtime()
    request = _request(runtime)
    correlation = _correlation("task", 2)
    assert runtime.schedule_task(request, correlation).accepted

    # When: a newer turn invalidates the original result.
    _ = runtime.receive_comment(CommentProposal("newer", _correlation("newer", 3)))
    outcome = runtime.reduce_task(_result(request), correlation)

    # Then: no effect is committed.
    assert outcome.accepted is False
    assert runtime.observables.task_commits == ()


def test_runtime_rejects_result_after_task_capability_is_revoked() -> None:
    runtime = _runtime()
    request = _request(runtime)
    correlation = _correlation("task", 2)
    assert runtime.schedule_task(request, correlation).accepted
    record = runtime.task_registry.task(request.task_id)
    assert record is not None
    runtime.agent_capabilities = frozenset()

    outcome = runtime.reduce_task(
        TaskResult(
            task_id=request.task_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            snapshot_revision=request.snapshot_revision,
            effect=TaskEffect("answer", "accepted"),
            cancellation_epoch=record.request.cancellation_epoch,
        ),
        correlation,
    )

    assert outcome.accepted is False
    cancelled = runtime.task_registry.task(request.task_id)
    assert cancelled is not None
    assert cancelled.state is TaskState.CANCELLED


def test_started_timeline_delivery_is_a_claimed_and_reducer_fenced_task() -> None:
    runtime = _runtime()
    turn_id = runtime.scheduler.snapshot.active_turn_id
    assert turn_id is not None
    correlation = _correlation("timeline", 2)

    scheduled = runtime.schedule_caption_timeline_delivery(
        turn_id,
        CaptionTimelineCommand(
            "timeline-1", "带标记的字幕", f"agent-{turn_id}", 1, 96_000
        ),
        correlation=correlation,
        segment_id=SegmentId("segment-1"),
    )

    assert scheduled is not None
    task_id, timeline = scheduled
    assert timeline.marked_text == "带标记的字幕"
    record = runtime.task_registry.task(task_id)
    assert record is not None
    assert record.state is TaskState.RUNNING
    assert record.request.segment_id == SegmentId("segment-1")
    assert runtime.caption_timeline_delivery_is_current(task_id)
    assert runtime.complete_caption_timeline_delivery(task_id, correlation)
    completed = runtime.task_registry.task(task_id)
    assert completed is not None
    assert completed.state is TaskState.COMPLETED


def test_sound_flush_is_a_claimed_and_reducer_fenced_task() -> None:
    runtime = _runtime()
    turn_id = runtime.scheduler.snapshot.active_turn_id
    assert turn_id is not None
    correlation = _correlation("flush", 2)

    task_id = runtime.schedule_sound_flush(
        turn_id,
        SegmentId("segment-replacement"),
        request_id="flush-1",
        correlation=correlation,
    )

    assert task_id is not None
    record = runtime.task_registry.task(task_id)
    assert record is not None
    assert record.state is TaskState.RUNNING
    assert runtime.sound_flush_is_current(task_id)
    assert runtime.complete_sound_flush(task_id, correlation)
    completed = runtime.task_registry.task(task_id)
    assert completed is not None
    assert completed.state is TaskState.COMPLETED


def test_new_turn_closes_pending_timeline_delivery_before_frontend_io() -> None:
    runtime = _runtime()
    turn_id = runtime.scheduler.snapshot.active_turn_id
    assert turn_id is not None
    scheduled = runtime.schedule_caption_timeline_delivery(
        turn_id,
        CaptionTimelineCommand(
            "timeline-1", "旧字幕", f"agent-{turn_id}", 1, 96_000
        ),
        correlation=_correlation("timeline", 2),
    )
    assert scheduled is not None
    task_id, _ = scheduled

    newer = runtime.receive_comment(CommentProposal("newer", _correlation("new", 3)))

    assert newer.accepted
    assert not runtime.caption_timeline_delivery_is_current(task_id)
    cancelled = runtime.task_registry.task(task_id)
    assert cancelled is not None
    assert cancelled.state is TaskState.CANCELLED


def test_runtime_routes_comment_through_agent_gate_and_brain_snapshot() -> None:
    pipeline = AgentPipeline(_AcceptGate(), _PlanBrain(), _NoTools())
    effects = _Effects()
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        clock=lambda: 10,
        agent_pipeline=pipeline,
        agent_capabilities=frozenset({"task:tts"}),
        agent_effect_dispatcher=effects,
    )

    outcome = runtime.receive_comment(CommentProposal("question", _correlation("q", 1)))

    assert outcome.accepted
    assert len(runtime.agent_results) == 1
    assert tuple(
        entry.text
        for entry in runtime.interaction_ingress.data.context.snapshot.entries
    ) == (
        "question",
        "answer",
    )
    assert runtime.task_registry.records[0].request.kind is TaskKind.INTERACTIVE
    assert runtime.interaction_ingress.data.memory.snapshot.entries[0].value == "小莓"
    assert effects.media == [MediaOperation("synthesize", text="answer")]
    assert effects.frontend == [FrontendOperation("caption", "answer")]


def test_runtime_emits_reducer_approved_tts_task_once() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-tts"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        clock=lambda: 10,
        agent_pipeline=AgentPipeline(_AcceptGate(), _PlanBrain(), _NoTools()),
        agent_capabilities=frozenset({"task:tts"}),
    )
    correlation = EventCorrelation(
        TraceId("tts"), SessionId("session-tts"), EventSequence(1)
    )
    outcome = runtime.receive_comment(CommentProposal("question", correlation))
    assert outcome.accepted
    assert outcome.turn_id is not None
    emitted: list[str] = []

    async def synthesize(text: str, output_started: Callable[[], bool]) -> bool:
        emitted.append(text)
        assert output_started()
        return True

    assert asyncio.run(
        runtime.run_agent_tts_for_turn(outcome.turn_id, synthesize, correlation)
    )
    assert emitted == ["answer"]
    assert runtime.observables.task_commits[-1].effect.effect_type == "tts.emitted"


def test_runtime_does_not_fill_interactive_queue_with_completed_agent_tts() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-tts-queue"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        clock=lambda: 10,
        agent_pipeline=AgentPipeline(_AcceptGate(), _PlanBrain(), _NoTools()),
        agent_capabilities=frozenset({"task:tts"}),
    )
    emitted: list[str] = []

    async def synthesize(text: str, output_started: Callable[[], bool]) -> bool:
        emitted.append(text)
        return output_started()

    for sequence in range(1, 6):
        correlation = EventCorrelation(
            TraceId(f"tts-{sequence}"),
            SessionId("session-tts-queue"),
            EventSequence(sequence),
        )
        outcome = runtime.receive_comment(CommentProposal("question", correlation))
        assert outcome.accepted
        assert outcome.turn_id is not None
        assert asyncio.run(
            runtime.run_agent_tts_for_turn(outcome.turn_id, synthesize, correlation)
        )

    assert emitted == ["answer"] * 5
    assert all(
        record.state is TaskState.COMPLETED
        for record in runtime.task_registry.records
    )


def test_runtime_commits_tts_before_paced_playback_can_be_superseded() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-tts-lifecycle"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        clock=lambda: 10,
        agent_pipeline=AgentPipeline(_AcceptGate(), _PlanBrain(), _NoTools()),
        agent_capabilities=frozenset({"task:tts"}),
    )
    first_correlation = EventCorrelation(
        TraceId("first"), SessionId("session-tts-lifecycle"), EventSequence(1)
    )
    first = runtime.receive_comment(CommentProposal("first", first_correlation))
    assert first.accepted
    assert first.turn_id is not None

    async def synthesize(_text: str, output_started: Callable[[], bool]) -> bool:
        assert output_started()
        # This models a user speaking while the accepted reply is still being
        # paced to Sound.  It must not invalidate the already admitted task.
        second = runtime.receive_comment(
            CommentProposal(
                "second",
                EventCorrelation(
                    TraceId("second"),
                    SessionId("session-tts-lifecycle"),
                    EventSequence(2),
                ),
            )
        )
        assert second.accepted
        return True

    assert asyncio.run(
        runtime.run_agent_tts_for_turn(first.turn_id, synthesize, first_correlation)
    )
    first_task = runtime.task_registry.records[0]
    assert first_task.state is TaskState.COMPLETED


def test_async_response_is_registry_owned_before_it_can_admit_tts() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-async-response"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _AsyncResponseBrain(), IntentRouter(()), _AsyncNoTools()
        ),
    )
    correlation = EventCorrelation(
        TraceId("async-response"), SessionId("session-async-response"), EventSequence(1)
    )

    outcome = asyncio.run(
        runtime.receive_comment_async(CommentProposal("问题", correlation))
    )

    assert outcome.accepted
    records = runtime.task_registry.records
    assert records[0].request.task_id == TaskId("response-llm-initial-turn-0001")
    assert records[0].request.cancellation_epoch == 1
    assert records[0].state is TaskState.COMPLETED
    assert records[1].request.parent_task_id == records[0].request.task_id
    assert records[1].state is TaskState.RUNNING


def test_async_tool_turn_records_initial_tool_and_final_provider_tasks() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-async-tool"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _ToolResponseBrain(),
            IntentRouter(
                (
                    IntentSpec(
                        "knowledge",
                        "knowledge",
                        "local",
                        "knowledge.lookup",
                        lambda snapshot: {"query": snapshot.input.text},
                    ),
                )
            ),
            _AsyncTools(),
        ),
        agent_capabilities=frozenset({"knowledge.lookup"}),
    )
    correlation = EventCorrelation(
        TraceId("async-tool"), SessionId("session-async-tool"), EventSequence(1)
    )

    outcome = asyncio.run(
        runtime.receive_comment_async(CommentProposal("查询", correlation))
    )

    assert outcome.accepted
    records = runtime.task_registry.records
    assert [record.request.task_id for record in records] == [
        TaskId("response-llm-initial-turn-0001"),
        TaskId("response-tool-turn-0001"),
        TaskId("response-llm-final-turn-0001"),
        TaskId("response-tts-turn-0001"),
    ]
    assert [record.state for record in records] == [
        TaskState.COMPLETED,
        TaskState.COMPLETED,
        TaskState.COMPLETED,
        TaskState.RUNNING,
    ]
    assert records[1].request.parent_task_id == records[0].request.task_id
    assert records[2].request.parent_task_id == records[1].request.task_id


def test_async_tool_task_uses_its_trusted_intent_deadline() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        clock=lambda: 100,
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _ToolResponseBrain(),
            IntentRouter(
                (
                    IntentSpec(
                        "knowledge",
                        "knowledge",
                        "local",
                        "knowledge.lookup",
                        lambda snapshot: {"query": snapshot.input.text},
                        timeout_ms=7,
                    ),
                )
            ),
            _AsyncTools(),
        ),
        agent_capabilities=frozenset({"knowledge.lookup"}),
    )

    outcome = asyncio.run(
        runtime.receive_comment_async(
            CommentProposal("查询", _correlation("async-tool-deadline", 1))
        )
    )

    assert outcome.accepted
    tool_task = runtime.task_registry.task(TaskId("response-tool-turn-0001"))
    assert tool_task is not None
    assert tool_task.request.deadline_ms == TaskDeadlineMs(107)


def test_response_provider_timeout_closes_its_registry_task() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _HangingResponseBrain(), IntentRouter(()), _AsyncNoTools()
        ),
        response_task_timeout_ms=10,
    )

    outcome = asyncio.run(
        runtime.receive_comment_async(
            CommentProposal("问题", _correlation("async-timeout", 1))
        )
    )

    assert outcome.accepted
    task = runtime.task_registry.task(TaskId("response-llm-initial-turn-0001"))
    assert task is not None
    assert task.state is TaskState.TIMED_OUT
    assert runtime.active_response_provider_task_ids == frozenset()


def test_new_turn_cancels_the_active_response_provider_after_fencing() -> None:
    brain = _FirstResponseBlocks()
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            brain, IntentRouter(()), _AsyncNoTools()
        ),
    )

    async def exercise() -> None:
        first = asyncio.create_task(
            runtime.receive_comment_async(
                CommentProposal("第一句", _correlation("async-cancel-1", 1))
            )
        )
        _ = await brain.started.wait()
        second = await runtime.receive_comment_async(
            CommentProposal("第二句", _correlation("async-cancel-2", 2))
        )
        first_outcome = await first
        assert first_outcome.accepted
        assert second.accepted

    asyncio.run(exercise())

    first_task = runtime.task_registry.task(TaskId("response-llm-initial-turn-0001"))
    assert first_task is not None
    assert first_task.state is TaskState.CANCELLED
    assert brain.cancelled
    assert runtime.active_response_provider_task_ids == frozenset()


def test_preoutput_tts_timeout_cancels_provider_without_context_commit() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _AsyncResponseBrain(), IntentRouter(()), _AsyncNoTools()
        ),
        response_task_timeout_ms=30,
    )
    provider_cancelled = False

    async def synthesize(_text: str, _output_started: Callable[[], bool]) -> bool:
        nonlocal provider_cancelled
        try:
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled = True
            raise
        return False

    async def exercise() -> bool:
        outcome = await runtime.receive_comment_async(
            CommentProposal("问题", _correlation("tts-timeout", 1))
        )
        assert outcome.turn_id is not None
        return await runtime.run_agent_tts_for_turn(
            outcome.turn_id, synthesize, _correlation("tts-timeout", 1)
        )

    assert asyncio.run(exercise()) is False
    tts_task = runtime.task_registry.task(TaskId("response-tts-turn-0001"))
    assert tts_task is not None
    assert tts_task.state is TaskState.TIMED_OUT
    assert provider_cancelled
    assert runtime.active_preoutput_tts_provider_task_ids == frozenset()
    assert runtime.interaction_ingress.data.context.snapshot.entries == ()


def test_session_end_cancels_tts_provider_after_audio_has_started() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _AsyncResponseBrain(), IntentRouter(()), _AsyncNoTools()
        ),
    )
    audio_started = asyncio.Event()
    provider_cancelled = False

    async def synthesize(_text: str, output_started: Callable[[], bool]) -> bool:
        nonlocal provider_cancelled
        assert output_started()
        _ = audio_started.set()
        try:
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled = True
            raise
        return False

    async def exercise() -> bool:
        outcome = await runtime.receive_comment_async(
            CommentProposal("问题", _correlation("tts-session-end", 1))
        )
        assert outcome.turn_id is not None
        running = asyncio.create_task(
            runtime.run_agent_tts_for_turn(
                outcome.turn_id, synthesize, _correlation("tts-session-end", 1)
            )
        )
        _ = await audio_started.wait()
        assert runtime.end_session(_correlation("tts-session-end", 2)).accepted
        return await running

    assert asyncio.run(exercise()) is False
    assert provider_cancelled
    assert runtime.active_preoutput_tts_provider_task_ids == frozenset()


def test_new_shadow_never_executes_selected_tool_or_creates_effect_tasks() -> None:
    tools = _CountingAsyncTools()
    runtime = SessionRuntime.create(
        session_id=SessionId("session-shadow"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _ToolResponseBrain(),
            IntentRouter(
                (
                    IntentSpec(
                        "knowledge",
                        "knowledge",
                        "local",
                        "knowledge.lookup",
                        lambda snapshot: {"query": snapshot.input.text},
                    ),
                )
            ),
            tools,
        ),
        response_execution_mode=ResponseExecutionMode.NEW_SHADOW,
        agent_capabilities=frozenset({"knowledge.lookup"}),
    )
    correlation = EventCorrelation(
        TraceId("shadow"), SessionId("session-shadow"), EventSequence(1)
    )

    outcome = asyncio.run(
        runtime.receive_comment_async(CommentProposal("查询", correlation))
    )

    assert outcome.accepted
    assert tools.calls == 0
    assert [record.request.task_id for record in runtime.task_registry.records] == [
        TaskId("response-llm-initial-turn-0001"),
    ]
    assert runtime.interaction_ingress.data.context.snapshot.entries == ()
    assert any(
        record.stage == "response_shadow"
        for record in runtime.operational_journal.records
    )


def test_memory_extraction_runs_only_after_tts_output_is_accepted() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-async-memory"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _AsyncResponseBrain(), IntentRouter(()), _AsyncNoTools()
        ),
        memory_candidate_extractor=_MemoryExtractor(),
    )
    correlation = EventCorrelation(
        TraceId("async-memory"), SessionId("session-async-memory"), EventSequence(1)
    )

    async def exercise() -> None:
        outcome = await runtime.receive_comment_async(
            CommentProposal("我喜欢绿茶", correlation)
        )
        assert outcome.turn_id is not None

        async def synthesize(_text: str, output_started: Callable[[], bool]) -> bool:
            assert output_started()
            await asyncio.sleep(0)
            return True

        assert await runtime.run_agent_tts_for_turn(
            outcome.turn_id, synthesize, correlation
        )
        await asyncio.sleep(0)

    asyncio.run(exercise())

    entries = runtime.interaction_ingress.data.memory.snapshot.entries
    assert [(entry.key, entry.value) for entry in entries] == [
        ("drink_preference", "喜欢绿茶")
    ]
    memory_task = runtime.task_registry.task(TaskId("memory-extract-turn-0001"))
    assert memory_task is not None
    assert memory_task.state is TaskState.COMPLETED


def test_context_compaction_is_a_maintenance_task_after_audio_start() -> None:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-async-compact"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_agent_pipeline=cast(
            "AsyncAgentPipeline", cast("object", _AsyncAcceptPipeline())
        ),
        async_response_coordinator=AsyncResponseCoordinator(
            _AsyncResponseBrain(), IntentRouter(()), _AsyncNoTools()
        ),
        context_compactor=_ContextCompactor(),
    )
    data = runtime.interaction_ingress.data
    data.consider_context(
        FinalizedInput(
            ContextProvenance(
                SessionId("session-async-compact"),
                TurnId("old-turn"),
                SegmentId("old-segment"),
                ContextSequence(0),
                ContextSourceId("old-trace"),
            ),
            "old " * 600,
        )
    )
    correlation = EventCorrelation(
        TraceId("async-compact"), SessionId("session-async-compact"), EventSequence(1)
    )

    async def exercise() -> None:
        outcome = await runtime.receive_comment_async(
            CommentProposal("压缩一下", correlation)
        )
        assert outcome.turn_id is not None

        async def synthesize(_text: str, output_started: Callable[[], bool]) -> bool:
            assert output_started()
            await asyncio.sleep(0)
            return True

        assert await runtime.run_agent_tts_for_turn(
            outcome.turn_id, synthesize, correlation
        )
        await asyncio.sleep(0)

    asyncio.run(exercise())

    assert data.context.snapshot.summary == "稳定摘要"
    task = runtime.task_registry.task(TaskId("context-compact-turn-0001"))
    assert task is not None
    assert task.state is TaskState.COMPLETED


def test_runtime_commits_brain_compaction_against_its_snapshot() -> None:
    pipeline = AgentPipeline(_AcceptGate(), _CompactingBrain(), _NoTools())
    runtime = SessionRuntime.create(
        session_id=SessionId("session-compaction"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        agent_pipeline=pipeline,
    )
    data = runtime.interaction_ingress.data
    for sequence in (1, 2):
        data.consider_context(
            FinalizedInput(
                ContextProvenance(
                    SessionId("session-compaction"),
                    TurnId(f"old-{sequence}"),
                    SegmentId(f"old-{sequence}"),
                    ContextSequence(sequence),
                    ContextSourceId(f"old-{sequence}"),
                ),
                "old " * 300,
            )
        )

    outcome = runtime.receive_comment(
        CommentProposal(
            "question",
            EventCorrelation(
                TraceId("compact"),
                SessionId("session-compaction"),
                EventSequence(3),
            ),
        )
    )

    assert outcome.accepted
    assert data.context.snapshot.summary == "稳定事实: 已压缩"


def test_mcp_observation_is_available_to_final_brain_but_not_persisted() -> None:
    pipeline = AgentPipeline(_AcceptGate(), _McpBrain(), _McpTools())
    runtime = SessionRuntime.create(
        session_id=SessionId("session-mcp"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        agent_pipeline=pipeline,
        agent_capabilities=frozenset({"mcp:web/search"}),
        agent_mcp_allowlist=frozenset({"web/search"}),
    )

    outcome = runtime.receive_comment(
        CommentProposal(
            "查询",
            EventCorrelation(
                TraceId("mcp"), SessionId("session-mcp"), EventSequence(1)
            ),
        )
    )

    assert outcome.accepted
    entries = runtime.interaction_ingress.data.context.snapshot.entries
    assert tuple(entry.text for entry in entries) == ("查询", "answer")


def test_next_brain_snapshot_includes_materialized_frontend_state() -> None:
    brain = _FrontendStateBrain()
    runtime = SessionRuntime.create(
        session_id=SessionId("session-frontend-state"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        agent_pipeline=AgentPipeline(_AcceptGate(), brain, _NoTools()),
    )

    first = runtime.receive_comment(
        CommentProposal(
            "展示字幕",
            EventCorrelation(
                TraceId("frontend-1"),
                SessionId("session-frontend-state"),
                EventSequence(1),
            ),
        )
    )
    second = runtime.receive_comment(
        CommentProposal(
            "继续",
            EventCorrelation(
                TraceId("frontend-2"),
                SessionId("session-frontend-state"),
                EventSequence(2),
            ),
        )
    )

    assert first.accepted
    assert second.accepted
    assert brain.snapshots[1].frontend_caption == "已显示"
    assert brain.snapshots[1].frontend_animation == "talk"
    assert brain.snapshots[1].ppt_deck_id == "deck-demo"
    assert brain.snapshots[1].ppt_page == 1


class _AcceptGate:
    def evaluate(self, *args: object, **kwargs: object) -> GateDecision:
        _ = args, kwargs
        return GateDecision.ACCEPT


class _AsyncAcceptPipeline:
    async def submit(self, *args: object, **kwargs: object) -> GateDecision:
        _ = args, kwargs
        return GateDecision.ACCEPT


class _AsyncResponseBrain:
    async def respond(self, *args: object, **kwargs: object) -> ResponseProposal:
        _ = args, kwargs
        return ResponseProposal("answer", "answer")


class _HangingResponseBrain:
    async def respond(self, *args: object, **kwargs: object) -> ResponseProposal:
        _ = args, kwargs
        _ = await asyncio.Event().wait()
        return ResponseProposal("不应返回", "answer")


class _FirstResponseBlocks:
    def __init__(self) -> None:
        self.started: asyncio.Event = asyncio.Event()
        self.cancelled: bool = False
        self.calls: int = 0

    async def respond(self, *args: object, **kwargs: object) -> ResponseProposal:
        _ = args, kwargs
        self.calls += 1
        if self.calls == 1:
            _ = self.started.set()
            try:
                _ = await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return ResponseProposal("answer", "answer")


class _AsyncNoTools:
    async def execute(self, *args: object, **kwargs: object) -> str | None:
        _ = args, kwargs
        return None


class _ToolResponseBrain:
    def __init__(self) -> None:
        self._responses: list[ResponseProposal] = [
            ResponseProposal("", "knowledge"),
            ResponseProposal("答案", "answer"),
        ]

    async def respond(self, *args: object, **kwargs: object) -> ResponseProposal:
        _ = args, kwargs
        return self._responses.pop(0)


class _AsyncTools:
    async def execute(self, *args: object, **kwargs: object) -> str:
        _ = args, kwargs
        return "受控检索结果"


class _CountingAsyncTools:
    def __init__(self) -> None:
        self.calls: int = 0

    async def execute(self, *args: object, **kwargs: object) -> str:
        _ = args, kwargs
        self.calls += 1
        return "不应执行"


class _MemoryExtractor:
    async def extract(self, *, user_text: str, reply_text: str) -> str:
        assert user_text == "我喜欢绿茶"
        assert reply_text == "answer"
        return '{"key":"drink_preference","value":"喜欢绿茶","confidence":95}'


class _ContextCompactor:
    async def compact(self, composition: ContextComposition) -> str:
        assert composition.digests
        return "稳定摘要"


class _PlanBrain:
    def plan(self, snapshot: BrainStateSnapshot, **kwargs: object) -> str:
        _ = kwargs
        assert snapshot.knowledge_references == (
            "本地知识库: corpus=fixture-corpus@1, index=fixture-index@1",
        )
        return json.dumps(
            {
                "response_text": "answer",
                "expected_revision": snapshot.revision,
                "state_operations": [
                    {"kind": "create_task", "payload": {"task_kind": "tts"}}
                ],
                "memory_patches": [
                    {
                        "op": "add",
                        "id": "preferred_name",
                        "category": "preference",
                        "value": "小莓",
                        "source_turn": snapshot.turn_id,
                        "confidence": 95,
                        "base_revision": snapshot.memory_revision,
                    }
                ],
                "media_operations": [{"kind": "synthesize", "text": "answer"}],
                "frontend_operations": [{"kind": "caption", "value": "answer"}],
            }
        )

    def repair(self, snapshot: object, invalid_plan: str) -> str:
        _ = snapshot, invalid_plan
        return "{}"


class _FrontendStateBrain:
    def __init__(self) -> None:
        self.snapshots: list[BrainStateSnapshot] = []

    def plan(self, snapshot: BrainStateSnapshot, **kwargs: object) -> str:
        _ = kwargs
        self.snapshots.append(snapshot)
        operations: list[dict[str, object]] = []
        if len(self.snapshots) == 1:
            operations = [
                {"kind": "caption", "value": "已显示"},
                {"kind": "animation", "value": "talk"},
                {"kind": "ppt.load", "value": "deck-demo"},
            ]
        return json.dumps(
            {
                "response_text": "",
                "expected_revision": snapshot.revision,
                "frontend_operations": operations,
            }
        )

    def repair(self, snapshot: object, invalid_plan: str) -> str:
        _ = snapshot, invalid_plan
        return "{}"


class _CompactingBrain:
    def plan(self, snapshot: BrainStateSnapshot, **kwargs: object) -> str:
        _ = kwargs
        assert snapshot.compaction_required
        return json.dumps(
            {
                "response_text": "answer",
                "expected_revision": snapshot.revision,
                "state_operations": [
                    {
                        "kind": "context.compact",
                        "payload": {"summary": "稳定事实: 已压缩"},
                    }
                ],
            }
        )

    def repair(self, snapshot: object, invalid_plan: str) -> str:
        _ = snapshot, invalid_plan
        return "{}"


class _McpBrain:
    def plan(self, snapshot: BrainStateSnapshot, **kwargs: object) -> str:
        observations = kwargs.get("observations", ())
        assert isinstance(observations, tuple)
        if observations:
            assert observations == ('{"source":"mcp"}',)
            tools: list[dict[str, object]] = []
        else:
            tools = [{"kind": "mcp", "name": "web/search", "arguments": {}}]
        return json.dumps(
            {
                "response_text": "answer",
                "expected_revision": snapshot.revision,
                "state_operations": [],
                "media_operations": [],
                "frontend_operations": [],
                "tool_requests": tools,
                "citations": [],
                "memory_patches": [],
            }
        )

    def repair(self, snapshot: object, invalid_plan: str) -> str:
        _ = snapshot, invalid_plan
        return "{}"


class _McpTools:
    def execute(self, request: object, snapshot: object) -> str:
        _ = request, snapshot
        return '{"source":"mcp"}'


class _NoTools:
    def execute(self, request: object, snapshot: object) -> None:
        _ = request, snapshot


class _Effects:
    def __init__(self) -> None:
        self.media: list[MediaOperation] = []
        self.frontend: list[FrontendOperation] = []

    def dispatch_media(self, operation: MediaOperation, **kwargs: object) -> None:
        _ = kwargs
        self.media.append(operation)

    def dispatch_frontend(self, operation: FrontendOperation, **kwargs: object) -> None:
        _ = kwargs
        self.frontend.append(operation)


def _runtime() -> SessionRuntime:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        clock=lambda: 0,
    )
    opened = runtime.receive_comment(CommentProposal("start", _correlation("turn", 1)))
    assert opened.accepted
    return runtime


def _request(runtime: SessionRuntime) -> TaskRequest:
    turn_id = runtime.scheduler.snapshot.active_turn_id
    assert turn_id is not None
    return TaskRequest(
        task_id=TaskId("task-1"),
        session_id=SessionId("session-1"),
        turn_id=turn_id,
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(100),
        snapshot_revision=runtime.scheduler.snapshot.revision,
        idempotency_key=IdempotencyKey("answer-1"),
        kind=TaskKind.INTERACTIVE,
    )


def _result(request: TaskRequest) -> TaskResult:
    return TaskResult(
        task_id=request.task_id,
        session_id=request.session_id,
        turn_id=request.turn_id,
        snapshot_revision=request.snapshot_revision,
        effect=TaskEffect("answer", "accepted"),
    )


def _correlation(trace_id: str, sequence: int) -> EventCorrelation:
    return EventCorrelation(
        TraceId(trace_id), SessionId("session-1"), EventSequence(sequence)
    )
