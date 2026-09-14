"""Accepted inputs, optional speech and operation results share one lifecycle."""

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from orchestrator.brain_contracts import BrainStateSnapshot, ToolRequest
from orchestrator.brain_runtime import (
    BrainProposalError,
    build_async_response_coordinator,
)
from orchestrator.caption_timeline import CaptionTimelineCommand
from orchestrator.ids import SessionId, TraceId
from orchestrator.intent_router import IntentRouter, IntentSpec
from orchestrator.interactions import CommentProposal
from orchestrator.llm import LLMRequest
from orchestrator.response_contracts import (
    BrainDecision,
    OperationProposal,
    ResponseProposal,
    parse_response_proposal,
)
from orchestrator.response_coordinator import AsyncResponseCoordinator
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence
from orchestrator.streaming_contracts import SegmentId, StreamKey
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind, TaskState
from orchestrator.transport_control import EnvelopeCorrelation


@dataclass
class ChoiceBrain:
    initial_speech: str = ""
    final_speech: str = ""
    operation: bool = False
    invalid_final: bool = False
    observations: list[str] = field(default_factory=list)
    calls: int = 0

    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        available_operations: tuple[dict[str, object], ...],
        observation: str | None = None,
    ) -> ResponseProposal:
        _ = snapshot, available_operations
        self.calls += 1
        if observation is not None:
            self.observations.append(observation)
            if self.invalid_final:
                raise BrainProposalError
            return ResponseProposal(BrainDecision.ACCEPT, self.final_speech, None)
        return ResponseProposal(
            BrainDecision.ACCEPT,
            self.initial_speech,
            OperationProposal("lookup", {}) if self.operation else None,
        )


@dataclass
class ChoiceTool:
    delay: float = 0

    async def execute(self, request: ToolRequest, snapshot: BrainStateSnapshot) -> str:
        _ = request, snapshot
        await asyncio.sleep(self.delay)
        return "操作完成"


def choice_runtime(
    brain: ChoiceBrain, *, timeout_ms: int = 1000, delay: float = 0
) -> SessionRuntime:
    runtime = SessionRuntime.create(
        session_id=SessionId("choice"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_response_coordinator=AsyncResponseCoordinator(
            brain,
            IntentRouter(
                (
                    IntentSpec(
                        "lookup",
                        "mcp",
                        "server/tool",
                        "mcp:server/tool",
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {},
                        },
                        timeout_ms=timeout_ms,
                    ),
                )
            ),
            ChoiceTool(delay),
        ),
    )
    runtime.agent_capabilities |= {"mcp:server/tool"}
    return runtime


def correlation(sequence: int = 1) -> EventCorrelation:
    return EventCorrelation(
        TraceId(f"choice-{sequence}"), SessionId("choice"), EventSequence(sequence)
    )


def test_version_two_explicit_silence_is_distinct_from_legacy_empty_speech() -> None:
    proposal = {
        "schema_version": "2.0.0",
        "decision": "accept",
        "speech": "",
        "operation": None,
    }
    assert parse_response_proposal(json.dumps(proposal)) is not None
    del proposal["schema_version"]
    assert parse_response_proposal(json.dumps(proposal)) is None


@pytest.mark.parametrize("operation", [False, True])
def test_silent_acceptance_commits_input_and_completes_without_audio(
    *, operation: bool
) -> None:
    async def scenario() -> None:
        brain = ChoiceBrain(operation=operation)
        runtime = choice_runtime(brain)
        result = await runtime.receive_comment_async(
            CommentProposal("请记录这个背景", correlation())
        )
        assert result.accepted
        assert result.turn_id is not None
        _ = await runtime.wait_for_operation_followup(result.turn_id)
        assert not any(
            "response-tts" in r.request.task_id for r in runtime.task_registry.records
        )
        assert brain.calls == (2 if operation else 1)
        assert runtime.response_turn_state.phase == "completed"
        assert not runtime.has_active_work
        entries = runtime.interaction_ingress.data.context.snapshot.entries
        assert entries[0].text == "请记录这个背景"
        assert all(e.kind.value != "output" for e in entries)
        assert (
            runtime.take_started_timeline(result.turn_id, audio_stream_id="unused")
            is None
        )

    asyncio.run(scenario())


def test_tool_timeout_still_requests_one_bounded_final_decision() -> None:
    async def scenario() -> None:
        brain = ChoiceBrain(initial_speech="正在查询", operation=True)
        runtime = choice_runtime(brain, timeout_ms=10, delay=1)
        result = await runtime.receive_comment_async(
            CommentProposal("查一下", correlation())
        )
        assert result.turn_id is not None
        _ = await runtime.wait_for_operation_followup(result.turn_id)
        assert len(brain.observations) == 1
        assert "status=failed" in brain.observations[0]

    asyncio.run(scenario())


def test_invalid_final_response_does_not_leave_running_brain_task() -> None:
    async def scenario() -> None:
        brain = ChoiceBrain(
            initial_speech="正在查询", operation=True, invalid_final=True
        )
        runtime = choice_runtime(brain)
        result = await runtime.receive_comment_async(
            CommentProposal("查一下", correlation())
        )
        assert result.turn_id is not None
        _ = await runtime.wait_for_operation_followup(result.turn_id)
        final = next(
            r
            for r in runtime.task_registry.records
            if "brain-final" in r.request.task_id
        )
        assert final.state is TaskState.FAILED

    asyncio.run(scenario())


def test_silent_turn_retains_already_playing_output_lease() -> None:

    async def scenario() -> None:
        runtime = choice_runtime(ChoiceBrain())
        stream = StreamKey("choice", "speaker")
        lease = runtime.output_fence.activate(
            stream=stream,
            segment_id=SegmentId("old-audio"),
            correlation=EnvelopeCorrelation("old-trace", "choice", 0),
        )
        assert runtime.output_fence.can_emit(stream, lease.cancellation_epoch)
        result = await runtime.receive_comment_async(
            CommentProposal("这是补充背景", correlation())
        )
        assert result.accepted
        assert runtime.response_turn_state.phase == "completed"
        assert runtime.output_fence.can_emit(stream, lease.cancellation_epoch)
        assert runtime.output_fence.has_active_lease(stream)
        assert not runtime.active_preoutput_tts_provider_task_ids

    asyncio.run(scenario())


def test_two_speech_segments_have_distinct_caption_delivery_tasks() -> None:

    async def scenario() -> None:
        runtime = choice_runtime(ChoiceBrain(initial_speech="你好"))
        result = await runtime.receive_comment_async(
            CommentProposal("你好", correlation())
        )
        assert result.turn_id is not None
        first = runtime.schedule_caption_timeline_delivery(
            result.turn_id,
            CaptionTimelineCommand("one", "第一段", "audio", 0, 0),
            correlation=correlation(),
        )
        assert first is not None
        assert runtime.complete_caption_timeline_delivery(first[0], correlation())
        second = runtime.schedule_caption_timeline_delivery(
            result.turn_id,
            CaptionTimelineCommand("two", "第二段", "audio", 0, 320),
            correlation=correlation(),
        )
        assert second is not None
        assert first[0] != second[0]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("initial_speech", "final_speech"),
    [("", "查询结果"), ("正在查询", ""), ("正在查询", "查询结果")],
)
def test_optional_speech_before_and_after_operation(
    initial_speech: str, final_speech: str
) -> None:
    async def scenario() -> None:
        runtime = choice_runtime(
            ChoiceBrain(
                initial_speech=initial_speech, final_speech=final_speech, operation=True
            )
        )
        result = await runtime.receive_comment_async(
            CommentProposal("请查询", correlation())
        )
        assert result.turn_id is not None
        spoken: list[str] = []

        def started_synthesis(text: str, started: object) -> bool:
            assert callable(started)
            spoken.append(text)
            return bool(started())

        async def synthesize(text: str, started: object) -> bool:
            return started_synthesis(text, started)

        _ = await runtime.run_agent_tts_for_turn(
            result.turn_id, synthesize, correlation()
        )
        _ = await runtime.wait_for_operation_followup(result.turn_id)
        _ = await runtime.run_agent_tts_for_turn(
            result.turn_id, synthesize, correlation()
        )
        assert spoken == [text for text in (initial_speech, final_speech) if text]

    asyncio.run(scenario())


@dataclass
class SilentMemoryExtractor:
    replies: list[str] = field(default_factory=list)

    async def extract(self, *, user_text: str, reply_text: str) -> str:
        assert user_text == "我偏好简洁的回答"
        self.replies.append(reply_text)
        return json.dumps(
            {
                "decision": "remember",
                "key": "reply_style",
                "value": "简洁",
                "confidence": 95,
            }
        )


def test_silent_turn_can_extract_memory_without_a_fabricated_reply() -> None:
    async def scenario() -> None:
        runtime = choice_runtime(ChoiceBrain())
        extractor = SilentMemoryExtractor()
        runtime.memory_candidate_extractor = extractor
        result = await runtime.receive_comment_async(
            CommentProposal("我偏好简洁的回答", correlation())
        )
        assert result.accepted
        for _ in range(100):
            if not runtime.has_active_work:
                break
            await asyncio.sleep(0.001)
        assert extractor.replies == [""]
        assert not runtime.has_active_work
        assert all(
            entry.kind.value != "output"
            for entry in runtime.interaction_ingress.data.context.snapshot.entries
        )

    asyncio.run(scenario())


def test_silent_avatar_uses_the_trusted_dispatch_without_creating_tts() -> None:

    class Completion:
        calls: int = 0

        async def complete_json(
            self, request: LLMRequest, *, schema_name: str, schema: dict[str, object]
        ) -> str:
            _ = request, schema_name
            assert schema["required"] == [
                "schema_version",
                "decision",
                "speech",
                "operation",
            ]
            self.calls += 1
            return json.dumps(
                {
                    "schema_version": "2.0.0",
                    "decision": "accept",
                    "speech": "",
                    "operation": {
                        "intent": "avatar.cue",
                        "arguments": {"kind": "expression", "name": "nod"},
                    }
                    if self.calls == 1
                    else None,
                }
            )

    async def scenario() -> None:
        runtime = choice_runtime(ChoiceBrain())
        runtime.agent_capabilities |= {"avatar.cue"}
        sent: list[tuple[str, str]] = []

        async def dispatch(snapshot: BrainStateSnapshot, kind: str, name: str) -> bool:
            assert snapshot.session_id == "choice"
            sent.append((kind, name))
            return True

        class Avatar:
            async def execute(
                self, request: ToolRequest, snapshot: BrainStateSnapshot
            ) -> str | None:
                return await runtime.execute_avatar_tool(request, snapshot)

        runtime.avatar_dispatch = dispatch
        runtime.async_response_coordinator = build_async_response_coordinator(
            Completion(), avatar_executor=Avatar()
        )
        result = await runtime.receive_comment_async(
            CommentProposal("请点头示意", correlation())
        )
        assert result.accepted
        assert result.turn_id is not None
        _ = await runtime.wait_for_operation_followup(result.turn_id)
        assert sent == [("expression", "nod")]
        assert not any(
            "response-tts" in record.request.task_id
            for record in runtime.task_registry.records
        )
        assert runtime.response_turn_state.phase == "completed"

    asyncio.run(scenario())
