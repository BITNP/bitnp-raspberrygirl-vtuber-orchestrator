import asyncio
from dataclasses import dataclass, field

import pytest

from orchestrator.brain_contracts import (
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
)
from orchestrator.intent_router import IntentRouter, IntentSpec
from orchestrator.response_contracts import ResponseProposal
from orchestrator.response_coordinator import (
    AsyncResponseCoordinator,
    ResponseSupersededError,
)


def _snapshot() -> BrainStateSnapshot:
    return BrainStateSnapshot(
        session_id="session-1",
        turn_id="turn-1",
        revision=1,
        cancellation_epoch=0,
        input=AudienceInput("session-1", "trace-1", 1, AudienceSource.ASR, 1, "查询"),
        context_summary="",
        recent_context=(),
        memory_markdown="",
        capabilities=frozenset({"knowledge.lookup"}),
    )


@dataclass
class _Brain:
    responses: list[ResponseProposal]
    allowed: list[frozenset[str]] = field(default_factory=list)

    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        allowed_intents: frozenset[str],
        observations: tuple[str, ...] = (),
    ) -> ResponseProposal:
        _ = snapshot, observations
        self.allowed.append(allowed_intents)
        return self.responses.pop(0)


class _Tools:
    async def execute(self, request: object, snapshot: BrainStateSnapshot) -> str:
        _ = request, snapshot
        return "检索结果"


class _FailingTools:
    async def execute(self, request: object, snapshot: BrainStateSnapshot) -> str:
        _ = request, snapshot
        raise OSError


def test_tool_intent_has_one_final_answer_call_without_reopening_tools() -> None:
    brain = _Brain(
        [ResponseProposal("", "knowledge"), ResponseProposal("答案", "answer")]
    )
    coordinator = AsyncResponseCoordinator(
        brain,
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
        _Tools(),
    )

    result = asyncio.run(coordinator.respond(_snapshot()))

    assert result.proposal.reply == "答案"
    assert result.observation == "检索结果"
    assert brain.allowed == [frozenset({"answer", "knowledge"}), frozenset({"answer"})]


def test_stale_turn_cannot_commit_a_model_result() -> None:
    coordinator = AsyncResponseCoordinator(
        _Brain([ResponseProposal("答案", "answer")]), IntentRouter(()), _Tools()
    )

    with pytest.raises(ResponseSupersededError):
        _ = asyncio.run(coordinator.respond(_snapshot(), is_current=lambda: False))


def test_failed_tool_still_has_exactly_one_tools_disabled_final_answer() -> None:
    brain = _Brain(
        [ResponseProposal("", "knowledge"), ResponseProposal("暂不可用", "answer")]
    )
    coordinator = AsyncResponseCoordinator(
        brain,
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
        _FailingTools(),
    )

    result = asyncio.run(coordinator.respond(_snapshot()))

    assert result.proposal == ResponseProposal("暂不可用", "answer")
    assert result.observation == "工具调用未成功完成。请基于已知信息简短说明。"
    assert brain.allowed == [frozenset({"answer", "knowledge"}), frozenset({"answer"})]
