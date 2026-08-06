import asyncio
from dataclasses import dataclass, field

import pytest

from orchestrator.brain_contracts import (
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
)
from orchestrator.intent_router import IntentRouter, IntentSpec
from orchestrator.response_contracts import (
    BrainDecision,
    OperationProposal,
    ResponseProposal,
)
from orchestrator.response_coordinator import (
    AsyncResponseCoordinator,
    ResponseSupersededError,
)


def _snapshot() -> BrainStateSnapshot:
    return BrainStateSnapshot(
        "session-1",
        "candidate-1",
        1,
        0,
        AudienceInput("session-1", "trace-1", 1, AudienceSource.ASR, 1, "查询"),
        "",
        (),
        "",
        frozenset({"mcp:web/search"}),
    )


@dataclass
class _Brain:
    responses: list[ResponseProposal]
    operations: list[tuple[dict[str, object], ...]] = field(default_factory=list)

    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        available_operations: tuple[dict[str, object], ...],
        observation: str | None = None,
    ) -> ResponseProposal:
        _ = snapshot, observation
        self.operations.append(available_operations)
        return self.responses.pop(0)


@dataclass
class _Tools:
    requests: int = 0

    async def execute(self, request: object, snapshot: BrainStateSnapshot) -> str:
        _ = request, snapshot
        self.requests += 1
        return "status=success digest=sha256:x text=晴"


def _coordinator(brain: _Brain, tools: _Tools) -> AsyncResponseCoordinator:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 64}},
    }
    return AsyncResponseCoordinator(
        brain,
        IntentRouter(
            (
                IntentSpec(
                    "mcp.web_search", "mcp", "web/search", "mcp:web/search", schema
                ),
            )
        ),
        tools,
    )


def test_no_operation_uses_one_brain_call() -> None:
    brain = _Brain([ResponseProposal(BrainDecision.ACCEPT, "您好", None)])
    result = asyncio.run(_coordinator(brain, _Tools()).respond(_snapshot()))
    assert result.proposal.speech == "您好"
    assert len(brain.operations) == 1


def test_one_operation_uses_one_tool_and_at_most_two_brain_calls() -> None:
    brain = _Brain(
        [
            ResponseProposal(
                BrainDecision.ACCEPT,
                "我来查询",
                OperationProposal("mcp.web_search", {"query": "天气"}),
            ),
            ResponseProposal(BrainDecision.ACCEPT, "明天晴", None),
        ]
    )
    tools = _Tools()
    result = asyncio.run(_coordinator(brain, tools).respond(_snapshot()))
    assert result.proposal.speech == "明天晴"
    assert tools.requests == 1
    assert len(brain.operations) == 2
    assert brain.operations[1] == ()


def test_stale_candidate_cannot_commit() -> None:
    coordinator = _coordinator(
        _Brain([ResponseProposal(BrainDecision.ACCEPT, "答案", None)]), _Tools()
    )
    with pytest.raises(ResponseSupersededError):
        _ = asyncio.run(coordinator.respond(_snapshot(), is_current=lambda: False))
