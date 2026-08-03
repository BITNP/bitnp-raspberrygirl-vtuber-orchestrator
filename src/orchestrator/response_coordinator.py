"""Two-stage response coordination with no model-controlled effects.

This is intentionally transport-free.  SessionRuntime owns task admission and
uses this coordinator only after it has captured an immutable state snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from orchestrator.response_contracts import ResponseProposal

if TYPE_CHECKING:
    from orchestrator.agent_pipeline import BrainStateSnapshot, ToolRequest
    from orchestrator.intent_router import IntentRouter


class AsyncResponseBrain(Protocol):
    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        allowed_intents: frozenset[str],
        observations: tuple[str, ...] = (),
    ) -> ResponseProposal: ...


class AsyncResponseToolExecutor(Protocol):
    async def execute(
        self, request: ToolRequest, snapshot: BrainStateSnapshot
    ) -> str | None: ...


@dataclass(frozen=True, slots=True)
class CoordinatedResponse:
    proposal: ResponseProposal
    tool_request: ToolRequest | None = None
    observation: str | None = None


@dataclass(slots=True)
class AsyncResponseCoordinator:
    brain: AsyncResponseBrain
    router: IntentRouter
    tools: AsyncResponseToolExecutor

    async def respond(self, snapshot: BrainStateSnapshot) -> CoordinatedResponse:
        allowed = self.router.allowed_intents(snapshot)
        initial = await self.brain.respond(snapshot, allowed_intents=allowed)
        if initial.intent == "answer":
            return CoordinatedResponse(initial)
        request = self.router.request(initial.intent, snapshot)
        if request is None:
            return CoordinatedResponse(
                ResponseProposal("抱歉, 这项功能暂时不可用。", "answer")
            )
        try:
            observation = await self.tools.execute(request, snapshot)
        except (OSError, TimeoutError, ValueError):
            observation = None
        if observation is None:
            observation = "工具调用未成功完成。请基于已知信息简短说明。"
        final = await self.brain.respond(
            snapshot,
            allowed_intents=frozenset({"answer"}),
            observations=(observation,),
        )
        if final.intent != "answer":
            final = ResponseProposal(final.reply, "answer", final.used_text_fallback)
        return CoordinatedResponse(final, request, observation)
