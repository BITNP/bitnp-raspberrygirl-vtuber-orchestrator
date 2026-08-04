"""Two-stage response coordination with no model-controlled effects.

This is intentionally transport-free.  SessionRuntime owns task admission and
uses this coordinator only after it has captured an immutable state snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from orchestrator.response_contracts import ResponseProposal

if TYPE_CHECKING:
    from collections.abc import Callable

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


class ResponseSupersededError(RuntimeError):
    """A provider returned after its immutable turn snapshot became stale."""


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

    async def initial_response(
        self, snapshot: BrainStateSnapshot
    ) -> ResponseProposal:
        """Run only the initial, intent-selecting model call."""
        return await self.brain.respond(
            snapshot, allowed_intents=self.router.allowed_intents(snapshot)
        )

    def tool_request(
        self, proposal: ResponseProposal, snapshot: BrainStateSnapshot
    ) -> ToolRequest | None:
        """Build the trusted request for a previously accepted intent."""
        if proposal.intent == "answer":
            return None
        return self.router.request(proposal.intent, snapshot)

    def tool_timeout_ms(self, proposal: ResponseProposal) -> int | None:
        """Expose the registered, model-independent budget for a tool turn."""
        if proposal.intent == "answer":
            return None
        return self.router.timeout_for(proposal.intent)

    async def execute_tool(
        self, request: ToolRequest, snapshot: BrainStateSnapshot
    ) -> str | None:
        """Execute one already-authorized tool request without model authority."""
        return await self.tools.execute(request, snapshot)

    async def final_response(
        self, snapshot: BrainStateSnapshot, observation: str
    ) -> ResponseProposal:
        """Run the sole post-observation model call with tools disabled."""
        final = await self.brain.respond(
            snapshot,
            allowed_intents=frozenset({"answer"}),
            observations=(observation,),
        )
        return (
            final
            if final.intent == "answer"
            else ResponseProposal(final.reply, "answer", final.used_text_fallback)
        )

    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        is_current: Callable[[], bool] | None = None,
    ) -> CoordinatedResponse:
        current = is_current if is_current is not None else lambda: True
        initial = await self.initial_response(snapshot)
        if not current():
            raise ResponseSupersededError
        if initial.intent == "answer":
            return CoordinatedResponse(initial)
        request = self.tool_request(initial, snapshot)
        if request is None:
            return CoordinatedResponse(
                ResponseProposal("抱歉, 这项功能暂时不可用。", "answer")
            )
        try:
            observation = await self.execute_tool(request, snapshot)
        except (OSError, TimeoutError, ValueError):
            observation = None
        if not current():
            raise ResponseSupersededError
        if observation is None:
            observation = "工具调用未成功完成。请基于已知信息简短说明。"
        final = await self.final_response(snapshot, observation)
        if not current():
            raise ResponseSupersededError
        return CoordinatedResponse(final, request, observation)
