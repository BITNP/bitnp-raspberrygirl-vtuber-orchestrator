"""At-most-two-call Brain coordination with one isolated operation."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol

from orchestrator.modes import AnswerCandidate, AudienceInput, AudienceSource
from orchestrator.response_contracts import BrainDecision, ResponseProposal

_BLOCKING_PROVIDER_POOL = ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="bounded-provider"
)


async def run_blocking_provider[R](
    function: Callable[..., R], *args: object
) -> R:
    operation: Callable[[], R] = partial(function, *args)
    future = _BLOCKING_PROVIDER_POOL.submit(operation)
    # Polling is intentional: some supported event-loop/sandbox combinations
    # lose the cross-thread completion wakeup after the provider has returned.
    while not future.done():  # noqa: ASYNC110
        await asyncio.sleep(0.001)
    return future.result()

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.brain_contracts import BrainStateSnapshot, ToolRequest
    from orchestrator.intent_router import IntentRouter
    from orchestrator.retrieval import VersionedRetrievalProvider


class AsyncResponseBrain(Protocol):
    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        available_operations: tuple[dict[str, object], ...],
        observation: str | None = None,
    ) -> ResponseProposal: ...


class AsyncResponseToolExecutor(Protocol):
    async def execute(
        self, request: ToolRequest, snapshot: BrainStateSnapshot
    ) -> str | None: ...


class ResponseSupersededError(RuntimeError):
    """A provider returned after its immutable snapshot became stale."""


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
    retrieval: VersionedRetrievalProvider | None = None

    async def retrieve_knowledge(self, snapshot: BrainStateSnapshot) -> tuple[str, ...]:
        """Retrieve controlled local knowledge before the first Brain call."""
        if self.retrieval is None:
            return ()
        result = await run_blocking_provider(
            self.retrieval.retrieve,
            AnswerCandidate(
                AudienceInput(
                    AudienceSource(snapshot.input.source.value),
                    snapshot.input.text,
                    snapshot.input.received_at_ms,
                )
            ),
        )
        return tuple(
            (
                f"corpus={int(ref.corpus_revision)} index={int(ref.index_revision)} "
                f"source={ref.ref_id} title={ref.title} excerpt={ref.text[:4000]}"
            )
            for ref in result.refs
        )

    async def initial_response(self, snapshot: BrainStateSnapshot) -> ResponseProposal:
        return await self.brain.respond(
            snapshot, available_operations=self.router.available_operations(snapshot)
        )

    def tool_request(
        self, proposal: ResponseProposal, snapshot: BrainStateSnapshot
    ) -> ToolRequest | None:
        if proposal.operation is None:
            return None
        return self.router.request(proposal.operation, snapshot)

    def tool_timeout_ms(self, proposal: ResponseProposal) -> int | None:
        return (
            None
            if proposal.operation is None
            else self.router.timeout_for(proposal.operation.intent)
        )

    async def execute_tool(
        self, request: ToolRequest, snapshot: BrainStateSnapshot
    ) -> str | None:
        return await self.tools.execute(request, snapshot)

    def tool_request_is_current(
        self, request: ToolRequest, capabilities: frozenset[str]
    ) -> bool:
        return self.router.permits_request(request, capabilities)

    async def final_response(
        self, snapshot: BrainStateSnapshot, observation: str
    ) -> ResponseProposal:
        return await self.brain.respond(
            snapshot, available_operations=(), observation=observation
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
        if initial.decision is BrainDecision.DISCARD or initial.operation is None:
            return CoordinatedResponse(initial)
        request = self.tool_request(initial, snapshot)
        observation = "status=rejected digest=none text=操作请求未通过校验"
        if request is not None:
            try:
                result = await self.execute_tool(request, snapshot)
            except (OSError, TimeoutError, ValueError):
                result = None
            observation = (
                result
                if result is not None
                else "status=failed digest=none text=操作未成功完成"
            )
        if not current():
            raise ResponseSupersededError
        final = await self.final_response(snapshot, observation)
        if not current():
            raise ResponseSupersededError
        return CoordinatedResponse(final, request, observation)
