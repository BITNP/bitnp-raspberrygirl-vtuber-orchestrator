"""Provider access and trusted mapping for scheduler-owned response orchestration."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol

from orchestrator.modes import AnswerCandidate, AudienceInput, AudienceSource

_LOGGER = logging.getLogger(__name__)

_BLOCKING_PROVIDER_POOL = ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="bounded-provider"
)


async def run_blocking_provider[R](function: Callable[..., R], *args: object) -> R:
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
    from orchestrator.intent_router import IntentRouter, OperationExecutionPolicy
    from orchestrator.response_contracts import ResponseProposal
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
        result = await self.retrieval.retrieve_async(
            AnswerCandidate(
                AudienceInput(
                    AudienceSource(snapshot.input.source.value),
                    snapshot.input.text,
                    snapshot.input.received_at_ms,
                )
            ),
        )
        _LOGGER.debug(
            "knowledge_retrieved trace=%s session=%s seq=%s turn=%s refs=%r outcome=success",  # noqa: E501
            snapshot.input.trace_id,
            snapshot.session_id,
            snapshot.input.sequence,
            snapshot.turn_id,
            result.refs,
        )
        return tuple(
            (
                f"corpus_id={ref.corpus_id} index_id={ref.index_id} "
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

    def execution_policy(self, request: ToolRequest) -> OperationExecutionPolicy | None:
        return self.router.execution_policy(request)

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
