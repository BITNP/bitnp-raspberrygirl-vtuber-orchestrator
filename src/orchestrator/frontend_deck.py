from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from orchestrator.mcp_adapters import (
    DeckDispatchIntent,
    DeckEffectResult,
    DeckEffectResultKind,
)

if TYPE_CHECKING:
    from orchestrator.ids import SessionId
    from orchestrator.provider_streaming import ProviderCancellationHandle


class FrontendDeckRoute(Protocol):
    async def __call__(
        self,
        session_id: SessionId,
        intent: DeckDispatchIntent,
        cancellation: ProviderCancellationHandle,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class FrontendDeckEffectExecutor:
    session_id: SessionId
    route: FrontendDeckRoute

    def dispatch(self, intent: DeckDispatchIntent) -> DeckEffectResult:
        _ = intent
        return DeckEffectResult(DeckEffectResultKind.FAILED)

    def reconcile(self, intent: DeckDispatchIntent) -> DeckEffectResult:
        _ = intent
        return DeckEffectResult(DeckEffectResultKind.FAILED)

    async def dispatch_async(
        self, intent: DeckDispatchIntent, cancellation: ProviderCancellationHandle
    ) -> DeckEffectResult:
        succeeded = await self.route(self.session_id, intent, cancellation)
        return DeckEffectResult(
            DeckEffectResultKind.SUCCEEDED
            if succeeded
            else DeckEffectResultKind.FAILED
        )
