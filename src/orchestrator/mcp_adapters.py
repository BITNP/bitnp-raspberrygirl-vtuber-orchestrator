"""Deadline-bound dispatch for the sole active MCP capability: presentation decks."""

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Protocol

from orchestrator.interactions import (
    CommandId,
    McpCapability,
    McpDispatchAccepted,
    McpDispatchProposal,
    PresentationCommand,
    PresentationResult,
    SessionInteractionReducer,
)
from orchestrator.provider_streaming import ProviderCancellationHandle


@unique
class DeckEffectResultKind(StrEnum):
    """Terminal outcome reported by the deck effect boundary."""
    SUCCEEDED = "succeeded"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DeckEffectResult:
    """Typed deck effect execution result."""
    kind: DeckEffectResultKind

    @classmethod
    def succeeded(cls) -> "DeckEffectResult":
        """Construct a successful result."""
        return cls(DeckEffectResultKind.SUCCEEDED)

    @classmethod
    def ambiguous(cls) -> "DeckEffectResult":
        """Construct a result that requires deck reconciliation."""
        return cls(DeckEffectResultKind.AMBIGUOUS)


@dataclass(frozen=True, slots=True)
class DeckDispatchIntent:
    """A reducer-approved deck command with a dispatch deadline."""
    command: PresentationCommand
    deadline_ms: int


class DeckEffectExecutor(Protocol):
    """Concrete deck side-effect boundary."""

    def dispatch(self, intent: DeckDispatchIntent) -> DeckEffectResult:
        """Issue one deck command."""
        ...

    def reconcile(self, intent: DeckDispatchIntent) -> DeckEffectResult:
        """Resolve one ambiguous deck command."""
        ...

    async def dispatch_async(
        self, intent: DeckDispatchIntent, cancellation: ProviderCancellationHandle
    ) -> DeckEffectResult:
        """Issue one cancellable deck command."""
        ...


@dataclass(frozen=True, slots=True)
class LocalDeckEffectExecutor:
    """Local bounded deck executor used by the runtime composition root."""

    def dispatch(self, intent: DeckDispatchIntent) -> DeckEffectResult:
        """Issue the local deck command."""
        _ = intent
        return DeckEffectResult.succeeded()

    def reconcile(self, intent: DeckDispatchIntent) -> DeckEffectResult:
        """Reissue the local deck command during reconciliation."""
        return self.dispatch(intent)

    async def dispatch_async(
        self, intent: DeckDispatchIntent, cancellation: ProviderCancellationHandle
    ) -> DeckEffectResult:
        """Issue the local deck command while observing cancellation."""
        _ = intent
        await asyncio.sleep(0)
        if cancellation.cancelled:
            return DeckEffectResult(DeckEffectResultKind.FAILED)
        return DeckEffectResult.succeeded()


@unique
class DeckJournalKind(StrEnum):
    """Redacted lifecycle facts emitted for deck effects."""
    DISPATCHED = "dispatched"
    SUCCEEDED = "succeeded"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"
    RECONCILED = "reconciled"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DeckJournalEntry:
    """Redacted deck effect journal record."""
    kind: DeckJournalKind
    command_id: CommandId


@dataclass(frozen=True, slots=True)
class DeckDispatchOutcome:
    """Deck dispatch result that still requires the frontend acknowledgement."""
    accepted: bool
    completion: PresentationResult | None = None


@dataclass(slots=True)
class DeckEffectDispatcher:
    """Own the typed deck effect lifecycle after reducer admission."""

    reducer: SessionInteractionReducer
    executor: DeckEffectExecutor
    _journal: list[DeckJournalEntry] = field(default_factory=list)
    _pending: dict[CommandId, DeckDispatchIntent] = field(default_factory=dict)
    _issued: set[CommandId] = field(default_factory=set)
    _active: dict[CommandId, ProviderCancellationHandle] = field(default_factory=dict)

    @property
    def journal(self) -> tuple[DeckJournalEntry, ...]:
        """Return immutable redacted deck lifecycle records."""
        return tuple(self._journal)

    def dispatch(
        self, intent: DeckDispatchIntent, *, now_ms: int
    ) -> DeckDispatchOutcome:
        """Synchronously issue a reducer-approved deck effect."""
        if not self._admit(intent, now_ms=now_ms):
            return DeckDispatchOutcome(accepted=False)
        return self._consume(self.executor.dispatch(intent), intent, reconciled=False)

    async def dispatch_async(
        self, intent: DeckDispatchIntent, *, now_ms: int
    ) -> DeckDispatchOutcome:
        """Asynchronously issue a reducer-approved deck effect."""
        if not self._admit(intent, now_ms=now_ms):
            return DeckDispatchOutcome(accepted=False)
        cancellation = ProviderCancellationHandle()
        self._active[intent.command.command_id] = cancellation
        try:
            result = await self.executor.dispatch_async(intent, cancellation)
            if cancellation.cancelled:
                self._record(DeckJournalKind.REJECTED, intent)
                return DeckDispatchOutcome(accepted=False)
            return self._consume(result, intent, reconciled=False)
        finally:
            _ = self._active.pop(intent.command.command_id, None)

    def cancel(self, command_id: CommandId) -> bool:
        """Cancel an active deck effect if it is still running."""
        cancellation = self._active.get(command_id)
        if cancellation is None:
            return False
        return cancellation.cancel(reason="task_cancelled")

    def reconcile(
        self, command_id: CommandId, *, now_ms: int
    ) -> DeckDispatchOutcome:
        """Resolve a previously ambiguous deck effect before its deadline."""
        intent = self._pending.get(command_id)
        if intent is None:
            return DeckDispatchOutcome(accepted=False)
        if now_ms > intent.deadline_ms:
            del self._pending[command_id]
            self._record(DeckJournalKind.TIMED_OUT, intent)
            return DeckDispatchOutcome(accepted=False)
        return self._consume(self.executor.reconcile(intent), intent, reconciled=True)

    def _admit(self, intent: DeckDispatchIntent, *, now_ms: int) -> bool:
        command = intent.command
        if not self.reducer.presentation_intent_is_pending(command):
            self._record(DeckJournalKind.REJECTED, intent)
            return False
        if command.command_id in self._issued:
            self._record(DeckJournalKind.REJECTED, intent)
            return False
        admission = self.reducer.reduce_mcp(
            McpDispatchProposal(
                McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
            )
        )
        match admission:
            case McpDispatchAccepted(
                command_id=command_id, capability=McpCapability.PRESENTATION_DECK
            ) if command_id == command.command_id:
                pass
            case _:
                self._record(DeckJournalKind.REJECTED, intent)
                return False
        if now_ms > intent.deadline_ms:
            self._record(DeckJournalKind.TIMED_OUT, intent)
            return False
        self._issued.add(command.command_id)
        self._record(DeckJournalKind.DISPATCHED, intent)
        return True

    def _consume(
        self,
        result: DeckEffectResult,
        intent: DeckDispatchIntent,
        *,
        reconciled: bool,
    ) -> DeckDispatchOutcome:
        match result.kind:
            case DeckEffectResultKind.SUCCEEDED:
                _ = self._pending.pop(intent.command.command_id, None)
                kind = (
                    DeckJournalKind.RECONCILED
                    if reconciled
                    else DeckJournalKind.SUCCEEDED
                )
                self._record(kind, intent)
                return DeckDispatchOutcome(
                    accepted=True,
                    completion=PresentationResult(
                        intent.command.command_id, succeeded=True
                    ),
                )
            case DeckEffectResultKind.AMBIGUOUS:
                self._pending[intent.command.command_id] = intent
                self._record(DeckJournalKind.AMBIGUOUS, intent)
                return DeckDispatchOutcome(accepted=False)
            case DeckEffectResultKind.FAILED:
                _ = self._pending.pop(intent.command.command_id, None)
                self._record(DeckJournalKind.FAILED, intent)
                return DeckDispatchOutcome(accepted=False)

    def _record(self, kind: DeckJournalKind, intent: DeckDispatchIntent) -> None:
        self._journal.append(DeckJournalEntry(kind, intent.command.command_id))
