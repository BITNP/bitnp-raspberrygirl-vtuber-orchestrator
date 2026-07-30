"""Reducer-gated external MCP adapter dispatch and reconciliation."""

import asyncio
from collections.abc import Mapping
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
class McpResultKind(StrEnum):
    """Closed external outcomes preserved by the command journal."""

    SUCCEEDED = "succeeded"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class McpAdapterResult:
    """One normalized result returned by a scoped external adapter."""

    kind: McpResultKind

    @classmethod
    def succeeded(cls) -> "McpAdapterResult":
        """Build a correlated successful adapter result."""
        return cls(McpResultKind.SUCCEEDED)

    @classmethod
    def ambiguous(cls) -> "McpAdapterResult":
        """Build a result requiring a later reconciliation query."""
        return cls(McpResultKind.AMBIGUOUS)


@dataclass(frozen=True, slots=True)
class McpIntent:
    """One deadline-bound deck command passed to a reducer-approved adapter."""

    proposal: McpDispatchProposal
    command: PresentationCommand
    deadline_ms: int


class McpAdapter(Protocol):
    """External adapter limited to an already reducer-approved intent."""

    def execute(self, intent: McpIntent) -> McpAdapterResult:
        """Execute the idempotent intent once at the external boundary."""
        ...

    def reconcile(self, intent: McpIntent) -> McpAdapterResult:
        """Resolve an ambiguous prior invocation without redispatching it."""
        ...


@dataclass(frozen=True, slots=True)
class LocalDeckAdapter:
    """Deterministic local deck boundary used when no external deck backend exists."""

    def execute(self, intent: McpIntent) -> McpAdapterResult:
        """Emit a bounded success proposal; Frontend ACK still owns deck commitment."""
        _ = intent
        return McpAdapterResult.succeeded()

    def reconcile(self, intent: McpIntent) -> McpAdapterResult:
        """Return the same bounded local outcome without retaining payload data."""
        return self.execute(intent)

    async def execute_async(
        self, intent: McpIntent, cancellation: ProviderCancellationHandle
    ) -> McpAdapterResult:
        """Yield once so cancellation can close this local execution boundary."""
        _ = intent
        await asyncio.sleep(0)
        if cancellation.cancelled:
            return McpAdapterResult(McpResultKind.FAILED)
        return McpAdapterResult.succeeded()


@unique
class McpJournalKind(StrEnum):
    """Exact external command lifecycle records."""

    INTENT_DISPATCHED = "intent_dispatched"
    RESULT_SUCCEEDED = "result_succeeded"
    RESULT_AMBIGUOUS = "result_ambiguous"
    RESULT_FAILED = "result_failed"
    RECONCILED = "reconciled"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class McpJournalEntry:
    """An append-only, command-correlated intent or result record."""

    kind: McpJournalKind
    command_id: CommandId
    capability: McpCapability


@dataclass(frozen=True, slots=True)
class McpDispatchOutcome:
    """Reducer-visible result of one external adapter lifecycle transition."""

    accepted: bool
    completion: PresentationResult | None = None


@dataclass(slots=True)
class ScopedMcpAdapterDispatcher:
    """Execute only reducer-approved deck adapters and reconcile ambiguity."""

    reducer: SessionInteractionReducer
    adapters: Mapping[McpCapability, McpAdapter]
    _journal: list[McpJournalEntry] = field(default_factory=list)
    _pending: dict[CommandId, McpIntent] = field(default_factory=dict)
    _issued: set[CommandId] = field(default_factory=set)
    _active: dict[CommandId, ProviderCancellationHandle] = field(default_factory=dict)

    @property
    def journal(self) -> tuple[McpJournalEntry, ...]:
        """Expose immutable intent/result evidence without adapter handles."""
        return tuple(self._journal)

    def dispatch(self, intent: McpIntent, *, now_ms: int) -> McpDispatchOutcome:
        """Record intent before exactly one bounded adapter invocation."""
        adapter = self._admit(intent, now_ms=now_ms)
        if adapter is None:
            return McpDispatchOutcome(accepted=False)
        return self._consume(adapter.execute(intent), intent, reconciled=False)

    def _admit(self, intent: McpIntent, *, now_ms: int) -> McpAdapter | None:
        """Admit and journal one command before either execution boundary runs."""
        if not self.reducer.presentation_intent_is_pending(intent.command):
            self._record(McpJournalKind.REJECTED, intent)
            return None
        if intent.command.command_id in self._issued:
            self._record(McpJournalKind.REJECTED, intent)
            return None
        admission = self.reducer.reduce_mcp(intent.proposal)
        if (
            not isinstance(admission, McpDispatchAccepted)
            or intent.proposal.command_id != intent.command.command_id
        ):
            self._record(McpJournalKind.REJECTED, intent)
            return None
        adapter = self.adapters.get(intent.proposal.capability)
        if adapter is None:
            self._record(McpJournalKind.REJECTED, intent)
            return None
        if now_ms > intent.deadline_ms:
            self._record(McpJournalKind.TIMED_OUT, intent)
            return None
        self._issued.add(intent.command.command_id)
        self._record(McpJournalKind.INTENT_DISPATCHED, intent)
        return adapter

    async def dispatch_async(
        self, intent: McpIntent, *, now_ms: int
    ) -> McpDispatchOutcome:
        """Execute one local adapter under a cancellable monotonic deadline."""
        adapter = self._admit(intent, now_ms=now_ms)
        if not isinstance(adapter, LocalDeckAdapter):
            return McpDispatchOutcome(accepted=False)
        cancellation = ProviderCancellationHandle()
        self._active[intent.command.command_id] = cancellation
        try:
            result = await adapter.execute_async(intent, cancellation)
            if cancellation.cancelled:
                self._record(McpJournalKind.REJECTED, intent)
                return McpDispatchOutcome(accepted=False)
            return self._consume(result, intent, reconciled=False)
        finally:
            _ = self._active.pop(intent.command.command_id, None)

    def cancel(self, command_id: CommandId) -> bool:
        """Abort one active adapter boundary without exposing its payload or handle."""
        cancellation = self._active.get(command_id)
        if cancellation is None:
            return False
        return cancellation.cancel(reason="task_cancelled")

    def reconcile(self, command_id: CommandId, *, now_ms: int) -> McpDispatchOutcome:
        """Resolve exactly one ambiguous adapter result without a duplicate intent."""
        intent = self._pending.get(command_id)
        if intent is None:
            return McpDispatchOutcome(accepted=False)
        if now_ms > intent.deadline_ms:
            del self._pending[command_id]
            self._record(McpJournalKind.TIMED_OUT, intent)
            return McpDispatchOutcome(accepted=False)
        adapter = self.adapters[intent.proposal.capability]
        return self._consume(adapter.reconcile(intent), intent, reconciled=True)

    def _consume(
        self, result: McpAdapterResult, intent: McpIntent, *, reconciled: bool
    ) -> McpDispatchOutcome:
        match result.kind:
            case McpResultKind.SUCCEEDED:
                _ = self._pending.pop(intent.command.command_id, None)
                self._record(
                    (
                        McpJournalKind.RECONCILED
                        if reconciled
                        else McpJournalKind.RESULT_SUCCEEDED
                    ),
                    intent,
                )
                return McpDispatchOutcome(
                    accepted=True,
                    completion=PresentationResult(
                        intent.command.command_id, succeeded=True
                    ),
                )
            case McpResultKind.AMBIGUOUS:
                self._pending[intent.command.command_id] = intent
                self._record(McpJournalKind.RESULT_AMBIGUOUS, intent)
                return McpDispatchOutcome(accepted=False)
            case McpResultKind.FAILED:
                _ = self._pending.pop(intent.command.command_id, None)
                self._record(McpJournalKind.RESULT_FAILED, intent)
                return McpDispatchOutcome(accepted=False)

    def _record(self, kind: McpJournalKind, intent: McpIntent) -> None:
        self._journal.append(
            McpJournalEntry(kind, intent.command.command_id, intent.proposal.capability)
        )
