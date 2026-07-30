"""Scheduler-owned reducers for audience proposals and controlled effects."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType, final

from orchestrator.sessions import (
    EventCorrelation,
    SchedulerEvent,
    SessionScheduler,
    StartTurn,
    TransitionAccepted,
)
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.task_reducer import TaskReductionResult, TaskResult, TaskResultReducer

CommandId = NewType("CommandId", str)


@unique
class InteractionRejectionReason(StrEnum):
    """Closed reasons an interaction proposal cannot cause an effect."""

    UNSUPPORTED_ACTION = "unsupported_action"
    DUPLICATE = "duplicate"
    INVALID_PRESENTATION_STATE = "invalid_presentation_state"
    FRONTEND_REJECTED = "frontend_rejected"


@dataclass(frozen=True, slots=True)
class InteractionAccepted:
    """A reducer-approved local transition or controlled outbound command."""

    command_id: CommandId | None = None
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class InteractionRejection:
    """A refused proposal that emitted no externally observable effect."""

    reason: InteractionRejectionReason


@dataclass(frozen=True, slots=True)
class CommentProposal:
    """Normalized comments ingress carrying the canonical correlation tuple."""

    text: str
    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class ActionProposal:
    """LLM-proposed avatar action validated against a finite registry."""

    action: str
    command_id: CommandId


@final
class ActionCapabilityRegistry:
    """Finite avatar action registry; unlisted names are never dispatched."""

    def __init__(self, actions: frozenset[str]) -> None:
        """Bind the finite action names exported by Frontend."""
        self._actions = actions

    def permits(self, action: str) -> bool:
        """Return whether an action has an explicit Frontend capability mapping."""
        return action in self._actions


@unique
class PresentationCommandKind(StrEnum):
    """Closed presentation operation set supported by the Frontend protocol."""

    LOAD = "load"
    PLAY = "play"
    NAVIGATE = "navigate"


@dataclass(frozen=True, slots=True)
class PresentationCommand:
    """Typed presentation proposal, never raw LLM text or a renderer request."""

    kind: PresentationCommandKind
    deck_id: str
    page: int
    command_id: CommandId
    deck_version: str = "v1"


@dataclass(frozen=True, slots=True)
class PresentationResult:
    """Frontend acknowledgement correlated to one reducer-admitted command."""

    command_id: CommandId
    succeeded: bool


@unique
class McpCapability(StrEnum):
    """Bounded MCP capabilities eligible for reducer admission."""

    KNOWLEDGE_LOOKUP = "knowledge_lookup"
    PRESENTATION_DECK = "presentation_deck"


@dataclass(frozen=True, slots=True)
class McpDispatchProposal:
    """MCP request that must be separately registered as a scheduler task."""

    capability: McpCapability
    command_id: CommandId
    cancelled: bool


@unique
class McpDispatchRejection(StrEnum):
    """Closed MCP dispatch refusals before any adapter invocation."""

    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    CANCELLED = "cancelled"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class McpDispatchAccepted:
    """A task-registration candidate; results still require TaskResultReducer."""

    command_id: CommandId
    capability: McpCapability


@dataclass(frozen=True, slots=True)
class McpDispatchRejected:
    """MCP refusal with no direct external call."""

    reason: McpDispatchRejection


@final
class SessionInteractionReducer:
    """Sole admission point for typed interaction proposals in one session."""

    def __init__(
        self,
        *,
        scheduler: SessionScheduler,
        actions: ActionCapabilityRegistry,
        mcp_capabilities: frozenset[McpCapability],
    ) -> None:
        """Bind all scheduler-owned capability registries for one session."""
        self._scheduler = scheduler
        self._actions = actions
        self._mcp_capabilities = mcp_capabilities
        self._command_ids: set[CommandId] = set()
        self._mcp_command_ids: set[CommandId] = set()
        self._pending_presentations: dict[CommandId, PresentationCommand] = {}
        self._presentation_state: tuple[str, str, int] | None = None

    @property
    def presentation_state(self) -> tuple[str, str, int] | None:
        """Expose only the latest reducer-acknowledged deck state."""
        return self._presentation_state

    def reduce_comment(
        self,
        proposal: CommentProposal,
    ) -> InteractionAccepted | InteractionRejection:
        """Materialize comments as scheduler events before opening a turn."""
        transition = self._scheduler.apply(
            StartTurn(
                expected_revision=self._scheduler.snapshot.revision,
                event=SchedulerEvent("audience.input", proposal.correlation),
            )
        )
        match transition:
            case TransitionAccepted(accepted_event=accepted):
                return InteractionAccepted(turn_id=str(accepted.turn_id))
            case _:
                return InteractionRejection(InteractionRejectionReason.DUPLICATE)

    def reduce_action(
        self,
        proposal: ActionProposal,
    ) -> InteractionAccepted | InteractionRejection:
        """Admit only registry-mapped actions once per idempotency key."""
        if proposal.command_id in self._command_ids:
            return InteractionRejection(InteractionRejectionReason.DUPLICATE)
        if not self._actions.permits(proposal.action):
            return InteractionRejection(InteractionRejectionReason.UNSUPPORTED_ACTION)
        self._command_ids.add(proposal.command_id)
        return InteractionAccepted(command_id=proposal.command_id)

    def reduce_presentation(
        self,
        proposal: PresentationCommand,
    ) -> InteractionAccepted | InteractionRejection:
        """Admit load/play/navigate only when their minimal state precondition holds."""
        if proposal.command_id in self._command_ids:
            return InteractionRejection(InteractionRejectionReason.DUPLICATE)
        if (
            proposal.page < 1
            or proposal.deck_id.strip() == ""
            or proposal.deck_version.strip() == ""
        ):
            return InteractionRejection(
                InteractionRejectionReason.INVALID_PRESENTATION_STATE
            )
        state = self._presentation_state
        if proposal.kind is not PresentationCommandKind.LOAD and (
            state is None
            or (proposal.deck_id, proposal.deck_version) != state[:2]
        ):
            return InteractionRejection(
                InteractionRejectionReason.INVALID_PRESENTATION_STATE
            )
        self._command_ids.add(proposal.command_id)
        self._pending_presentations[proposal.command_id] = proposal
        return InteractionAccepted(command_id=proposal.command_id)

    def reduce_presentation_result(
        self,
        result: PresentationResult,
    ) -> InteractionAccepted | InteractionRejection:
        """Commit deck state only after the correlated Frontend acknowledgement."""
        proposal = self._pending_presentations.pop(result.command_id, None)
        if proposal is None:
            return InteractionRejection(InteractionRejectionReason.DUPLICATE)
        if not result.succeeded:
            return InteractionRejection(InteractionRejectionReason.FRONTEND_REJECTED)
        self._presentation_state = (
            proposal.deck_id,
            proposal.deck_version,
            proposal.page,
        )
        return InteractionAccepted(command_id=result.command_id)

    def reduce_mcp(
        self,
        proposal: McpDispatchProposal,
    ) -> McpDispatchAccepted | McpDispatchRejected:
        """Admit bounded, live MCP work whose result goes through TaskResultReducer."""
        if proposal.command_id in self._mcp_command_ids:
            return McpDispatchRejected(McpDispatchRejection.DUPLICATE)
        if proposal.cancelled:
            return McpDispatchRejected(McpDispatchRejection.CANCELLED)
        if (
            proposal.capability is not McpCapability.PRESENTATION_DECK
            or proposal.capability not in self._mcp_capabilities
        ):
            return McpDispatchRejected(McpDispatchRejection.UNSUPPORTED_CAPABILITY)
        self._mcp_command_ids.add(proposal.command_id)
        return McpDispatchAccepted(proposal.command_id, proposal.capability)

    def presentation_intent_is_pending(self, proposal: PresentationCommand) -> bool:
        """Return whether this exact reducer-admitted command still awaits an ACK."""
        return self._pending_presentations.get(proposal.command_id) == proposal

    def cancel_presentation(self, command_id: CommandId) -> None:
        """Discard a pending command whose scheduler task became terminal."""
        _ = self._pending_presentations.pop(command_id, None)

    def reduce_mcp_result(
        self,
        result: TaskResult,
        *,
        task_reducer: TaskResultReducer,
        now_ms: int,
        data_snapshot: TaskStateSnapshot | None = None,
    ) -> TaskReductionResult:
        """Route every MCP completion through live task lifecycle validation."""
        return task_reducer.reduce(
            result,
            snapshot=self._scheduler.snapshot,
            now_ms=now_ms,
            data_snapshot=data_snapshot,
        )
