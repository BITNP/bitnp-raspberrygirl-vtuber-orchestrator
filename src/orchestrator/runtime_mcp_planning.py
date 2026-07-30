"""Pure construction of presentation-scoped MCP task plans."""

from dataclasses import dataclass

from orchestrator.ids import TurnId
from orchestrator.interactions import (
    McpCapability,
    McpDispatchProposal,
    PresentationCommand,
)
from orchestrator.mcp_adapters import McpIntent
from orchestrator.sessions import SessionSnapshot
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.task_registry import (
    IdempotencyKey,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRequest,
)


@dataclass(frozen=True, slots=True)
class PresentationMcpPlanInput:
    """Inputs captured by the runtime before MCP task admission."""

    proposal: PresentationCommand
    snapshot: SessionSnapshot
    turn_id: TurnId
    data_snapshot: TaskStateSnapshot
    deadline_ms: int


@dataclass(frozen=True, slots=True)
class PresentationMcpPlan:
    """One typed task request and its deck-dispatch intent."""

    request: TaskRequest
    intent: McpIntent


def build_presentation_mcp_plan(
    plan_input: PresentationMcpPlanInput,
) -> PresentationMcpPlan:
    """Build a typed deck task plan from an active runtime snapshot."""
    command_id = plan_input.proposal.command_id
    return PresentationMcpPlan(
        request=TaskRequest(
            task_id=TaskId(f"mcp-{command_id}"),
            session_id=plan_input.snapshot.session_id,
            turn_id=plan_input.turn_id,
            parent_task_id=None,
            deadline_ms=TaskDeadlineMs(plan_input.deadline_ms),
            snapshot_revision=plan_input.snapshot.revision,
            idempotency_key=IdempotencyKey(f"mcp-{command_id}"),
            kind=TaskKind.INTERACTIVE,
            data_snapshot=plan_input.data_snapshot,
        ),
        intent=McpIntent(
            McpDispatchProposal(
                McpCapability.PRESENTATION_DECK, command_id, cancelled=False
            ),
            plan_input.proposal,
            plan_input.deadline_ms,
        ),
    )
