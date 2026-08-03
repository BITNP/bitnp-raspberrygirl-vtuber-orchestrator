"""Trusted execution data constructed by the turn coordinator.

The response model deliberately has no authority over identity, deadlines,
media policy, or cue capabilities.  Keeping these values in one immutable
envelope makes that boundary explicit and gives every asynchronous child task
the same result fence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.ids import SegmentId, SessionId, TurnId
    from orchestrator.sessions import StateRevision


@dataclass(frozen=True, slots=True)
class ExecutionEnvelope:
    """Reducer-authored execution identity and policy for one response turn."""

    session_id: SessionId
    turn_id: TurnId
    segment_id: SegmentId
    revision: StateRevision
    cancellation_epoch: int
    deadline_ms: int
    allowed_actions: frozenset[str]
    allowed_expressions: frozenset[str]
    replacement: bool = False

    def is_current(
        self,
        *,
        session_id: SessionId,
        revision: StateRevision,
        cancellation_epoch: int,
        now_ms: int,
        session_ended: bool,
    ) -> bool:
        """Apply the common result fence before work may produce an effect."""
        return (
            not session_ended
            and self.session_id == session_id
            and self.revision == revision
            and self.cancellation_epoch == cancellation_epoch
            and now_ms <= self.deadline_ms
        )
