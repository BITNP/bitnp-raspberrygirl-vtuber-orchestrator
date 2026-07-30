"""Immutable public outcomes emitted by one scheduler session runtime."""

from dataclasses import dataclass

from orchestrator.ids import TurnId
from orchestrator.sessions import EventCorrelation, SessionSnapshot
from orchestrator.task_reducer import TaskResult


@dataclass(frozen=True, slots=True)
class RuntimeDispatch:
    """One scheduler-approved interactive dispatch record."""

    correlation: EventCorrelation
    turn_id: TurnId


@dataclass(frozen=True, slots=True)
class RuntimeRejection:
    """A correlated refusal that did not commit a state or effect transition."""

    correlation: EventCorrelation
    reason: str


@dataclass(frozen=True, slots=True)
class RuntimeObservables:
    """Immutable runtime observability surface for transport integration."""

    snapshot: SessionSnapshot
    dispatches: tuple[RuntimeDispatch, ...]
    task_commits: tuple[TaskResult, ...]
    generated_rtp: tuple[bytes, ...]
    sound_transitions: tuple[str, ...]
    rejections: tuple[RuntimeRejection, ...]


@dataclass(frozen=True, slots=True)
class RuntimeOutcome:
    """Typed result of one external proposal or worker completion."""

    accepted: bool
    correlation: EventCorrelation
    turn_id: TurnId | None = None
