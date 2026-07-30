
from dataclasses import dataclass

from orchestrator.ids import TurnId
from orchestrator.sessions import EventCorrelation, SessionSnapshot
from orchestrator.task_reducer import TaskResult


@dataclass(frozen=True, slots=True)
class RuntimeDispatch:

    correlation: EventCorrelation

    turn_id: TurnId


@dataclass(frozen=True, slots=True)
class RuntimeRejection:

    correlation: EventCorrelation

    reason: str


@dataclass(frozen=True, slots=True)
class RuntimeObservables:

    snapshot: SessionSnapshot

    dispatches: tuple[RuntimeDispatch, ...]

    task_commits: tuple[TaskResult, ...]

    generated_rtp: tuple[bytes, ...]

    sound_transitions: tuple[str, ...]

    rejections: tuple[RuntimeRejection, ...]


@dataclass(frozen=True, slots=True)
class RuntimeOutcome:

    accepted: bool

    correlation: EventCorrelation

    turn_id: TurnId | None = None
