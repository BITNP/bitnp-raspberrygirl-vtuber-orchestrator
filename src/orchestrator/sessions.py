"""Deterministic session IDs and local scheduler state for the Orchestrator."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType

from orchestrator.ids import SessionId, TraceId, TurnId

StateRevision = NewType("StateRevision", int)
EventSequence = NewType("EventSequence", int)


@dataclass(frozen=True, slots=True)
class EventCorrelation:
    """Correlation retained for every scheduler-accepted event."""

    trace_id: TraceId
    session_id: SessionId
    sequence: EventSequence


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    """Local event proposal eligible for scheduler acceptance."""

    event_type: str
    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class StartTurn:
    """Propose opening the next monotonic turn from one input event."""

    expected_revision: StateRevision
    event: SchedulerEvent


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Materialized session state emitted only after an accepted transition."""

    session_id: SessionId
    revision: StateRevision
    active_turn_id: TurnId | None


@dataclass(frozen=True, slots=True)
class AcceptedEvent:
    """Append-only accepted event record with its scheduler-assigned turn."""

    event: SchedulerEvent
    turn_id: TurnId


@unique
class TransitionRejection(StrEnum):
    """Closed reasons a scheduler transition cannot commit."""

    STALE_REVISION = "stale_revision"
    SESSION_MISMATCH = "session_mismatch"


@dataclass(frozen=True, slots=True)
class TransitionAccepted:
    """Result returned when the scheduler commits a new snapshot."""

    snapshot: SessionSnapshot
    accepted_event: AcceptedEvent


@dataclass(frozen=True, slots=True)
class TransitionRejected:
    """Result returned when a transition cannot alter the current snapshot."""

    snapshot: SessionSnapshot
    reason: TransitionRejection


type TransitionResult = TransitionAccepted | TransitionRejected


@dataclass(frozen=True, slots=True)
class Session:
    """Created Orchestrator session."""

    session_id: SessionId


class SessionManager:
    """Allocates deterministic session IDs for local tests."""

    def __init__(self, *, session_id_prefix: str) -> None:
        """Create a session manager with a stable ID prefix."""
        self._session_id_prefix: str = session_id_prefix
        self._next_seq: int = 1

    def create_session(self) -> Session:
        """Create the next session with a monotonic sequence number."""
        session = Session(
            session_id=SessionId(f"{self._session_id_prefix}-{self._next_seq:04d}"),
        )
        self._next_seq += 1
        return session


class SessionScheduler:
    """Owns one session's mutable transition cursor and immutable public state."""

    def __init__(self, *, session_id: SessionId, turn_id_prefix: str) -> None:
        """Create an empty local scheduler with deterministic turn identifiers."""
        self._turn_id_prefix: str = turn_id_prefix
        self._turn_sequence: int = 0
        self._snapshot: SessionSnapshot = SessionSnapshot(
            session_id=session_id,
            revision=StateRevision(0),
            active_turn_id=None,
        )
        self._event_history: list[AcceptedEvent] = []

    @property
    def snapshot(self) -> SessionSnapshot:
        """Return the current immutable materialized session state."""
        return self._snapshot

    @property
    def event_history(self) -> tuple[AcceptedEvent, ...]:
        """Return accepted event records in append-only acceptance order."""
        return tuple(self._event_history)

    def apply(self, transition: StartTurn) -> TransitionResult:
        """Commit an approved turn transition only against the current revision."""
        if transition.expected_revision != self._snapshot.revision:
            return TransitionRejected(
                snapshot=self._snapshot,
                reason=TransitionRejection.STALE_REVISION,
            )
        if transition.event.correlation.session_id != self._snapshot.session_id:
            return TransitionRejected(
                snapshot=self._snapshot,
                reason=TransitionRejection.SESSION_MISMATCH,
            )
        self._turn_sequence += 1
        turn_id = TurnId(f"{self._turn_id_prefix}-{self._turn_sequence:04d}")
        accepted_event = AcceptedEvent(event=transition.event, turn_id=turn_id)
        snapshot = SessionSnapshot(
            session_id=self._snapshot.session_id,
            revision=StateRevision(self._snapshot.revision + 1),
            active_turn_id=turn_id,
        )
        self._event_history.append(accepted_event)
        self._snapshot = snapshot
        return TransitionAccepted(snapshot=snapshot, accepted_event=accepted_event)
