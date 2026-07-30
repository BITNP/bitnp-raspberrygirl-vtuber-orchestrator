
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType

from orchestrator.ids import SessionId, TraceId, TurnId

StateRevision = NewType("StateRevision", int)

EventSequence = NewType("EventSequence", int)


@dataclass(frozen=True, slots=True)
class EventCorrelation:

    trace_id: TraceId

    session_id: SessionId

    sequence: EventSequence


@dataclass(frozen=True, slots=True)
class SchedulerEvent:

    event_type: str

    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class StartTurn:

    expected_revision: StateRevision

    event: SchedulerEvent


@dataclass(frozen=True, slots=True)
class SessionSnapshot:

    session_id: SessionId

    revision: StateRevision

    active_turn_id: TurnId | None


@dataclass(frozen=True, slots=True)
class AcceptedEvent:

    event: SchedulerEvent

    turn_id: TurnId


@unique
class TransitionRejection(StrEnum):

    STALE_REVISION = "stale_revision"

    SESSION_MISMATCH = "session_mismatch"


@dataclass(frozen=True, slots=True)
class TransitionAccepted:

    snapshot: SessionSnapshot

    accepted_event: AcceptedEvent


@dataclass(frozen=True, slots=True)
class TransitionRejected:

    snapshot: SessionSnapshot

    reason: TransitionRejection


type TransitionResult = TransitionAccepted | TransitionRejected


@dataclass(frozen=True, slots=True)
class Session:

    session_id: SessionId


class SessionManager:

    def __init__(self, *, session_id_prefix: str) -> None:
        self._session_id_prefix: str = session_id_prefix

        self._next_seq: int = 1

    def create_session(self) -> Session:
        session = Session(
            session_id=SessionId(f"{self._session_id_prefix}-{self._next_seq:04d}"),
        )

        self._next_seq += 1

        return session


class SessionScheduler:

    def __init__(self, *, session_id: SessionId, turn_id_prefix: str) -> None:
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
        return self._snapshot

    @property
    def event_history(self) -> tuple[AcceptedEvent, ...]:
        return tuple(self._event_history)

    def apply(self, transition: StartTurn) -> TransitionResult:
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
