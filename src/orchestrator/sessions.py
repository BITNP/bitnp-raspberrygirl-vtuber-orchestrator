"""Deterministic session IDs for the Orchestrator shell."""

from dataclasses import dataclass

from orchestrator.ids import SessionId


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
