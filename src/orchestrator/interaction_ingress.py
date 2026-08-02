from dataclasses import dataclass, field
from hashlib import sha256
from os import environ
from pathlib import Path

from orchestrator.ids import SessionId, TraceId
from orchestrator.interactions import (
    ActionCapabilityRegistry,
    ActionProposal,
    CommentProposal,
    InteractionAccepted,
    InteractionRejection,
    McpCapability,
    McpDispatchAccepted,
    McpDispatchProposal,
    McpDispatchRejected,
    PresentationCommand,
    PresentationResult,
    SessionInteractionReducer,
)
from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.memory_store import MarkdownMemoryStore
from orchestrator.profile_store import JsonVoiceProfileStore
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.session_data import ProfilePersistence, SessionDataState
from orchestrator.sessions import EventCorrelation, EventSequence, SessionScheduler
from orchestrator.voice_profile_service import VoiceProfileService


@dataclass(frozen=True, slots=True)
class SessionInteractionIngress:
    data: SessionDataState

    profiles: VoiceProfileService

    reducer: SessionInteractionReducer

    _consumed_correlations: set[EventCorrelation] = field(default_factory=set)

    @classmethod
    def create(cls, scheduler: SessionScheduler) -> "SessionInteractionIngress":
        session_root = session_storage_root(scheduler.snapshot.session_id)

        data = SessionDataState.create(
            session_id=scheduler.snapshot.session_id,
            retrieval=RetrievalFixtureProvider(refs=()),
            memory_store=MarkdownMemoryStore(session_root / "memory.md"),
            profile_persistence=ProfilePersistence(
                store=JsonVoiceProfileStore(session_root / "voice-profiles.json"),
                vault_directory=session_root / "voice-templates",
            ),
        )

        return cls(
            data=data,
            profiles=data.profiles,
            reducer=SessionInteractionReducer(
                scheduler=scheduler,
                actions=ActionCapabilityRegistry(
                    frozenset({"breathe", "dance", "explain_point", "speak"})
                ),
                mcp_capabilities=frozenset({McpCapability.PRESENTATION_DECK}),
            ),
        )

    def receive_comment(
        self,
        *,
        text: str,
        correlation: EventCorrelation,
    ) -> InteractionAccepted | InteractionRejection:
        return self.reducer.reduce_comment(CommentProposal(text, correlation))

    def receive_action(
        self,
        proposal: ActionProposal,
    ) -> InteractionAccepted | InteractionRejection:
        return self.reducer.reduce_action(proposal)

    def receive_presentation(
        self,
        proposal: PresentationCommand,
    ) -> InteractionAccepted | InteractionRejection:
        return self.reducer.reduce_presentation(proposal)

    def receive_presentation_result(
        self,
        result: PresentationResult,
    ) -> InteractionAccepted | InteractionRejection:
        return self.reducer.reduce_presentation_result(result)

    def receive_mcp(
        self,
        proposal: McpDispatchProposal,
    ) -> McpDispatchAccepted | McpDispatchRejected:
        return self.reducer.reduce_mcp(proposal)

    def receive_control(self, raw_message: str) -> bool:
        proposal = parse_comment_proposal(raw_message)

        if proposal is None:
            return False

        correlation = proposal.correlation

        if correlation in self._consumed_correlations:
            return True

        outcome = self.receive_comment(
            text=proposal.text,
            correlation=correlation,
        )

        if isinstance(outcome, InteractionAccepted):
            self._consumed_correlations.add(correlation)

        return True


def _state_root() -> Path:
    return Path(environ.get("ORCHESTRATOR_STATE_DIR", ".orchestrator-state"))


def session_storage_root(session_id: SessionId) -> Path:
    storage_key = sha256(str(session_id).encode()).hexdigest()

    return _state_root() / storage_key


def parse_comment_proposal(raw_message: str) -> CommentProposal | None:
    try:
        value = parse_json_value(raw_message)

    except JsonBoundaryError:
        return None

    if not isinstance(value, dict):
        return None

    if value.get("event_type") != "audience.input" or value.get("source") != "comments":
        return None

    data = value.get("data")

    if not isinstance(data, dict):
        return None

    text = data.get("text")

    trace_id = value.get("trace_id")

    session_id = value.get("session_id")

    sequence = value.get("seq")

    if (
        not isinstance(text, str)
        or text.strip() == ""
        or not isinstance(trace_id, str)
        or not isinstance(session_id, str)
        or type(sequence) is not int
        or sequence < 0
    ):
        return None

    return CommentProposal(
        text=text,
        correlation=EventCorrelation(
            trace_id=TraceId(trace_id),
            session_id=SessionId(session_id),
            sequence=EventSequence(sequence),
        ),
    )
