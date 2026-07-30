from dataclasses import dataclass, field

from orchestrator.ids import SessionId
from orchestrator.interactions import (
    ActionCapabilityRegistry,
    CommandId,
    InteractionAccepted,
    McpCapability,
    PresentationCommand,
    PresentationCommandKind,
    SessionInteractionReducer,
)
from orchestrator.mcp_adapters import (
    DeckDispatchIntent,
    DeckEffectDispatcher,
    DeckEffectResult,
    DeckJournalKind,
)
from orchestrator.provider_streaming import ProviderCancellationHandle
from orchestrator.sessions import SessionScheduler


def test_deck_dispatch_requires_reducer_admission_and_frontend_ack() -> None:
    reducer = _reducer()
    command = _command("load-1")
    assert reducer.reduce_presentation(command) == InteractionAccepted(
        command.command_id
    )
    executor = _DeckExecutor([DeckEffectResult.succeeded()])
    dispatcher = DeckEffectDispatcher(reducer, executor)

    outcome = dispatcher.dispatch(_intent(command), now_ms=10)

    assert outcome.accepted
    assert executor.calls == [command.command_id]
    assert outcome.completion is not None
    assert reducer.presentation_state is None
    assert reducer.reduce_presentation_result(
        outcome.completion
    ) == InteractionAccepted(
        command.command_id
    )
    assert reducer.presentation_state == ("deck-1", "v1", 1)
    assert [entry.kind for entry in dispatcher.journal] == [
        DeckJournalKind.DISPATCHED,
        DeckJournalKind.SUCCEEDED,
    ]


def test_deck_dispatch_timeout_and_ambiguous_reconcile_are_non_committing() -> None:
    reducer = _reducer()
    timed_out = _command("timeout")
    ambiguous = _command("ambiguous")
    _ = reducer.reduce_presentation(timed_out)
    _ = reducer.reduce_presentation(ambiguous)
    executor = _DeckExecutor(
        [DeckEffectResult.ambiguous(), DeckEffectResult.succeeded()]
    )
    dispatcher = DeckEffectDispatcher(reducer, executor)

    late = dispatcher.dispatch(_intent(timed_out, deadline_ms=10), now_ms=11)
    pending = dispatcher.dispatch(_intent(ambiguous), now_ms=10)
    reconciled = dispatcher.reconcile(ambiguous.command_id, now_ms=11)

    assert not late.accepted
    assert not pending.accepted
    assert reconciled.accepted
    assert executor.calls == [ambiguous.command_id, ambiguous.command_id]
    assert reconciled.completion is not None
    assert reducer.presentation_state is None


def test_duplicate_or_forged_deck_intent_never_reaches_executor() -> None:
    reducer = _reducer()
    command = _command("load-1")
    _ = reducer.reduce_presentation(command)
    executor = _DeckExecutor([DeckEffectResult.ambiguous()])
    dispatcher = DeckEffectDispatcher(reducer, executor)
    forged = PresentationCommand(
        PresentationCommandKind.LOAD, "other-deck", 1, command.command_id
    )

    first = dispatcher.dispatch(_intent(command), now_ms=10)
    replay = dispatcher.dispatch(_intent(command), now_ms=10)
    forged_outcome = dispatcher.dispatch(_intent(forged), now_ms=10)

    assert not first.accepted
    assert not replay.accepted
    assert not forged_outcome.accepted
    assert executor.calls == [command.command_id]


def test_deck_executor_is_not_a_capability_map() -> None:
    reducer = _reducer()
    command = _command("local")
    _ = reducer.reduce_presentation(command)
    executor = _DeckExecutor([DeckEffectResult.succeeded()])
    dispatcher = DeckEffectDispatcher(reducer, executor)

    assert dispatcher.dispatch(_intent(command), now_ms=0).accepted
    assert executor.calls == [command.command_id]


def _reducer() -> SessionInteractionReducer:
    return SessionInteractionReducer(
        scheduler=SessionScheduler(
            session_id=SessionId("session-1"), turn_id_prefix="turn"
        ),
        actions=ActionCapabilityRegistry(frozenset()),
        mcp_capabilities=frozenset({McpCapability.PRESENTATION_DECK}),
    )


def _command(command_id: str) -> PresentationCommand:
    return PresentationCommand(
        PresentationCommandKind.LOAD, "deck-1", 1, CommandId(command_id)
    )


def _intent(command: PresentationCommand, deadline_ms: int = 100) -> DeckDispatchIntent:
    return DeckDispatchIntent(command, deadline_ms)


@dataclass(slots=True)
class _DeckExecutor:
    results: list[DeckEffectResult]
    calls: list[CommandId] = field(default_factory=list)

    def dispatch(self, intent: DeckDispatchIntent) -> DeckEffectResult:
        self.calls.append(intent.command.command_id)
        return self.results[len(self.calls) - 1]

    def reconcile(self, intent: DeckDispatchIntent) -> DeckEffectResult:
        return self.dispatch(intent)

    async def dispatch_async(
        self, intent: DeckDispatchIntent, cancellation: ProviderCancellationHandle
    ) -> DeckEffectResult:
        _ = cancellation
        return self.dispatch(intent)
