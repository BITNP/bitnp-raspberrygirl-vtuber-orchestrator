
from orchestrator.ids import SessionId, TraceId
from orchestrator.interactions import (
    ActionCapabilityRegistry,
    ActionProposal,
    CommandId,
    CommentProposal,
    InteractionAccepted,
    InteractionRejection,
    InteractionRejectionReason,
    McpCapability,
    McpDispatchAccepted,
    McpDispatchProposal,
    McpDispatchRejected,
    McpDispatchRejection,
    PresentationCommand,
    PresentationCommandKind,
    PresentationResult,
    SessionInteractionReducer,
)
from orchestrator.sessions import EventCorrelation, EventSequence, SessionScheduler


def test_comment_ingress_and_action_rejection_enter_reducer() -> None:
    # Given: a session reducer with only its finite avatar action allowlist.


    reducer = _reducer()

    # When: a comment and unsupported LLM-proposed action arrive.

    comment = reducer.reduce_comment(_comment())

    action = reducer.reduce_action(
        ActionProposal("inject_external_call", CommandId("action-1"))
    )

    allowed = reducer.reduce_action(ActionProposal("hello", CommandId("action-2")))

    replayed = reducer.reduce_action(ActionProposal("hello", CommandId("action-2")))

    # Then: the comment opens a correlated turn but the action emits no effect.

    assert isinstance(comment, InteractionAccepted)

    assert comment.turn_id is not None

    assert action == InteractionRejection(InteractionRejectionReason.UNSUPPORTED_ACTION)

    assert allowed == InteractionAccepted(command_id=CommandId("action-2"))

    assert replayed == InteractionRejection(InteractionRejectionReason.DUPLICATE)


def test_failed_presentation_result_and_duplicate_command_are_rejected() -> None:
    # Given: a load command accepted by the presentation capability.


    reducer = _reducer()

    command = PresentationCommand(
        PresentationCommandKind.LOAD,
        "deck-1",
        1,
        CommandId("cmd-1"),
    )

    accepted = reducer.reduce_presentation(command)

    # When: Frontend rejects it and an idempotent replay is proposed.

    failed = reducer.reduce_presentation_result(
        PresentationResult(CommandId("cmd-1"), succeeded=False)
    )

    duplicate = reducer.reduce_presentation(command)

    # Then: failure is recorded without a deck state and replay has no side effect.

    assert accepted == InteractionAccepted(command_id=CommandId("cmd-1"))

    assert failed == InteractionRejection(InteractionRejectionReason.FRONTEND_REJECTED)

    assert duplicate == InteractionRejection(InteractionRejectionReason.DUPLICATE)


def test_presentation_commits_matching_deck_version_and_page_only_after_ack() -> None:
    # Given: a reducer with no committed deck state.


    reducer = _reducer()

    load = PresentationCommand(
        PresentationCommandKind.LOAD,
        "deck-1",
        1,
        CommandId("load-1"),
        deck_version="v1",
    )

    # When: an acknowledged load is followed by a same-version navigation.

    admitted_load = reducer.reduce_presentation(load)

    committed_load = reducer.reduce_presentation_result(
        PresentationResult(CommandId("load-1"), succeeded=True)
    )

    navigation = reducer.reduce_presentation(
        PresentationCommand(
            PresentationCommandKind.NAVIGATE,
            "deck-1",
            2,
            CommandId("navigate-2"),
            deck_version="v1",
        )
    )

    committed_navigation = reducer.reduce_presentation_result(
        PresentationResult(CommandId("navigate-2"), succeeded=True)
    )

    # Then: reducer state reflects only the correlated acknowledged deck state.

    assert admitted_load == InteractionAccepted(command_id=CommandId("load-1"))

    assert committed_load == InteractionAccepted(command_id=CommandId("load-1"))

    assert navigation == InteractionAccepted(command_id=CommandId("navigate-2"))

    assert committed_navigation == InteractionAccepted(
        command_id=CommandId("navigate-2")
    )

    assert reducer.presentation_state == ("deck-1", "v1", 2)


def test_presentation_rejects_invalid_page_and_wrong_loaded_deck_version() -> None:
    # Given: an acknowledged version-one deck load.


    reducer = _reducer()

    load = PresentationCommand(
        PresentationCommandKind.LOAD,
        "deck-1",
        1,
        CommandId("load-1"),
        deck_version="v1",
    )

    _ = reducer.reduce_presentation(load)

    _ = reducer.reduce_presentation_result(
        PresentationResult(load.command_id, succeeded=True)
    )

    # When: a zero page and a different deck version are proposed.

    invalid_page = reducer.reduce_presentation(
        PresentationCommand(
            PresentationCommandKind.NAVIGATE,
            "deck-1",
            0,
            CommandId("page-0"),
            deck_version="v1",
        )
    )

    wrong_version = reducer.reduce_presentation(
        PresentationCommand(
            PresentationCommandKind.PLAY,
            "deck-1",
            1,
            CommandId("play-v2"),
            deck_version="v2",
        )
    )

    # Then: neither proposal becomes a pending external intent.

    expected = InteractionRejection(
        InteractionRejectionReason.INVALID_PRESENTATION_STATE
    )

    assert invalid_page == expected

    assert wrong_version == expected


def test_mcp_requires_allowlist_and_cancellation_blocks_dispatch() -> None:
    # Given: a reducer with one bounded MCP capability.


    reducer = _reducer(mcp_capabilities=frozenset({McpCapability.PRESENTATION_DECK}))

    blocked = McpDispatchProposal(
        McpCapability.KNOWLEDGE_LOOKUP,
        CommandId("mcp-1"),
        cancelled=False,
    )

    cancelled = McpDispatchProposal(
        McpCapability.KNOWLEDGE_LOOKUP,
        CommandId("mcp-2"),
        cancelled=True,
    )

    allowed = McpDispatchProposal(
        McpCapability.PRESENTATION_DECK,
        CommandId("mcp-3"),
        cancelled=False,
    )

    # When: dispatch is attempted with a missing capability or cancellation.

    unsupported = reducer.reduce_mcp(blocked)

    cancelled_result = reducer.reduce_mcp(cancelled)

    accepted = reducer.reduce_mcp(allowed)

    duplicate = reducer.reduce_mcp(allowed)

    # Then: no direct external call is admitted for either proposal.

    assert unsupported == McpDispatchRejected(
        McpDispatchRejection.UNSUPPORTED_CAPABILITY
    )

    assert cancelled_result == McpDispatchRejected(McpDispatchRejection.CANCELLED)

    assert accepted == McpDispatchAccepted(
        command_id=CommandId("mcp-3"), capability=McpCapability.PRESENTATION_DECK
    )

    assert duplicate == McpDispatchRejected(McpDispatchRejection.DUPLICATE)
    assert reducer.presentation_state is None


def _reducer(
    *, mcp_capabilities: frozenset[McpCapability] | None = None
) -> SessionInteractionReducer:

    return SessionInteractionReducer(
        scheduler=SessionScheduler(
            session_id=SessionId("session-1"),
            turn_id_prefix="turn",
        ),
        actions=ActionCapabilityRegistry(frozenset({"hello"})),
        mcp_capabilities=frozenset() if mcp_capabilities is None else mcp_capabilities,
    )


def _comment() -> CommentProposal:

    return CommentProposal(
        text="请解释量化",
        correlation=EventCorrelation(
            trace_id=TraceId("trace-1"),
            session_id=SessionId("session-1"),
            sequence=EventSequence(1),
        ),
    )
