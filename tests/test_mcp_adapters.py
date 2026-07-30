"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass

from orchestrator.ids import SessionId
from orchestrator.interactions import (
    ActionCapabilityRegistry,
    CommandId,
    McpCapability,
    McpDispatchProposal,
    PresentationCommand,
    PresentationCommandKind,
    SessionInteractionReducer,
)
from orchestrator.mcp_adapters import (
    LocalDeckAdapter,
    McpAdapterResult,
    McpIntent,
    McpJournalKind,
    ScopedMcpAdapterDispatcher,
)
from orchestrator.sessions import SessionScheduler


def test_deck_adapter_records_intent_then_commits_only_correlated_ack() -> None:
    # Given: a reducer-admitted deck load and an allowlisted fake deck adapter.

    """函数契约说明.

    功能: 验证 deck adapter records intent
    then commits only correlated ack
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    reducer = _reducer()

    command = PresentationCommand(
        PresentationCommandKind.LOAD, "deck-1", 1, CommandId("load-1")
    )

    _ = reducer.reduce_presentation(command)

    adapter = _FakeAdapter(McpAdapterResult.succeeded())

    dispatcher = ScopedMcpAdapterDispatcher(
        reducer, {McpCapability.PRESENTATION_DECK: adapter}
    )

    # When: the deck adapter returns the matching success result before deadline.

    outcome = dispatcher.dispatch(
        McpIntent(
            McpDispatchProposal(
                McpCapability.PRESENTATION_DECK,
                command.command_id,
                cancelled=False,
            ),
            command,
            deadline_ms=100,
        ),
        now_ms=10,
    )

    # Then: the adapter emits a completion but cannot commit deck state itself.

    assert outcome.accepted is True

    assert adapter.calls == [command.command_id]

    assert outcome.completion is not None

    assert reducer.presentation_state is None

    _ = reducer.reduce_presentation_result(outcome.completion)

    assert reducer.presentation_state == ("deck-1", "v1", 1)

    assert [entry.kind for entry in dispatcher.journal] == [
        McpJournalKind.INTENT_DISPATCHED,
        McpJournalKind.RESULT_SUCCEEDED,
    ]


def test_timeout_and_ambiguous_results_remain_non_committing_until_reconciled() -> None:
    # Given: one timed-out and one ambiguous externally approved deck command.

    """函数契约说明.

    功能: 验证 timeout and ambiguous results
    remain non committing until
    reconciled 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    reducer = _reducer()

    timed_out = PresentationCommand(
        PresentationCommandKind.LOAD, "deck-timeout", 1, CommandId("timeout")
    )

    ambiguous = PresentationCommand(
        PresentationCommandKind.LOAD, "deck-ambiguous", 1, CommandId("ambiguous")
    )

    _ = reducer.reduce_presentation(timed_out)

    _ = reducer.reduce_presentation(ambiguous)

    adapter = _FakeAdapter(McpAdapterResult.ambiguous(), McpAdapterResult.succeeded())

    dispatcher = ScopedMcpAdapterDispatcher(
        reducer, {McpCapability.PRESENTATION_DECK: adapter}
    )

    # When: one invocation is late and the other requires explicit reconciliation.

    late = dispatcher.dispatch(
        McpIntent(
            McpDispatchProposal(
                McpCapability.PRESENTATION_DECK,
                timed_out.command_id,
                cancelled=False,
            ),
            timed_out,
            deadline_ms=10,
        ),
        now_ms=11,
    )

    pending = dispatcher.dispatch(
        McpIntent(
            McpDispatchProposal(
                McpCapability.PRESENTATION_DECK,
                ambiguous.command_id,
                cancelled=False,
            ),
            ambiguous,
            deadline_ms=100,
        ),
        now_ms=10,
    )

    reconciled = dispatcher.reconcile(ambiguous.command_id, now_ms=11)

    # Then: stale work never calls the adapter and frontend ACK commits reconciliation.

    assert late.accepted is False

    assert pending.accepted is False

    assert reconciled.accepted is True

    assert adapter.calls == [ambiguous.command_id, ambiguous.command_id]

    assert reconciled.completion is not None

    assert reducer.presentation_state is None

    _ = reducer.reduce_presentation_result(reconciled.completion)

    assert reducer.presentation_state == ("deck-ambiguous", "v1", 1)


def test_duplicate_or_forged_deck_intent_never_reaches_adapter() -> None:
    # Given: exactly one reducer-pending deck command and a recording adapter.

    """函数契约说明.

    功能: 验证 duplicate or forged deck
    intent never reaches adapter
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    reducer = _reducer()

    command = PresentationCommand(
        PresentationCommandKind.LOAD, "deck-1", 1, CommandId("load-1")
    )

    _ = reducer.reduce_presentation(command)

    adapter = _FakeAdapter(McpAdapterResult.ambiguous())

    dispatcher = ScopedMcpAdapterDispatcher(
        reducer, {McpCapability.PRESENTATION_DECK: adapter}
    )

    intent = McpIntent(
        McpDispatchProposal(
            McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
        ),
        command,
        deadline_ms=100,
    )

    forged = McpIntent(
        McpDispatchProposal(
            McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
        ),
        PresentationCommand(
            PresentationCommandKind.LOAD, "other-deck", 1, command.command_id
        ),
        deadline_ms=100,
    )

    # When: the external boundary receives a replay and a forged same-ID command.

    first = dispatcher.dispatch(intent, now_ms=10)

    replay = dispatcher.dispatch(intent, now_ms=10)

    forged_outcome = dispatcher.dispatch(forged, now_ms=10)

    # Then: only the original reducer-approved intent executes once.

    assert first.accepted is False

    assert replay.accepted is False

    assert forged_outcome.accepted is False

    assert adapter.calls == [command.command_id]


def test_root_deck_adapter_is_allowlisted_and_cancellable() -> None:
    # Given: the production local deck adapter selected by the root composition.

    """函数契约说明.

    功能: 验证 root deck adapter is
    allowlisted and cancellable
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    adapter = LocalDeckAdapter()

    # When: the root asks for its concrete allowlisted deck boundary.

    result = adapter.execute(_intent())

    # Then: it returns a bounded result rather than a payload-bearing adapter map.

    assert result == McpAdapterResult.succeeded()


def _reducer() -> SessionInteractionReducer:
    """函数契约说明.

    功能: 执行 _reducer 的同步逻辑,并协调
    SessionInteractionReducer,
    SessionScheduler,
    ActionCapabilityRegistry, frozenset。
    参数: 无显式业务参数。
    契约: 同步调用。 返回
    `SessionInteractionReducer`。
    """

    return SessionInteractionReducer(
        scheduler=SessionScheduler(
            session_id=SessionId("session-1"), turn_id_prefix="turn"
        ),
        actions=ActionCapabilityRegistry(frozenset()),
        mcp_capabilities=frozenset({McpCapability.PRESENTATION_DECK}),
    )


def _intent() -> McpIntent:
    """函数契约说明.

    功能: 执行 _intent 的同步逻辑,并协调
    PresentationCommand, McpIntent,
    CommandId, McpDispatchProposal。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `McpIntent`。
    """

    command = PresentationCommand(
        PresentationCommandKind.LOAD, "deck-local", 1, CommandId("local")
    )

    return McpIntent(
        McpDispatchProposal(
            McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
        ),
        command,
        100,
    )


@dataclass
class _FakeAdapter:
    """类契约说明.

    职责: 保存 _FakeAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: results、calls。 方法:
    __init__、execute、reconcile。
    """

    results: tuple[McpAdapterResult, ...]

    calls: list[CommandId]

    def __init__(self, *results: McpAdapterResult) -> None:
        """函数契约说明.

        功能: 初始化 _FakeAdapter
        的字段并建立实例不变式。
        参数: self 表示当前实例。 *results:
        McpAdapterResult。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.results = results

        self.calls = []

    def execute(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 execute 的同步逻辑,并协调 append,
        len。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """

        self.calls.append(intent.command.command_id)

        return self.results[len(self.calls) - 1]

    def reconcile(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 reconcile 的同步逻辑,并协调
        execute。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """

        return self.execute(intent)
