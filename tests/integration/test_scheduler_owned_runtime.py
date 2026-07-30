"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import asyncio
import json
from dataclasses import replace
from typing import final

from orchestrator.asr_semantic_gate import AsrSemanticGate
from orchestrator.identity import (
    EncryptedVoiceTemplate,
    ProfileEnrollment,
    VoiceProfileId,
)
from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.interactions import (
    ActionProposal,
    CommandId,
    CommentProposal,
    McpCapability,
    McpDispatchProposal,
    PresentationCommand,
    PresentationCommandKind,
    PresentationResult,
)
from orchestrator.mcp_adapters import (
    McpAdapterResult,
    McpDispatchOutcome,
    McpIntent,
    ScopedMcpAdapterDispatcher,
)
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.runtime_contracts import RuntimeObservables
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence, StateRevision
from orchestrator.task_reducer import TaskEffect, TaskResult
from orchestrator.task_registry import (
    IdempotencyKey,
    SchedulerTaskConfig,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRequest,
    TaskState,
)


def test_runtime_dispatches_valid_proposal_and_rejects_replay_without_effect() -> None:
    # Given: one production-composed session runtime and its first comment proposal.

    """函数契约说明.

    功能: 验证 runtime dispatches valid
    proposal and rejects replay without
    effect 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )

    proposal = CommentProposal("解释量化", _correlation("session-1", "trace-1", 1))

    # When: the external proposal is accepted, then delivered a second time.

    accepted = runtime.receive_comment(proposal)

    baseline = runtime.observables

    duplicate = runtime.receive_comment(proposal)

    # Then: only the reducer-approved proposal opens a turn and dispatches once.

    assert accepted.accepted is True

    assert accepted.correlation == proposal.correlation

    assert runtime.scheduler.snapshot.revision == StateRevision(1)

    assert runtime.scheduler.snapshot.active_turn_id == TurnId("turn-0001")

    assert len(runtime.observables.dispatches) == 1

    assert duplicate.accepted is False

    assert runtime.scheduler.snapshot == baseline.snapshot

    assert runtime.observables.dispatches == baseline.dispatches

    assert runtime.observables.task_commits == baseline.task_commits == ()

    assert runtime.observables.generated_rtp == baseline.generated_rtp == ()

    assert runtime.observables.sound_transitions == baseline.sound_transitions == ()

    assert runtime.observables.rejections[-1].correlation == proposal.correlation


def test_runtime_rejects_invalid_task_results_without_effect() -> None:
    # Given: a registered interactive task for the runtime's current turn.

    """函数契约说明.

    功能: 验证 runtime rejects invalid task
    results without effect 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )

    first = runtime.receive_comment(
        CommentProposal("第一条", _correlation("session-1", "trace-1", 1))
    )

    assert first.turn_id is not None

    assert first.turn_id == TurnId("turn-0001")

    request = _request(runtime, first.turn_id)

    task_correlation = _correlation("session-1", "trace-task", 3)

    assert runtime.schedule_task(request, task_correlation).accepted is True

    _ = runtime.receive_comment(
        CommentProposal("第二条", _correlation("session-1", "trace-2", 2))
    )

    baseline = runtime.observables

    # When: stale, foreign-session, and inactive-turn completions reach the reducer.

    stale = runtime.reduce_task(_result(request), task_correlation)

    foreign = runtime.reduce_task(
        replace(_result(request), session_id=SessionId("foreign-session")),
        task_correlation,
    )

    inactive = runtime.reduce_task(
        replace(_result(request), turn_id=TurnId("inactive-turn")), task_correlation
    )

    # Then: each correlated refusal leaves all state and effects unchanged.

    assert (stale.accepted, foreign.accepted, inactive.accepted) == (
        False,
        False,
        False,
    )

    assert runtime.scheduler.snapshot == baseline.snapshot

    assert runtime.observables.dispatches == baseline.dispatches

    assert runtime.observables.task_commits == baseline.task_commits == ()

    assert runtime.observables.generated_rtp == baseline.generated_rtp == ()

    assert runtime.observables.sound_transitions == baseline.sound_transitions == ()

    assert len(runtime.observables.rejections) == len(baseline.rejections) + 3


def test_runtime_composes_the_adaptive_agent_policy() -> None:
    # Given: one production-composed session runtime.

    """函数契约说明.

    功能: 验证 runtime composes the adaptive
    agent policy 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    # When: the runtime is created without a product-mode selector.

    runtime = SessionRuntime.create(
        session_id=SessionId("session-adaptive"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )

    # Then: the runtime owns the single adaptive policy.

    assert isinstance(runtime.mode_policy, AdaptiveAgentPolicy)


def test_runtime_opens_one_turn_only_for_semantically_accepted_asr_final() -> None:
    # Given: final ASR events and deterministic accept and discard gate providers.

    """函数契约说明.

    功能: 验证 runtime opens one turn only
    for semantically accepted asr final
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 3),
    )

    event = ASRAudienceEvent("请介绍 BitNet", 20, "asr-1", 1)

    accepted_gate = AsrSemanticGate(lambda request: '{"decision":"accept"}')

    discarded_gate = AsrSemanticGate(lambda request: "malformed")

    # When: the accepted final is followed by a malformed semantic decision.

    accepted = runtime.receive_asr_final(
        event, _correlation("session-1", "trace-asr-1", 1), accepted_gate
    )

    baseline = runtime.observables

    discarded = runtime.receive_asr_final(
        event, _correlation("session-1", "trace-asr-2", 2), discarded_gate
    )

    # Then: only the accepted gate result opens a monotonic scheduler turn.

    assert accepted.accepted is True

    assert accepted.turn_id == TurnId("turn-0001")

    assert discarded.accepted is False

    assert runtime.observables.snapshot == baseline.snapshot

    assert runtime.observables.dispatches == baseline.dispatches

    assert runtime.observables.task_commits == baseline.task_commits == ()

    assert runtime.observables.generated_rtp == baseline.generated_rtp == ()

    assert runtime.observables.sound_transitions == baseline.sound_transitions == ()

    assert runtime.observables.rejections[-1].correlation == _correlation(
        "session-1", "trace-asr-2", 2
    )


def test_runtime_rejects_stale_task_before_lane_enqueue() -> None:
    # Given: a live turn and a task request captured before that turn revision.

    """函数契约说明.

    功能: 验证 runtime rejects stale task
    before lane enqueue 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = _runtime()

    accepted = _open_turn(runtime)

    request = replace(_request(runtime, accepted), snapshot_revision=StateRevision(0))

    correlation = _correlation("session-1", "trace-stale", 2)

    baseline = runtime.observables

    # When: a stale request is scheduled through the production runtime.

    outcome = runtime.schedule_task(request, correlation)

    # Then: runtime admission rejects it before lane ownership or effects change.

    assert outcome.accepted is False

    assert runtime.next_task(now_ms=0) is None

    _assert_no_effect(runtime, baseline, correlation)


def test_runtime_rejects_overdue_and_cancelled_task_completions() -> None:
    # Given: a fake monotonic clock, one overdue task, and one cancelled task.

    """函数契约说明.

    功能: 验证 runtime rejects overdue and
    cancelled task completions
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    clock = _Clock(now_ms=201)

    runtime = _runtime(clock)

    turn_id = _open_turn(runtime)

    overdue = _request(runtime, turn_id, task_id="overdue", deadline_ms=200)

    cancelled = _request(runtime, turn_id, task_id="cancelled", deadline_ms=300)

    correlation = _correlation("session-1", "trace-task", 2)

    assert runtime.schedule_task(overdue, correlation).accepted is True

    assert runtime.schedule_task(cancelled, correlation).accepted is True

    assert (
        runtime.task_registry.cancel(cancelled.task_id, reason="interrupted")
        is not None
    )

    baseline = runtime.observables

    # When: both terminal completions reach the reducer.

    overdue_outcome = runtime.reduce_task(_result(overdue), correlation)

    cancelled_outcome = runtime.reduce_task(_result(cancelled), correlation)

    # Then: current monotonic time and cancellation prevent every effect.

    assert overdue_outcome.accepted is False

    assert cancelled_outcome.accepted is False

    _assert_no_effect(runtime, baseline, correlation, rejection_count=2)


def test_runtime_skips_cancelled_task_queued_before_worker_selection() -> None:
    # Given: a scheduler-admitted task that is cancelled while still queued.

    """函数契约说明.

    功能: 验证 runtime skips cancelled task
    queued before worker selection
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = _runtime()

    turn_id = _open_turn(runtime)

    request = _request(runtime, turn_id, task_id="cancelled-queued")

    correlation = _correlation("session-1", "trace-queued", 2)

    assert runtime.schedule_task(request, correlation).accepted is True

    assert (
        runtime.task_registry.cancel(request.task_id, reason="newer_turn") is not None
    )

    baseline = runtime.observables

    # When: the worker asks for its next lane item.

    selected = runtime.next_task(now_ms=0)

    # Then: terminal queued work is never handed to a worker or dispatched.

    assert selected is None

    assert runtime.observables == baseline


def test_profile_revoke_cancels_pending_task_before_worker_selection() -> None:
    # Given: scheduler-admitted pending work that may depend on profile context.

    """函数契约说明.

    功能: 验证 profile revoke cancels
    pending task before worker selection
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = _runtime()

    turn_id = _open_turn(runtime)

    request = _request(runtime, turn_id, task_id="profile-dependent")

    correlation = _correlation("session-1", "trace-profile", 2)

    assert runtime.schedule_task(request, correlation).accepted is True

    # When: consent is revoked through the runtime-owned session data authority.

    runtime.interaction_ingress.data.revoke_profile_consent(VoiceProfileId("profile-1"))

    # Then: pending work is terminally cancelled before a worker can select it.

    assert runtime.task_registry.task(request.task_id) is not None

    assert runtime.next_task(now_ms=0) is None


def test_runtime_records_governed_interaction_outcomes_without_sensitive_values() -> (
    None
):
    # Given: one live scheduler session, a consented profile, and a pending task.

    """函数契约说明.

    功能: 验证 runtime records governed
    interaction outcomes without
    sensitive values 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = _runtime()

    correlation = _correlation("session-1", "trace-sensitive", 2)

    profile_id = VoiceProfileId("profile-sensitive")

    _ = runtime.enroll_profile(
        ProfileEnrollment(
            profile_id,
            "private-name",
            EncryptedVoiceTemplate(b"voice-template-sensitive"),
            consented=True,
        ),
        correlation,
    )

    turn_id = _open_turn(runtime)

    mcp_task = _request(runtime, turn_id, task_id="mcp-sensitive")

    # When: lifecycle, reducer acknowledgements, and cancelled MCP work arrive.

    revoked = runtime.revoke_profile_consent(profile_id, correlation)

    action = runtime.receive_action(
        ActionProposal("speak", CommandId("action-1")), correlation
    )

    command = PresentationCommand(
        PresentationCommandKind.LOAD,
        "deck-1",
        1,
        CommandId("load-1"),
    )

    proposed = runtime.receive_presentation(command, correlation)

    failed_ack = runtime.receive_control(
        json.dumps(
            {
                "event_type": "presentation.result",
                "source": "frontend",
                "trace_id": "trace-sensitive",
                "session_id": "session-1",
                "seq": 3,
                "data": {"command_id": str(command.command_id), "succeeded": False},
            }
        )
    )

    scheduled_mcp = runtime.schedule_mcp_task(
        McpIntent(
            McpDispatchProposal(
                McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
            ),
            command,
            deadline_ms=100,
        ),
        mcp_task,
        correlation,
    )

    cancelled_mcp = runtime.cancel_task(mcp_task.task_id, correlation)

    late_mcp_result = runtime.reduce_task(_result(mcp_task), correlation)

    # Then: only approved reducers alter state and the operational record is redacted.

    assert (revoked.accepted, action.accepted, proposed.accepted) == (True, True, True)

    assert (failed_ack, scheduled_mcp.accepted, cancelled_mcp.accepted) == (
        True,
        True,
        True,
    )

    assert late_mcp_result.accepted is False

    assert runtime.interaction_ingress.reducer.presentation_state is None

    journal = runtime.operational_journal.records

    assert {record.stage for record in journal} >= {
        "profile_revoked",
        "action",
        "presentation_command",
        "presentation_ack",
        "mcp_task",
        "task_cancelled",
        "task_result",
    }

    assert "sensitive" not in repr(journal)

    assert "private-name" not in repr(journal)

    assert "voice-template-sensitive" not in repr(journal)


def test_live_mcp_worker_requires_exact_correlated_ack_before_deck_commit() -> None:
    # Given: a reducer-admitted deck command and a live successful adapter.

    """函数契约说明.

    功能: 验证 live mcp worker requires
    exact correlated ack before deck
    commit 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = _runtime()

    turn_id = _open_turn(runtime)

    correlation = _correlation("session-1", "trace-deck", 2)

    command = PresentationCommand(
        PresentationCommandKind.LOAD, "deck-1", 1, CommandId("deck-load")
    )

    assert runtime.receive_presentation(command, correlation).accepted is True

    adapter = _SuccessfulAdapter()

    runtime.mcp_dispatcher = ScopedMcpAdapterDispatcher(
        runtime.interaction_ingress.reducer, {McpCapability.PRESENTATION_DECK: adapter}
    )

    intent = McpIntent(
        McpDispatchProposal(
            McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
        ),
        command,
        deadline_ms=100,
    )

    scheduled = runtime.schedule_mcp_task(
        intent,
        _request(runtime, turn_id),
        correlation,
    )

    assert scheduled.accepted

    # When: the worker succeeds, then forged and exact acknowledgements arrive.

    outcome = runtime.run_mcp_worker(now_ms=0, correlation=correlation)

    assert outcome.completion is not None

    forged = runtime.receive_presentation_result(
        outcome.completion,
        _correlation("session-1", "trace-deck", 3),
    )

    acknowledged = runtime.receive_presentation_result(
        outcome.completion,
        _correlation("session-1", "trace-deck", 2),
    )

    # Then: adapter completion and forged acknowledgement cannot mutate deck state.

    assert outcome.accepted is True

    assert adapter.calls == [command.command_id]

    assert forged.accepted is False

    assert acknowledged.accepted is True

    assert runtime.interaction_ingress.reducer.presentation_state == ("deck-1", "v1", 1)


def test_live_mcp_worker_reconciles_ambiguity_without_adapter_state_commit() -> None:
    # Given: an ambiguous adapter result followed by its explicit reconciliation.

    """函数契约说明.

    功能: 验证 live mcp worker reconciles
    ambiguity without adapter state
    commit 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime = _runtime()

    turn_id = _open_turn(runtime)

    correlation = _correlation("session-1", "trace-ambiguous", 2)

    command = PresentationCommand(
        PresentationCommandKind.LOAD, "deck-ambiguous", 1, CommandId("deck-ambiguous")
    )

    assert runtime.receive_presentation(command, correlation).accepted

    adapter = _ResultAdapter(McpAdapterResult.ambiguous(), McpAdapterResult.succeeded())

    runtime.mcp_dispatcher = ScopedMcpAdapterDispatcher(
        runtime.interaction_ingress.reducer, {McpCapability.PRESENTATION_DECK: adapter}
    )

    request = _request(runtime, turn_id, task_id="ambiguous")

    intent = McpIntent(
        McpDispatchProposal(
            McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
        ),
        command,
        deadline_ms=100,
    )

    assert runtime.schedule_mcp_task(intent, request, correlation).accepted

    # When: the worker observes ambiguity, then reconciliation succeeds.

    ambiguous = runtime.run_mcp_worker(now_ms=0, correlation=correlation)

    reconciled = runtime.reconcile_mcp_worker(
        request.task_id, now_ms=1, correlation=correlation
    )

    # Then: neither adapter path commits deck state before the canonical ACK.

    assert ambiguous.accepted is False

    assert reconciled.accepted is True

    assert adapter.calls == [command.command_id, command.command_id]

    assert runtime.interaction_ingress.reducer.presentation_state is None

    assert reconciled.completion is not None

    assert runtime.receive_presentation_result(
        reconciled.completion, correlation
    ).accepted

    assert runtime.interaction_ingress.reducer.presentation_state == (
        "deck-ambiguous",
        "v1",
        1,
    )


class _SuccessfulAdapter:
    """类契约说明.

    职责: 定义 _SuccessfulAdapter
    的状态、行为和对外协作边界。
    契约: 方法: __init__、execute、reconcile。
    """

    def __init__(self) -> None:
        """函数契约说明.

        功能: 初始化 _SuccessfulAdapter
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        self.calls: list[CommandId] = []

    def execute(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 execute 的同步逻辑,并协调 append,
        succeeded。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """

        self.calls.append(intent.command.command_id)

        return McpAdapterResult.succeeded()

    def reconcile(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 reconcile 的同步逻辑,并协调
        execute。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """

        return self.execute(intent)


@final
class _ResultAdapter:
    """类契约说明.

    职责: 定义 _ResultAdapter 的状态、行为和对外协作边界。
    契约: 方法: __init__、execute、reconcile。
    """

    def __init__(self, *results: McpAdapterResult) -> None:
        """函数契约说明.

        功能: 初始化 _ResultAdapter
        的字段并建立实例不变式。
        参数: self 表示当前实例。 *results:
        McpAdapterResult。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self._results = results

        self.calls: list[CommandId] = []

    def execute(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 execute 的同步逻辑,并协调 append,
        len。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """

        self.calls.append(intent.command.command_id)

        return self._results[len(self.calls) - 1]

    def reconcile(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 reconcile 的同步逻辑,并协调
        execute。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """

        return self.execute(intent)


def test_mcp_cancel_before_worker_removes_retained_intent_without_ack_commit() -> None:
    # Given: scheduled deck work that has not reached the worker.

    """函数契约说明.

    功能: 验证 mcp cancel before worker
    removes retained intent without ack
    commit 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime, request, _, correlation, adapter = _scheduled_deck("cancel")

    # When: scheduler cancellation occurs before worker selection.

    cancelled = runtime.cancel_task(request.task_id, correlation)

    outcome = runtime.run_mcp_worker(now_ms=0, correlation=correlation)

    # Then: no adapter or ACK path can commit the deck state.

    assert cancelled.accepted

    assert not outcome.accepted

    assert adapter.calls == []

    assert runtime.interaction_ingress.reducer.presentation_state is None


def test_mcp_deadline_before_worker_times_out_without_adapter_call() -> None:
    # Given: deck work whose scheduler deadline is already elapsed.

    """函数契约说明.

    功能: 验证 mcp deadline before worker
    times out without adapter call
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime, request, _, correlation, adapter = _scheduled_deck("before", deadline_ms=0)

    # When: the worker selects pending work after its deadline.

    outcome = runtime.run_mcp_worker(now_ms=1, correlation=correlation)

    # Then: the task times out without adapter work or acknowledged deck state.

    record = runtime.task_registry.task(request.task_id)

    assert not outcome.accepted

    assert record is not None

    assert record.state is TaskState.TIMED_OUT

    assert adapter.calls == []

    assert runtime.interaction_ingress.reducer.presentation_state is None


def test_ambiguous_reconcile_after_deadline_times_out_without_commit() -> None:
    # Given: a live ambiguous adapter invocation awaiting reconciliation.

    """函数契约说明.

    功能: 验证 ambiguous reconcile after
    deadline times out without commit
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime, request, _, correlation, adapter = _scheduled_deck(
        "ambiguous-timeout",
        adapter_results=(McpAdapterResult.ambiguous(),),
        deadline_ms=1,
    )

    assert not runtime.run_mcp_worker(now_ms=0, correlation=correlation).accepted

    # When: reconciliation arrives after the retained intent deadline.

    outcome = runtime.reconcile_mcp_worker(
        request.task_id, now_ms=2, correlation=correlation
    )

    # Then: scheduler terminalizes work without a retry or deck commit.

    record = runtime.task_registry.task(request.task_id)

    assert not outcome.accepted

    assert record is not None

    assert record.state is TaskState.TIMED_OUT

    assert adapter.calls == [CommandId("deck-ambiguous-timeout")]

    assert runtime.interaction_ingress.reducer.presentation_state is None


def test_late_exact_ack_after_cancellation_is_rejected_without_deck_state() -> None:
    # Given: a scheduled deck command cancelled before its adapter executes.

    """函数契约说明.

    功能: 验证 late exact ack after
    cancellation is rejected without
    deck state 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    runtime, request, command, correlation, _ = _scheduled_deck("late-ack")

    assert runtime.cancel_task(request.task_id, correlation).accepted

    # When: a syntactically exact but late success acknowledgement arrives.

    outcome = runtime.receive_presentation_result(
        PresentationResult(command.command_id, succeeded=True), correlation
    )

    # Then: cancelled work cannot be resurrected into acknowledged deck state.

    assert not outcome.accepted

    assert runtime.interaction_ingress.reducer.presentation_state is None


def test_active_local_mcp_cancellation_rejects_late_completion_without_ack_commit() -> (
    None
):
    # Given: a root-composed local deck adapter and one admitted MCP task.

    """函数契约说明.

    功能: 验证 active local mcp cancellation
    rejects late completion without ack
    commit 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    runtime = _runtime()

    turn_id = _open_turn(runtime)

    correlation = _correlation("session-1", "trace-active", 2)

    command = PresentationCommand(
        PresentationCommandKind.LOAD, "deck-active", 1, CommandId("deck-active")
    )

    assert runtime.receive_presentation(command, correlation).accepted

    request = _request(runtime, turn_id, task_id="active")

    intent = McpIntent(
        McpDispatchProposal(
            McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
        ),
        command,
        100,
    )

    assert runtime.schedule_mcp_task(intent, request, correlation).accepted

    # When: task cancellation wins during the adapter's async execution yield point.

    async def run() -> McpDispatchOutcome:
        """函数契约说明.

        功能: 运行流程并协调其依赖步骤。
        参数: 无显式业务参数。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `McpDispatchOutcome`。
        """

        worker = asyncio.create_task(
            runtime.run_mcp_worker_async(now_ms=0, correlation=correlation)
        )

        await asyncio.sleep(0)

        assert runtime.cancel_task(request.task_id, correlation).accepted

        return await worker

    outcome = asyncio.run(run())

    # Then: the late local result and an exact ACK cannot commit presentation state.

    assert outcome.accepted is False

    assert not runtime.receive_presentation_result(
        PresentationResult(command.command_id, succeeded=True), correlation
    ).accepted

    assert runtime.interaction_ingress.reducer.presentation_state is None


def _scheduled_deck(
    suffix: str,
    *,
    adapter_results: tuple[McpAdapterResult, ...] | None = None,
    deadline_ms: int = 100,
) -> tuple[
    SessionRuntime, TaskRequest, PresentationCommand, EventCorrelation, _ResultAdapter
]:
    """函数契约说明.

    功能: 执行 _scheduled_deck 的同步逻辑,并协调
    _runtime, _open_turn, _correlation,
    PresentationCommand。
    参数: suffix: str。 必填。
    adapter_results:
    tuple[McpAdapterResult, ...] | None。
    可省略。 deadline_ms: int。 可省略。
    契约: 同步调用。 返回 `tuple[SessionRuntime,
    TaskRequest, PresentationCommand,
    EventCorrelation, _ResultAdapter]`。
    """

    runtime = _runtime()

    turn_id = _open_turn(runtime)

    correlation = _correlation("session-1", f"trace-{suffix}", 2)

    command = PresentationCommand(
        PresentationCommandKind.LOAD, f"deck-{suffix}", 1, CommandId(f"deck-{suffix}")
    )

    assert runtime.receive_presentation(command, correlation).accepted

    results = (
        (McpAdapterResult.succeeded(),) if adapter_results is None else adapter_results
    )

    adapter = _ResultAdapter(*results)

    runtime.mcp_dispatcher = ScopedMcpAdapterDispatcher(
        runtime.interaction_ingress.reducer, {McpCapability.PRESENTATION_DECK: adapter}
    )

    request = _request(runtime, turn_id, task_id=suffix, deadline_ms=deadline_ms)

    intent = McpIntent(
        McpDispatchProposal(
            McpCapability.PRESENTATION_DECK, command.command_id, cancelled=False
        ),
        command,
        deadline_ms=deadline_ms,
    )

    assert runtime.schedule_mcp_task(intent, request, correlation).accepted

    return runtime, request, command, correlation, adapter


def _correlation(session_id: str, trace_id: str, sequence: int) -> EventCorrelation:
    """函数契约说明.

    功能: 执行 _correlation 的同步逻辑,并协调
    EventCorrelation, TraceId,
    SessionId, EventSequence。
    参数: session_id: str。 必填。 trace_id:
    str。 必填。 sequence: int。 必填。
    契约: 同步调用。 返回 `EventCorrelation`。
    """

    return EventCorrelation(
        TraceId(trace_id),
        SessionId(session_id),
        EventSequence(sequence),
    )


def _runtime(clock: "_Clock | None" = None) -> SessionRuntime:
    """函数契约说明.

    功能: 执行 _runtime 的同步逻辑,并协调 create,
    SessionId, SchedulerTaskConfig,
    frozenset。
    参数: clock: '_Clock | None'。 可省略。
    契约: 同步调用。 返回 `SessionRuntime`。
    """

    return SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 3),
        clock=_monotonic_clock if clock is None else clock.monotonic_ms,
    )


def _open_turn(runtime: SessionRuntime) -> TurnId:
    """函数契约说明.

    功能: 执行 _open_turn 的同步逻辑,并协调
    receive_comment, CommentProposal,
    _correlation。
    参数: runtime: SessionRuntime。 必填。
    契约: 同步调用。 返回 `TurnId`。
    """

    outcome = runtime.receive_comment(
        CommentProposal("开始", _correlation("session-1", "trace-turn", 1))
    )

    assert outcome.turn_id is not None

    return outcome.turn_id


def _request(
    runtime: SessionRuntime,
    turn_id: TurnId,
    *,
    task_id: str = "task-1",
    deadline_ms: int = 200,
) -> TaskRequest:
    """函数契约说明.

    功能: 执行 _request 的同步逻辑,并协调
    TaskRequest, TaskId, TaskDeadlineMs,
    IdempotencyKey。
    参数: runtime: SessionRuntime。 必填。
    turn_id: TurnId。 必填。 task_id: str。
    可省略。 deadline_ms: int。 可省略。
    契约: 同步调用。 返回 `TaskRequest`。
    """

    return TaskRequest(
        task_id=TaskId(task_id),
        session_id=runtime.scheduler.snapshot.session_id,
        turn_id=turn_id,
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(deadline_ms),
        snapshot_revision=runtime.scheduler.snapshot.revision,
        idempotency_key=IdempotencyKey(f"answer-{task_id}"),
        kind=TaskKind.INTERACTIVE,
    )


def _result(request: TaskRequest) -> TaskResult:
    """函数契约说明.

    功能: 执行 _result 的同步逻辑,并协调 TaskResult,
    TaskEffect。
    参数: request: TaskRequest。 必填。
    契约: 同步调用。 返回 `TaskResult`。
    """

    return TaskResult(
        task_id=request.task_id,
        session_id=request.session_id,
        turn_id=request.turn_id,
        snapshot_revision=request.snapshot_revision,
        effect=TaskEffect("answer", "accepted"),
    )


class _Clock:
    """类契约说明.

    职责: 定义 _Clock 的状态、行为和对外协作边界。
    契约: 字段: now_ms。 方法:
    __init__、monotonic_ms。
    """

    now_ms: int

    def __init__(self, *, now_ms: int) -> None:
        """函数契约说明.

        功能: 初始化 _Clock 的字段并建立实例不变式。
        参数: self 表示当前实例。 now_ms: int。
        必填。
        契约: 同步调用。 返回 `None`。
        """

        self.now_ms = now_ms

    def monotonic_ms(self) -> int:
        """函数契约说明.

        功能: 执行 monotonic_ms
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `int`。
        """

        return self.now_ms


def _assert_no_effect(
    runtime: SessionRuntime,
    baseline: RuntimeObservables,
    correlation: EventCorrelation,
    *,
    rejection_count: int = 1,
) -> None:
    """函数契约说明.

    功能: 执行 _assert_no_effect
    的同步逻辑,并维持签名契约。
    参数: runtime: SessionRuntime。 必填。
    baseline: RuntimeObservables。 必填。
    correlation: EventCorrelation。 必填。
    rejection_count: int。 可省略。
    契约: 同步调用。 返回 `None`。
    """

    assert runtime.observables.snapshot == baseline.snapshot

    assert runtime.observables.dispatches == baseline.dispatches

    assert runtime.observables.task_commits == baseline.task_commits

    assert runtime.observables.generated_rtp == baseline.generated_rtp

    assert runtime.observables.sound_transitions == baseline.sound_transitions

    assert runtime.observables.rejections[-rejection_count].correlation == correlation


def _monotonic_clock() -> int:
    """函数契约说明.

    功能: 执行 _monotonic_clock
    的同步逻辑,并维持签名契约。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `int`。
    """

    return 0
