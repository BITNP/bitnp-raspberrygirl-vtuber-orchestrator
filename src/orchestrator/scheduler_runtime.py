"""模块契约说明.

职责: 提供 orchestrator.scheduler_runtime
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic_ns

from orchestrator.asr_semantic_gate import AsrGateDecision, AsrSemanticGate
from orchestrator.control_ingress import (
    ActionControl,
    PresentationControl,
    PresentationResultControl,
    ProfileEnrollmentControl,
    ProfileRevocationControl,
    SessionControl,
)
from orchestrator.identity import ProfileEnrollment, VoiceProfileId
from orchestrator.ids import SessionId, TurnId
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.interactions import (
    ActionProposal,
    CommandId,
    CommentProposal,
    InteractionAccepted,
    McpCapability,
    PresentationCommand,
    PresentationResult,
)
from orchestrator.mcp_adapters import (
    LocalDeckAdapter,
    McpDispatchOutcome,
    McpIntent,
    ScopedMcpAdapterDispatcher,
)
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.operational_journal import OperationalJournal, OperationalRecord
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.runtime_contracts import (
    RuntimeDispatch,
    RuntimeObservables,
    RuntimeOutcome,
    RuntimeRejection,
)
from orchestrator.runtime_control_parsing import parse_presentation_result_control
from orchestrator.runtime_mcp_planning import (
    PresentationMcpPlanInput,
    build_presentation_mcp_plan,
)
from orchestrator.runtime_records import interaction_record, task_result_record
from orchestrator.scheduler_reflex import SchedulerOutputFence
from orchestrator.sessions import (
    EventCorrelation,
    SchedulerEvent,
    SessionScheduler,
    StartTurn,
    TransitionAccepted,
)
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.streaming_contracts import CancellationEpoch
from orchestrator.task_admission import scheduling_rejection, with_current_data_snapshot
from orchestrator.task_executor import TaskLaneExecutor
from orchestrator.task_reducer import (
    TaskEffect,
    TaskResult,
    TaskResultAccepted,
    TaskResultReducer,
)
from orchestrator.task_registry import (
    SchedulerTaskConfig,
    TaskId,
    TaskRegistrationAccepted,
    TaskRegistrationRejected,
    TaskRegistrationRejection,
    TaskRegistrationResult,
    TaskRegistry,
    TaskRequest,
    TaskState,
)


def _monotonic_ms() -> int:
    """函数契约说明.

    功能: 执行 _monotonic_ms 的同步逻辑,并协调
    monotonic_ns。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `int`。
    """
    return monotonic_ns() // 1_000_000


@dataclass(slots=True)
class _RuntimeJournal:
    dispatches: list[RuntimeDispatch] = field(default_factory=list)
    task_commits: list[TaskResult] = field(default_factory=list)
    rejections: list[RuntimeRejection] = field(default_factory=list)


@dataclass(slots=True)
class SessionRuntime:
    """类契约说明.

    职责: 保存 SessionRuntime
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: scheduler、tasks、executor、out
    put_fence、interaction_ingress、mcp_di
    spatcher。 方法: create、observables、rec
    eive_comment、receive_control、receive
    _session_control、receive_session_con
    trol_async。
    """

    scheduler: SessionScheduler

    task_registry: TaskRegistry

    task_reducer: TaskResultReducer

    executor: TaskLaneExecutor

    output_fence: SchedulerOutputFence

    interaction_ingress: SessionInteractionIngress

    mcp_dispatcher: ScopedMcpAdapterDispatcher

    mode_policy: AdaptiveAgentPolicy

    clock: Callable[[], int] = _monotonic_ms

    cancellation_epoch: CancellationEpoch = field(
        default_factory=lambda: CancellationEpoch(0)
    )

    _correlations: set[EventCorrelation] = field(default_factory=set)

    _mcp_intents: dict[TaskId, McpIntent] = field(default_factory=dict)

    _active_mcp_tasks: dict[TaskId, CommandId] = field(default_factory=dict)

    _presentation_correlations: dict[CommandId, EventCorrelation] = field(
        default_factory=dict
    )

    _journal: _RuntimeJournal = field(default_factory=_RuntimeJournal)

    operational_journal: OperationalJournal = field(default_factory=OperationalJournal)

    @classmethod
    def create(
        cls,
        *,
        session_id: SessionId,
        turn_id_prefix: str,
        task_config: SchedulerTaskConfig,
        clock: Callable[[], int] = _monotonic_ms,
    ) -> "SessionRuntime":
        """函数契约说明.

        功能: 执行 create 的同步逻辑,并协调
        SessionScheduler, create, cls,
        cancel_pending。
        参数: cls 表示当前类。 session_id:
        SessionId。 必填。 turn_id_prefix:
        str。 必填。 task_config:
        SchedulerTaskConfig。 必填。 clock:
        Callable[[], int]。 可省略。
        契约: 同步调用。 返回 `'SessionRuntime'`。
        """
        scheduler = SessionScheduler(
            session_id=session_id,
            turn_id_prefix=turn_id_prefix,
        )

        interaction_ingress = SessionInteractionIngress.create(scheduler)

        task_registry = TaskRegistry(
            session_id=session_id,
            config=task_config,
        )

        task_reducer = TaskResultReducer(task_registry)

        def invalidate_pending(reason: str) -> None:
            """函数契约说明.

            功能: 执行 invalidate_pending
            的同步逻辑,并协调 cancel_pending。
            参数: reason: str。 必填。
            契约: 同步调用。 返回 `None`。
            """
            _ = task_registry.cancel_pending(reason=reason)

        interaction_ingress.data.invalidate_pending = invalidate_pending

        return cls(
            scheduler=scheduler,
            task_registry=task_registry,
            task_reducer=task_reducer,
            executor=TaskLaneExecutor(task_registry, max_pending_per_lane=4),
            output_fence=SchedulerOutputFence(scheduler),
            interaction_ingress=interaction_ingress,
            mcp_dispatcher=ScopedMcpAdapterDispatcher(
                interaction_ingress.reducer,
                {McpCapability.PRESENTATION_DECK: LocalDeckAdapter()},
            ),
            mode_policy=AdaptiveAgentPolicy(),
            clock=clock,
        )

    @property
    def observables(self) -> RuntimeObservables:
        """函数契约说明.

        功能: 执行 observables 的同步逻辑,并协调
        RuntimeObservables, tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `RuntimeObservables`。
        """
        return RuntimeObservables(
            snapshot=self.scheduler.snapshot,
            dispatches=tuple(self._journal.dispatches),
            task_commits=tuple(self._journal.task_commits),
            generated_rtp=(),
            sound_transitions=(),
            rejections=tuple(self._journal.rejections),
        )

    def receive_comment(self, proposal: CommentProposal) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 receive_comment 的同步逻辑,并协调
        receive_comment, _reject, add,
        TurnId。
        参数: self 表示当前实例。 proposal:
        CommentProposal。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        correlation = proposal.correlation

        if correlation in self._correlations:
            return self._reject(correlation, "duplicate_correlation")

        outcome = self.interaction_ingress.receive_comment(
            text=proposal.text,
            correlation=correlation,
        )

        match outcome:
            case InteractionAccepted(turn_id=turn_id) if turn_id is not None:
                self._correlations.add(correlation)

                accepted_turn = TurnId(turn_id)

                self._journal.dispatches.append(
                    RuntimeDispatch(correlation, accepted_turn)
                )

                return RuntimeOutcome(
                    accepted=True,
                    correlation=correlation,
                    turn_id=accepted_turn,
                )

            case InteractionAccepted():
                return self._reject(correlation, "missing_turn")

            case _:
                return self._reject(correlation, "scheduler_rejected")

    def receive_control(self, raw_message: str) -> bool:
        """函数契约说明.

        功能: 执行 receive_control 的同步逻辑,并协调
        get,
        receive_presentation_result,
        parse_json_value, isinstance。
        参数: self 表示当前实例。 raw_message:
        str。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        parsed = parse_presentation_result_control(
            raw_message, self.scheduler.snapshot.session_id
        )
        if parsed is None:
            return False
        result, correlation = parsed
        _ = self.receive_presentation_result(
            result,
            correlation,
        )

        return True

    def receive_session_control(self, control: SessionControl) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 receive_session_control
        的同步逻辑,并协调 enroll_profile,
        revoke_profile_consent,
        receive_action,
        receive_presentation。
        参数: self 表示当前实例。 control:
        SessionControl。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        match control:
            case ProfileEnrollmentControl(
                enrollment=enrollment, correlation=correlation
            ):
                return self.enroll_profile(enrollment, correlation)

            case ProfileRevocationControl(
                profile_id=profile_id, correlation=correlation
            ):
                return self.revoke_profile_consent(profile_id, correlation)

            case ActionControl(proposal=proposal, correlation=correlation):
                return self.receive_action(proposal, correlation)

            case PresentationControl(proposal=proposal, correlation=correlation):
                return self.receive_presentation(proposal, correlation)

            case PresentationResultControl(result=result, correlation=correlation):
                return self.receive_presentation_result(result, correlation)

    async def receive_session_control_async(
        self, control: SessionControl
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行
        receive_session_control_async
        的异步逻辑,并协调
        receive_session_control,
        _schedule_presentation_mcp。
        参数: self 表示当前实例。 control:
        SessionControl。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `RuntimeOutcome`。
        """
        match control:
            case PresentationControl(proposal=proposal, correlation=correlation):
                return await self._schedule_presentation_mcp(proposal, correlation)

            case _:
                return self.receive_session_control(control)

    async def _schedule_presentation_mcp(
        self, proposal: PresentationCommand, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行
        _schedule_presentation_mcp
        的异步逻辑,并协调 receive_presentation,
        TaskRequest, McpIntent,
        schedule_mcp_task。
        参数: self 表示当前实例。 proposal:
        PresentationCommand。 必填。
        correlation: EventCorrelation。
        必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `RuntimeOutcome`。
        """
        outcome = self.receive_presentation(proposal, correlation)

        turn_id = self.scheduler.snapshot.active_turn_id

        if not outcome.accepted or turn_id is None:
            return outcome

        plan = build_presentation_mcp_plan(
            PresentationMcpPlanInput(
                proposal=proposal,
                snapshot=self.scheduler.snapshot,
                turn_id=turn_id,
                data_snapshot=self._task_data_snapshot,
                deadline_ms=self.clock() + 5_000,
            )
        )

        scheduled = self.schedule_mcp_task(plan.intent, plan.request, correlation)

        if not scheduled.accepted:
            self.interaction_ingress.reducer.cancel_presentation(proposal.command_id)

            return scheduled

        _ = await self.run_mcp_worker_async(
            now_ms=self.clock(), correlation=correlation
        )

        return scheduled

    def receive_asr_final(
        self,
        event: ASRAudienceEvent,
        correlation: EventCorrelation,
        gate: AsrSemanticGate,
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 receive_asr_final
        的同步逻辑,并协调 apply, _reject,
        evaluate, StartTurn。
        参数: self 表示当前实例。 event:
        ASRAudienceEvent。 必填。
        correlation: EventCorrelation。
        必填。 gate: AsrSemanticGate。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        if correlation in self._correlations:
            return self._reject(correlation, "duplicate_correlation")

        if gate.evaluate(event.text) is AsrGateDecision.DISCARD:
            return self._reject(correlation, "asr_gate_discarded")

        transition = self.scheduler.apply(
            StartTurn(
                expected_revision=self.scheduler.snapshot.revision,
                event=SchedulerEvent(event_type="asr.final", correlation=correlation),
            )
        )

        match transition:
            case TransitionAccepted(accepted_event=accepted_event):
                self._correlations.add(correlation)

                self._journal.dispatches.append(
                    RuntimeDispatch(correlation, accepted_event.turn_id)
                )

                return RuntimeOutcome(
                    accepted=True,
                    correlation=correlation,
                    turn_id=accepted_event.turn_id,
                )

            case _:
                return self._reject(correlation, "scheduler_rejected")

    def enroll_profile(
        self, enrollment: ProfileEnrollment, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 enroll_profile 的同步逻辑,并协调
        enroll_profile,
        _interaction_outcome。
        参数: self 表示当前实例。 enrollment:
        ProfileEnrollment。 必填。
        correlation: EventCorrelation。
        必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        _ = self.interaction_ingress.data.enroll_profile(enrollment)

        return self._interaction_outcome(
            correlation, "profile_enrolled", accepted=True, task_id=None
        )

    def revoke_profile_consent(
        self, profile_id: VoiceProfileId, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 revoke_profile_consent
        的同步逻辑,并协调
        revoke_profile_consent,
        _interaction_outcome。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。 correlation:
        EventCorrelation。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        self.interaction_ingress.data.revoke_profile_consent(profile_id)

        return self._interaction_outcome(
            correlation, "profile_revoked", accepted=True, task_id=None
        )

    def receive_action(
        self, proposal: ActionProposal, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 receive_action 的同步逻辑,并协调
        _interaction_outcome,
        isinstance, receive_action。
        参数: self 表示当前实例。 proposal:
        ActionProposal。 必填。 correlation:
        EventCorrelation。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        return self._interaction_outcome(
            correlation,
            "action",
            isinstance(
                self.interaction_ingress.receive_action(proposal), InteractionAccepted
            ),
            None,
        )

    def receive_presentation(
        self, proposal: PresentationCommand, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 receive_presentation
        的同步逻辑,并协调 _interaction_outcome,
        isinstance,
        receive_presentation。
        参数: self 表示当前实例。 proposal:
        PresentationCommand。 必填。
        correlation: EventCorrelation。
        必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        outcome = self._interaction_outcome(
            correlation,
            "presentation_command",
            isinstance(
                self.interaction_ingress.receive_presentation(proposal),
                InteractionAccepted,
            ),
            None,
        )

        if outcome.accepted:
            self._presentation_correlations[proposal.command_id] = correlation

        return outcome

    def receive_presentation_result(
        self, result: PresentationResult, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行
        receive_presentation_result
        的同步逻辑,并协调 get,
        _interaction_outcome,
        isinstance,
        receive_presentation_result。
        参数: self 表示当前实例。 result:
        PresentationResult。 必填。
        correlation: EventCorrelation。
        必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        expected = self._presentation_correlations.get(result.command_id)

        if (
            expected is None
            or expected.trace_id != correlation.trace_id
            or expected.session_id != correlation.session_id
            or expected.sequence != correlation.sequence
        ):
            return self._interaction_outcome(
                correlation, "presentation_ack", accepted=False, task_id=None
            )

        return self._interaction_outcome(
            correlation,
            "presentation_ack",
            isinstance(
                self.interaction_ingress.receive_presentation_result(result),
                InteractionAccepted,
            ),
            None,
        )

    def schedule_mcp_task(
        self,
        intent: McpIntent,
        request: TaskRequest,
        correlation: EventCorrelation,
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 schedule_mcp_task
        的同步逻辑,并协调 schedule_task,
        _record_interaction。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。 request:
        TaskRequest。 必填。 correlation:
        EventCorrelation。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        outcome = self.schedule_task(request, correlation)

        if outcome.accepted:
            self._mcp_intents[request.task_id] = intent

        self.operational_journal.append(
            interaction_record(
                correlation,
                self.scheduler.snapshot,
                "mcp_task",
                outcome.accepted,
                request.task_id,
            )
        )

        return outcome

    def run_mcp_worker(
        self, *, now_ms: int, correlation: EventCorrelation
    ) -> McpDispatchOutcome:
        """函数契约说明.

        功能: 运行流程并协调其依赖步骤。
        参数: self 表示当前实例。 now_ms: int。
        必填。 correlation:
        EventCorrelation。 必填。
        契约: 同步调用。 返回
        `McpDispatchOutcome`。
        """
        request = self.next_task(now_ms=now_ms)

        if request is None:
            self._discard_terminal_mcp_intents()

            return McpDispatchOutcome(accepted=False)

        intent = self._mcp_intents.pop(request.task_id, None)

        if intent is None:
            return McpDispatchOutcome(accepted=False)

        outcome = self.mcp_dispatcher.dispatch(intent, now_ms=now_ms)

        if not outcome.accepted:
            self._mcp_intents[request.task_id] = intent

            return outcome

        _ = self.reduce_task(
            TaskResult(
                request.task_id,
                request.session_id,
                request.turn_id,
                request.snapshot_revision,
                TaskEffect("presentation.dispatch", str(intent.command.command_id)),
            ),
            correlation,
        )

        return outcome

    async def run_mcp_worker_async(
        self, *, now_ms: int, correlation: EventCorrelation
    ) -> McpDispatchOutcome:
        """函数契约说明.

        功能: 运行流程并协调其依赖步骤。
        参数: self 表示当前实例。 now_ms: int。
        必填。 correlation:
        EventCorrelation。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `McpDispatchOutcome`。
        """
        request = self.next_task(now_ms=now_ms)

        if request is None:
            self._discard_terminal_mcp_intents()

            return McpDispatchOutcome(accepted=False)

        intent = self._mcp_intents.pop(request.task_id, None)

        if intent is None:
            return McpDispatchOutcome(accepted=False)

        self._active_mcp_tasks[request.task_id] = intent.command.command_id

        try:
            outcome = await self.mcp_dispatcher.dispatch_async(intent, now_ms=now_ms)

        finally:
            _ = self._active_mcp_tasks.pop(request.task_id, None)

        self.operational_journal.append(
            interaction_record(
                correlation,
                self.scheduler.snapshot,
                "mcp_dispatch",
                outcome.accepted,
                request.task_id,
            )
        )

        if not outcome.accepted:
            self._mcp_intents[request.task_id] = intent

            return outcome

        _ = self.reduce_task(
            TaskResult(
                request.task_id,
                request.session_id,
                request.turn_id,
                request.snapshot_revision,
                TaskEffect("presentation.dispatch", str(intent.command.command_id)),
            ),
            correlation,
        )

        return outcome

    def reconcile_mcp_worker(
        self,
        task_id: TaskId,
        *,
        now_ms: int,
        correlation: EventCorrelation,
    ) -> McpDispatchOutcome:
        """函数契约说明.

        功能: 执行 reconcile_mcp_worker
        的同步逻辑,并协调 get, reconcile, task,
        pop。
        参数: self 表示当前实例。 task_id:
        TaskId。 必填。 now_ms: int。 必填。
        correlation: EventCorrelation。
        必填。
        契约: 同步调用。 返回
        `McpDispatchOutcome`。
        """
        intent = self._mcp_intents.get(task_id)

        if intent is None:
            return McpDispatchOutcome(accepted=False)

        if now_ms > intent.deadline_ms:
            _ = self.task_registry.timeout(task_id)

            _ = self._mcp_intents.pop(task_id)

            _ = self.mcp_dispatcher.reconcile(intent.command.command_id, now_ms=now_ms)

            return McpDispatchOutcome(accepted=False)

        outcome = self.mcp_dispatcher.reconcile(
            intent.command.command_id, now_ms=now_ms
        )

        if not outcome.accepted:
            return outcome

        request = self.task_registry.task(task_id)

        if request is None:
            return McpDispatchOutcome(accepted=False)

        _ = self._mcp_intents.pop(task_id)

        _ = self.reduce_task(
            TaskResult(
                task_id,
                request.request.session_id,
                request.request.turn_id,
                request.request.snapshot_revision,
                TaskEffect("presentation.dispatch", str(intent.command.command_id)),
            ),
            correlation,
        )

        return outcome

    def cancel_task(
        self, task_id: TaskId, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 cancel_task 的同步逻辑,并协调
        cancel, _interaction_outcome,
        pop, get。
        参数: self 表示当前实例。 task_id:
        TaskId。 必填。 correlation:
        EventCorrelation。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        cancelled = self.task_registry.cancel(task_id, reason="cancelled")

        if cancelled is not None:
            intent = self._mcp_intents.pop(task_id, None)

            if intent is not None:
                _ = self.mcp_dispatcher.cancel(intent.command.command_id)

                self.interaction_ingress.reducer.cancel_presentation(
                    intent.command.command_id
                )

            active_command = self._active_mcp_tasks.get(task_id)

            if active_command is not None:
                _ = self.mcp_dispatcher.cancel(active_command)

                self.interaction_ingress.reducer.cancel_presentation(active_command)

        return self._interaction_outcome(
            correlation,
            "task_cancelled",
            cancelled is not None,
            task_id,
        )

    def _discard_terminal_mcp_intents(self) -> None:
        """函数契约说明.

        功能: 执行
        _discard_terminal_mcp_intents
        的同步逻辑,并协调 tuple, task, pop。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        for task_id in tuple(self._mcp_intents):
            record = self.task_registry.task(task_id)

            if record is None or record.state is not TaskState.PENDING:
                _ = self._mcp_intents.pop(task_id)

    @property
    def _task_data_snapshot(self) -> TaskStateSnapshot:
        return self.interaction_ingress.data.task_snapshot

    def _admit_task(self, request: TaskRequest) -> TaskRegistrationResult:
        rejection = scheduling_rejection(request, self.scheduler.snapshot)

        if rejection is not None:
            return TaskRegistrationRejected(rejection)

        if request.data_snapshot != self._task_data_snapshot:
            return TaskRegistrationRejected(TaskRegistrationRejection.STALE_SNAPSHOT)

        return self.task_registry.register(request)

    def schedule_task(
        self, request: TaskRequest, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 schedule_task 的同步逻辑,并协调
        _with_current_data_snapshot,
        schedule, enqueue, append。
        参数: self 表示当前实例。 request:
        TaskRequest。 必填。 correlation:
        EventCorrelation。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        task_request = with_current_data_snapshot(request, self._task_data_snapshot)

        admission = self._admit_task(task_request)

        match admission:
            case TaskRegistrationAccepted() if self.executor.enqueue(task_request):
                self.operational_journal.append(
                    OperationalRecord(
                        stage="task_scheduled",
                        trace_id=str(correlation.trace_id),
                        session_id=str(correlation.session_id),
                        turn_id=str(request.turn_id),
                        segment_id=None,
                        task_id=str(request.task_id),
                        outcome="accepted",
                    )
                )

                return RuntimeOutcome(accepted=True, correlation=correlation)

            case TaskRegistrationAccepted():
                _ = self.task_registry.withdraw(task_request.task_id)

                return self._reject(correlation, "task_queue_full")

            case _:
                self.operational_journal.append(
                    OperationalRecord(
                        stage="task_scheduled",
                        trace_id=str(correlation.trace_id),
                        session_id=str(correlation.session_id),
                        turn_id=str(request.turn_id),
                        segment_id=None,
                        task_id=str(request.task_id),
                        outcome="rejected",
                    )
                )

                return self._reject(correlation, "task_rejected")

    def next_task(self, *, now_ms: int) -> TaskRequest | None:
        """函数契约说明.

        功能: 执行 next_task 的同步逻辑,并协调 next。
        参数: self 表示当前实例。 now_ms: int。
        必填。
        契约: 同步调用。 返回 `TaskRequest |
        None`。
        """
        return self.executor.next(now_ms=now_ms)

    def reduce_task(
        self, result: TaskResult, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 reduce_task 的同步逻辑,并协调
        reduce, clock, append,
        RuntimeOutcome。
        参数: self 表示当前实例。 result:
        TaskResult。 必填。 correlation:
        EventCorrelation。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        outcome = self.task_reducer.reduce(
            result,
            snapshot=self.scheduler.snapshot,
            data_snapshot=self._task_data_snapshot,
            now_ms=self.clock(),
        )

        match outcome:
            case TaskResultAccepted():
                self._journal.task_commits.append(result)

                self.operational_journal.append(
                    task_result_record(result, correlation, "accepted")
                )

                return RuntimeOutcome(accepted=True, correlation=correlation)

            case _:
                self.operational_journal.append(
                    task_result_record(result, correlation, "rejected")
                )

                return self._reject(correlation, "task_result_rejected")

    def _reject(self, correlation: EventCorrelation, reason: str) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 _reject 的同步逻辑,并协调 append,
        RuntimeOutcome,
        RuntimeRejection。
        参数: self 表示当前实例。 correlation:
        EventCorrelation。 必填。 reason:
        str。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        self._journal.rejections.append(RuntimeRejection(correlation, reason))

        return RuntimeOutcome(accepted=False, correlation=correlation)

    def _interaction_outcome(
        self,
        correlation: EventCorrelation,
        stage: str,
        accepted: bool,
        task_id: TaskId | None,
    ) -> RuntimeOutcome:
        """函数契约说明.

        功能: 执行 _interaction_outcome
        的同步逻辑,并协调 _record_interaction,
        _reject, RuntimeOutcome。
        参数: self 表示当前实例。 correlation:
        EventCorrelation。 必填。 stage:
        str。 必填。 accepted: bool。 必填。
        task_id: TaskId | None。 必填。
        契约: 同步调用。 返回 `RuntimeOutcome`。
        """
        self.operational_journal.append(
            interaction_record(
                correlation,
                self.scheduler.snapshot,
                stage,
                accepted,
                task_id,
            )
        )

        if accepted:
            return RuntimeOutcome(accepted=True, correlation=correlation)

        return self._reject(correlation, stage)
