"""Production composition for one scheduler-owned live session."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.interactions import (
    ActionProposal,
    CommandId,
    CommentProposal,
    InteractionAccepted,
    McpCapability,
    McpDispatchProposal,
    PresentationCommand,
    PresentationResult,
)
from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.mcp_adapters import (
    LocalDeckAdapter,
    McpDispatchOutcome,
    McpIntent,
    ScopedMcpAdapterDispatcher,
)
from orchestrator.modes import (
    LecturerModePolicy,
    LecturerState,
    ModePolicy,
    OnsiteExplainerModePolicy,
    OrchestratorMode,
    ScriptStep,
    SlideStep,
    VirtualStreamerModePolicy,
)
from orchestrator.operational_journal import OperationalJournal, OperationalRecord
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.runtime_contracts import (
    RuntimeDispatch,
    RuntimeObservables,
    RuntimeOutcome,
    RuntimeRejection,
)
from orchestrator.scheduler_reflex import SchedulerOutputFence
from orchestrator.scheduler_tasks import SchedulerTaskFacade
from orchestrator.sessions import (
    EventCorrelation,
    EventSequence,
    SchedulerEvent,
    SessionScheduler,
    SessionSnapshot,
    StartTurn,
    TransitionAccepted,
)
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.streaming_contracts import CancellationEpoch
from orchestrator.task_executor import TaskLaneExecutor
from orchestrator.task_reducer import TaskEffect, TaskResult, TaskResultAccepted
from orchestrator.task_registry import (
    IdempotencyKey,
    SchedulerTaskConfig,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRegistrationAccepted,
    TaskRequest,
    TaskState,
)


def _monotonic_ms() -> int:
    return monotonic_ns() // 1_000_000


@dataclass(slots=True)
class _RuntimeJournal:
    """Mutable append-only records owned exclusively by the session runtime."""

    dispatches: list[RuntimeDispatch] = field(default_factory=list)
    task_commits: list[TaskResult] = field(default_factory=list)
    rejections: list[RuntimeRejection] = field(default_factory=list)


@dataclass(slots=True)
class SessionRuntime:
    """Compose canonical session state, task reduction, and effect admission."""

    scheduler: SessionScheduler
    tasks: SchedulerTaskFacade
    executor: TaskLaneExecutor
    output_fence: SchedulerOutputFence
    interaction_ingress: SessionInteractionIngress
    mcp_dispatcher: ScopedMcpAdapterDispatcher
    mode_policy: (
        LecturerModePolicy | VirtualStreamerModePolicy | OnsiteExplainerModePolicy
    )
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
        mode: OrchestratorMode,
        clock: Callable[[], int] = _monotonic_ms,
    ) -> "SessionRuntime":
        """Create every scheduler-owned control component for one live session."""
        scheduler = SessionScheduler(
            session_id=session_id,
            turn_id_prefix=turn_id_prefix,
        )
        interaction_ingress = SessionInteractionIngress.create(scheduler)
        tasks = SchedulerTaskFacade.create(
            scheduler,
            task_config,
            data_snapshot_provider=lambda: interaction_ingress.data.task_snapshot,
        )

        def invalidate_pending(reason: str) -> None:
            _ = tasks.registry.cancel_pending(reason=reason)

        interaction_ingress.data.invalidate_pending = invalidate_pending
        return cls(
            scheduler=scheduler,
            tasks=tasks,
            executor=TaskLaneExecutor(tasks.registry, max_pending_per_lane=4),
            output_fence=SchedulerOutputFence(scheduler),
            interaction_ingress=interaction_ingress,
            mcp_dispatcher=ScopedMcpAdapterDispatcher(
                interaction_ingress.reducer,
                {McpCapability.PRESENTATION_DECK: LocalDeckAdapter()},
            ),
            mode_policy=_mode_policy(mode),
            clock=clock,
        )

    @property
    def observables(self) -> RuntimeObservables:
        """Expose immutable records without exposing mutable scheduler internals."""
        return RuntimeObservables(
            snapshot=self.scheduler.snapshot,
            dispatches=tuple(self._journal.dispatches),
            task_commits=tuple(self._journal.task_commits),
            generated_rtp=(),
            sound_transitions=(),
            rejections=tuple(self._journal.rejections),
        )

    def receive_comment(self, proposal: CommentProposal) -> RuntimeOutcome:
        """Reduce one already-parsed external comment through the live scheduler."""
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
        """Reduce a correlated canonical presentation acknowledgement from control."""
        try:
            value = parse_json_value(raw_message)
        except JsonBoundaryError:
            return False
        if (
            not isinstance(value, dict)
            or value.get("event_type") != "presentation.result"
        ):
            return False
        data = value.get("data")
        trace_id = value.get("trace_id")
        session_id = value.get("session_id")
        sequence = value.get("seq")
        command_id = data.get("command_id") if isinstance(data, dict) else None
        succeeded = data.get("succeeded") if isinstance(data, dict) else None
        if (
            value.get("source") != "frontend"
            or not isinstance(trace_id, str)
            or not isinstance(session_id, str)
            or session_id != self.scheduler.snapshot.session_id
            or type(sequence) is not int
            or not isinstance(command_id, str)
            or type(succeeded) is not bool
        ):
            return False
        _ = self.receive_presentation_result(
            PresentationResult(CommandId(command_id), succeeded),
            EventCorrelation(
                TraceId(trace_id),
                SessionId(session_id),
                EventSequence(sequence),
            ),
        )
        return True

    def receive_session_control(self, control: SessionControl) -> RuntimeOutcome:
        """Route a typed, authenticated control fact through scheduler authorities."""
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
        """Route an authenticated presentation command through admitted MCP work."""
        match control:
            case PresentationControl(proposal=proposal, correlation=correlation):
                return await self._schedule_presentation_mcp(proposal, correlation)
            case _:
                return self.receive_session_control(control)

    async def _schedule_presentation_mcp(
        self, proposal: PresentationCommand, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        outcome = self.receive_presentation(proposal, correlation)
        turn_id = self.scheduler.snapshot.active_turn_id
        if not outcome.accepted or turn_id is None:
            return outcome
        deadline_ms = self.clock() + 5_000
        request = TaskRequest(
            task_id=TaskId(f"mcp-{proposal.command_id}"),
            session_id=self.scheduler.snapshot.session_id,
            turn_id=turn_id,
            parent_task_id=None,
            deadline_ms=TaskDeadlineMs(deadline_ms),
            snapshot_revision=self.scheduler.snapshot.revision,
            idempotency_key=IdempotencyKey(f"mcp-{proposal.command_id}"),
            kind=TaskKind.INTERACTIVE,
            data_snapshot=self.tasks.data_snapshot,
        )
        intent = McpIntent(
            McpDispatchProposal(
                McpCapability.PRESENTATION_DECK,
                proposal.command_id,
                cancelled=False,
            ),
            proposal,
            deadline_ms,
        )
        scheduled = self.schedule_mcp_task(intent, request, correlation)
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
        """Open a turn only after the interactive semantic gate accepts an ASR final."""
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
        """Enroll an explicitly consented profile through session-owned data state."""
        _ = self.interaction_ingress.data.enroll_profile(enrollment)
        return self._interaction_outcome(
            correlation, "profile_enrolled", accepted=True, task_id=None
        )

    def revoke_profile_consent(
        self, profile_id: VoiceProfileId, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """Revoke consent and invalidate pending work through one authority."""
        self.interaction_ingress.data.revoke_profile_consent(profile_id)
        return self._interaction_outcome(
            correlation, "profile_revoked", accepted=True, task_id=None
        )

    def receive_action(
        self, proposal: ActionProposal, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """Journal the reducer-approved or rejected finite action outcome."""
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
        """Journal the reducer-approved or rejected presentation intent."""
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
        """Journal a reducer-owned presentation acknowledgement outcome."""
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
        """Schedule only reducer-approved scoped MCP work through task admission."""
        outcome = self.schedule_task(request, correlation)
        if outcome.accepted:
            self._mcp_intents[request.task_id] = intent
        self._record_interaction(
            correlation, "mcp_task", outcome.accepted, request.task_id
        )
        return outcome

    def run_mcp_worker(
        self, *, now_ms: int, correlation: EventCorrelation
    ) -> McpDispatchOutcome:
        """Execute one selected MCP intent without granting adapter state authority."""
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
        """Run one admitted deck task with an active cancellation handle."""
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
        self._record_interaction(
            correlation, "mcp_dispatch", outcome.accepted, request.task_id
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
        """Resolve one scheduler-retained ambiguous deck invocation."""
        intent = self._mcp_intents.get(task_id)
        if intent is None:
            return McpDispatchOutcome(accepted=False)
        if now_ms > intent.deadline_ms:
            _ = self.tasks.registry.timeout(task_id)
            _ = self._mcp_intents.pop(task_id)
            _ = self.mcp_dispatcher.reconcile(intent.command.command_id, now_ms=now_ms)
            return McpDispatchOutcome(accepted=False)
        outcome = self.mcp_dispatcher.reconcile(
            intent.command.command_id, now_ms=now_ms
        )
        if not outcome.accepted:
            return outcome
        request = self.tasks.registry.task(task_id)
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
        """Cancel scheduler-authorized work without exposing registry mutation."""
        cancelled = self.tasks.registry.cancel(task_id, reason="cancelled")
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
        for task_id in tuple(self._mcp_intents):
            record = self.tasks.registry.task(task_id)
            if record is None or record.state is not TaskState.PENDING:
                _ = self._mcp_intents.pop(task_id)

    def schedule_task(
        self, request: TaskRequest, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """Register typed work only through the scheduler-owned task facade."""
        task_request = _with_current_data_snapshot(request, self.tasks.data_snapshot)
        admission = self.tasks.schedule(task_request)
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
                _ = self.tasks.registry.withdraw(task_request.task_id)
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
        """Select one bounded task in lane priority order for a worker boundary."""
        return self.executor.next(now_ms=now_ms)

    def reduce_task(
        self, result: TaskResult, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        """Admit worker completion through the existing TaskResultReducer facade."""
        outcome = self.tasks.reduce(result, now_ms=self.clock())
        match outcome:
            case TaskResultAccepted():
                self._journal.task_commits.append(result)
                self.operational_journal.append(
                    _task_result_record(result, correlation, "accepted")
                )
                return RuntimeOutcome(accepted=True, correlation=correlation)
            case _:
                self.operational_journal.append(
                    _task_result_record(result, correlation, "rejected")
                )
                return self._reject(correlation, "task_result_rejected")

    def _reject(self, correlation: EventCorrelation, reason: str) -> RuntimeOutcome:
        self._journal.rejections.append(RuntimeRejection(correlation, reason))
        return RuntimeOutcome(accepted=False, correlation=correlation)

    def _interaction_outcome(
        self,
        correlation: EventCorrelation,
        stage: str,
        accepted: bool,
        task_id: TaskId | None,
    ) -> RuntimeOutcome:
        self._record_interaction(correlation, stage, accepted, task_id)
        if accepted:
            return RuntimeOutcome(accepted=True, correlation=correlation)
        return self._reject(correlation, stage)

    def _record_interaction(
        self,
        correlation: EventCorrelation,
        stage: str,
        accepted: bool,
        task_id: TaskId | None,
    ) -> None:
        self.operational_journal.append(
            OperationalRecord(
                stage=stage,
                trace_id=str(correlation.trace_id),
                session_id=str(correlation.session_id),
                turn_id=_active_turn_id(self.scheduler.snapshot),
                segment_id=None,
                task_id=None if task_id is None else str(task_id),
                outcome="accepted" if accepted else "rejected",
            )
        )


def _task_result_record(
    result: TaskResult, correlation: EventCorrelation, outcome: str
) -> OperationalRecord:
    return OperationalRecord(
        stage="task_result",
        trace_id=str(correlation.trace_id),
        session_id=str(correlation.session_id),
        turn_id=str(result.turn_id),
        segment_id=None,
        task_id=str(result.task_id),
        outcome=outcome,
    )


def _active_turn_id(snapshot: SessionSnapshot) -> str | None:
    active_turn_id = snapshot.active_turn_id
    if active_turn_id is None:
        return None
    return str(active_turn_id)


def _mode_policy(
    mode: OrchestratorMode,
) -> LecturerModePolicy | VirtualStreamerModePolicy | OnsiteExplainerModePolicy:
    """Build the selected production mode policy without a second state machine."""
    match mode:
        case OrchestratorMode.LECTURER:
            return ModePolicy.lecturer(
                LecturerState(
                    script_step=ScriptStep(0),
                    slide_step=SlideStep(1),
                    immediate_interruption_enabled=True,
                    qa_window=None,
                )
            )
        case OrchestratorMode.VIRTUAL_STREAMER:
            return ModePolicy.virtual_streamer(topic="")
        case OrchestratorMode.ONSITE_EXPLAINER:
            return ModePolicy.onsite_explainer()


def _with_current_data_snapshot(
    request: TaskRequest,
    data_snapshot: TaskStateSnapshot,
) -> TaskRequest:
    if request.data_snapshot == TaskStateSnapshot.initial():
        return replace(request, data_snapshot=data_snapshot)
    return request
