import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Protocol

from orchestrator.agent_pipeline import (
    AgentPipeline,
    AsyncAgentPipeline,
    BrainStateSnapshot,
    FrontendOperation,
    GateDecision,
    MediaOperation,
    PlanAccepted,
    PlanResult,
    TaskSnapshot,
)
from orchestrator.agent_pipeline import (
    AudienceInput as BrainAudienceInput,
)
from orchestrator.agent_pipeline import (
    AudienceSource as BrainAudienceSource,
)
from orchestrator.asr_semantic_gate import AsrGateDecision, AsrSemanticGate
from orchestrator.brain_runtime import build_mock_agent_pipeline
from orchestrator.control_ingress import (
    ActionControl,
    ContextResetControl,
    MemoryDeleteControl,
    PresentationControl,
    PresentationResultControl,
    ProfileEnrollmentControl,
    ProfileRevocationControl,
    SessionControl,
    SessionEndControl,
)
from orchestrator.identity import ProfileEnrollment, VoiceProfileId
from orchestrator.ids import SegmentId, SessionId, TurnId
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.interactions import (
    ActionProposal,
    CommandId,
    CommentProposal,
    InteractionAccepted,
    PresentationCommand,
    PresentationResult,
)
from orchestrator.mcp_adapters import (
    DeckDispatchIntent,
    DeckDispatchOutcome,
    DeckEffectDispatcher,
    LocalDeckEffectExecutor,
)
from orchestrator.memory import (
    MemoryCategory,
    MemoryConfidence,
    MemoryKey,
    MemoryProposal,
    MemoryProvenance,
    MemorySource,
    ProposalRevision,
)
from orchestrator.memory_store import render_markdown_memory
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
    IdempotencyKey,
    SchedulerTaskConfig,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRegistrationAccepted,
    TaskRegistrationRejected,
    TaskRegistrationRejection,
    TaskRegistrationResult,
    TaskRegistry,
    TaskRequest,
    TaskState,
)
from orchestrator.transient_context import (
    AcceptedOutput,
    ContextCompactionError,
    ContextComposition,
    ContextProvenance,
    ContextSequence,
    ContextSourceId,
    FinalizedInput,
    ModelContextBudget,
    ModelId,
    StaticContextBudgetPolicy,
    TokenBudget,
    ToolObservation,
)
from orchestrator.transport_control import VoiceEvidence

_LOGGER = logging.getLogger(__name__)

def _monotonic_ms() -> int:
    return monotonic_ns() // 1_000_000


def _brain_task_kind(value: object) -> TaskKind | None:
    if value in {"tts", "playback"}:
        return TaskKind.INTERACTIVE
    if value in {"retrieval", "mcp"}:
        return TaskKind.DELIBERATIVE
    if value in {"context_compaction", "memory_patch"}:
        return TaskKind.MAINTENANCE
    return None


class AgentEffectDispatcher(Protocol):
    """Dedicated adapters execute effects only after reducer acceptance."""

    def dispatch_media(
        self,
        operation: MediaOperation,
        *,
        session_id: SessionId,
        turn_id: TurnId,
        cancellation_epoch: CancellationEpoch,
    ) -> None: ...

    def dispatch_frontend(
        self,
        operation: FrontendOperation,
        *,
        session_id: SessionId,
        turn_id: TurnId,
    ) -> None: ...


_BRAIN_CONTEXT_MODEL = ModelId("agent-brain")

_BRAIN_CONTEXT_POLICY = StaticContextBudgetPolicy(
    model_id=_BRAIN_CONTEXT_MODEL,
    budget=ModelContextBudget(input_tokens=TokenBudget(512)),
)

_DEFAULT_AGENT_CAPABILITIES = frozenset(
    {
        "knowledge.lookup",
        "task:tts",
        "task:playback",
        "task:retrieval",
        "task:context_compaction",
        "task:memory_patch",
    }
)


@dataclass(slots=True)
class _RuntimeJournal:
    dispatches: list[RuntimeDispatch] = field(default_factory=list)
    task_commits: list[TaskResult] = field(default_factory=list)
    rejections: list[RuntimeRejection] = field(default_factory=list)


@dataclass(slots=True)
class SessionRuntime:
    scheduler: SessionScheduler

    task_registry: TaskRegistry

    task_reducer: TaskResultReducer

    executor: TaskLaneExecutor

    output_fence: SchedulerOutputFence

    interaction_ingress: SessionInteractionIngress

    deck_dispatcher: DeckEffectDispatcher

    mode_policy: AdaptiveAgentPolicy

    agent_pipeline: AgentPipeline | None = None

    async_agent_pipeline: AsyncAgentPipeline | None = None

    agent_capabilities: frozenset[str] = frozenset()

    agent_mcp_allowlist: frozenset[str] = frozenset()

    agent_effect_dispatcher: AgentEffectDispatcher | None = None

    clock: Callable[[], int] = _monotonic_ms

    cancellation_epoch: CancellationEpoch = field(
        default_factory=lambda: CancellationEpoch(0)
    )

    _correlations: set[EventCorrelation] = field(default_factory=set)

    _deck_intents: dict[TaskId, DeckDispatchIntent] = field(default_factory=dict)

    _active_deck_tasks: dict[TaskId, CommandId] = field(default_factory=dict)

    _presentation_correlations: dict[CommandId, EventCorrelation] = field(
        default_factory=dict
    )

    _journal: _RuntimeJournal = field(default_factory=_RuntimeJournal)

    _agent_results: list[PlanResult] = field(default_factory=list)

    _agent_tts_text: dict[TaskId, str] = field(default_factory=dict)

    _voice_evidence_ranges: list[tuple[str, int, int]] = field(default_factory=list)

    _frontend_caption: str = ""

    _frontend_animation: str | None = None

    _planned_ppt_deck_id: str | None = None

    _planned_ppt_page: int | None = None

    _ended: bool = False

    operational_journal: OperationalJournal = field(default_factory=OperationalJournal)

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        *,
        session_id: SessionId,
        turn_id_prefix: str,
        task_config: SchedulerTaskConfig,
        clock: Callable[[], int] = _monotonic_ms,
        agent_pipeline: AgentPipeline | None = None,
        async_agent_pipeline: AsyncAgentPipeline | None = None,
        agent_capabilities: frozenset[str] | None = None,
        agent_mcp_allowlist: frozenset[str] | None = None,
        agent_effect_dispatcher: AgentEffectDispatcher | None = None,
    ) -> "SessionRuntime":
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
            _ = task_registry.cancel_pending(reason=reason)

        interaction_ingress.data.invalidate_pending = invalidate_pending

        return cls(
            scheduler=scheduler,
            task_registry=task_registry,
            task_reducer=task_reducer,
            executor=TaskLaneExecutor(task_registry, max_pending_per_lane=4),
            output_fence=SchedulerOutputFence(scheduler),
            interaction_ingress=interaction_ingress,
            deck_dispatcher=DeckEffectDispatcher(
                interaction_ingress.reducer, LocalDeckEffectExecutor()
            ),
            mode_policy=AdaptiveAgentPolicy(),
            clock=clock,
            agent_pipeline=(
                build_mock_agent_pipeline(interaction_ingress.data.retrieval)
                if agent_pipeline is None
                else agent_pipeline
            ),
            async_agent_pipeline=async_agent_pipeline,
            agent_capabilities=(
                _DEFAULT_AGENT_CAPABILITIES
                if agent_capabilities is None
                else agent_capabilities
            ),
            agent_mcp_allowlist=(
                frozenset() if agent_mcp_allowlist is None else agent_mcp_allowlist
            ),
            agent_effect_dispatcher=agent_effect_dispatcher,
        )

    @property
    def agent_results(self) -> tuple[PlanResult, ...]:
        return tuple(self._agent_results)

    @property
    def voice_evidence_ranges(self) -> tuple[tuple[str, int, int], ...]:
        """Non-sensitive evidence correlation only; embeddings never persist."""
        return tuple(self._voice_evidence_ranges)

    def receive_voice_evidence(self, evidence: VoiceEvidence) -> bool:
        if evidence.session_id != str(self.scheduler.snapshot.session_id):
            return False
        self._voice_evidence_ranges.append(
            (
                evidence.stream_id,
                evidence.rtp_start_timestamp,
                evidence.rtp_end_timestamp,
            )
        )
        return True

    @property
    def observables(self) -> RuntimeObservables:
        return RuntimeObservables(
            snapshot=self.scheduler.snapshot,
            dispatches=tuple(self._journal.dispatches),
            task_commits=tuple(self._journal.task_commits),
            generated_rtp=(),
            sound_transitions=(),
            rejections=tuple(self._journal.rejections),
        )

    def receive_comment(self, proposal: CommentProposal) -> RuntimeOutcome:
        correlation = proposal.correlation

        if self._ended:
            return self._reject(correlation, "session_ended")

        if correlation in self._correlations:
            return self._reject(correlation, "duplicate_correlation")

        agent_input = self._submit_agent_comment(proposal)
        if agent_input is None:
            return self._reject(correlation, "agent_gate_discarded")

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

                self._run_agent_plan(agent_input, accepted_turn, correlation)

                return RuntimeOutcome(
                    accepted=True,
                    correlation=correlation,
                    turn_id=accepted_turn,
                )

            case InteractionAccepted():
                return self._reject(correlation, "missing_turn")

            case _:
                return self._reject(correlation, "scheduler_rejected")

    async def receive_comment_async(self, proposal: CommentProposal) -> RuntimeOutcome:
        pipeline = self.async_agent_pipeline
        if pipeline is None:
            return self.receive_comment(proposal)
        correlation = proposal.correlation
        if self._ended:
            return self._reject(correlation, "session_ended")
        if correlation in self._correlations:
            return self._reject(correlation, "duplicate_correlation")
        audience_input = BrainAudienceInput(
            session_id=str(correlation.session_id),
            trace_id=str(correlation.trace_id),
            sequence=int(correlation.sequence),
            source=BrainAudienceSource.COMMENT,
            received_at_ms=self.clock(),
            text=proposal.text,
        )
        if await pipeline.submit(audience_input) is GateDecision.DISCARD:
            return self._reject(correlation, "agent_gate_discarded")
        outcome = self.interaction_ingress.receive_comment(
            text=proposal.text, correlation=correlation
        )
        if not isinstance(outcome, InteractionAccepted) or outcome.turn_id is None:
            return self._reject(correlation, "scheduler_rejected")
        accepted_turn = TurnId(outcome.turn_id)
        self._correlations.add(correlation)
        self._journal.dispatches.append(RuntimeDispatch(correlation, accepted_turn))
        await self._run_async_agent_plan(
            pipeline, audience_input, accepted_turn, correlation
        )
        return RuntimeOutcome(
            accepted=True, correlation=correlation, turn_id=accepted_turn
        )

    def _submit_agent_comment(
        self, proposal: CommentProposal
    ) -> BrainAudienceInput | None:
        pipeline = self.agent_pipeline
        if pipeline is None:
            return BrainAudienceInput(
                session_id=str(proposal.correlation.session_id),
                trace_id=str(proposal.correlation.trace_id),
                sequence=int(proposal.correlation.sequence),
                source=BrainAudienceSource.COMMENT,
                received_at_ms=self.clock(),
                text=proposal.text,
            )
        audience_input = BrainAudienceInput(
            session_id=str(proposal.correlation.session_id),
            trace_id=str(proposal.correlation.trace_id),
            sequence=int(proposal.correlation.sequence),
            source=BrainAudienceSource.COMMENT,
            received_at_ms=self.clock(),
            text=proposal.text,
        )
        decision = pipeline.submit(audience_input)
        return audience_input if decision is GateDecision.ACCEPT else None

    def _run_agent_plan(
        self,
        audience_input: BrainAudienceInput,
        turn_id: TurnId,
        correlation: EventCorrelation,
    ) -> None:
        pipeline = self.agent_pipeline
        if pipeline is None:
            return
        composition = self.interaction_ingress.data.context.compose(
            _BRAIN_CONTEXT_MODEL, _BRAIN_CONTEXT_POLICY
        )
        context = composition.snapshot
        memory = self.interaction_ingress.data.memory.snapshot
        retrieval = self.interaction_ingress.data.retrieval.snapshot
        presentation = self.interaction_ingress.reducer.presentation_state
        ppt_deck_id = (
            presentation[0] if presentation is not None else self._planned_ppt_deck_id
        )
        ppt_page = (
            presentation[2] if presentation is not None else self._planned_ppt_page
        )
        corpus_version = f"{retrieval.corpus_id}@{retrieval.corpus_revision}"
        index_version = f"{retrieval.index_id}@{retrieval.index_revision}"
        knowledge_reference = (
            f"本地知识库: corpus={corpus_version}, index={index_version}"
        )
        snapshot = BrainStateSnapshot(
            session_id=str(self.scheduler.snapshot.session_id),
            turn_id=str(turn_id),
            revision=int(self.scheduler.snapshot.revision),
            cancellation_epoch=int(self.cancellation_epoch),
            input=audience_input,
            context_summary=context.summary,
            recent_context=tuple(entry.text for entry in composition.entries),
            memory_markdown=render_markdown_memory(
                memory, self.scheduler.snapshot.session_id
            ),
            capabilities=self.agent_capabilities,
            frontend_caption=self._frontend_caption,
            frontend_animation=self._frontend_animation,
            ppt_deck_id=ppt_deck_id,
            ppt_page=ppt_page,
            tasks=tuple(
                TaskSnapshot(
                    task_id=str(record.request.task_id),
                    kind=record.request.kind.value,
                    lane=record.request.kind.value,
                    status=record.state.value,
                    deadline_ms=int(record.request.deadline_ms),
                    owner_turn_id=str(record.request.turn_id),
                    cancellation_reason=record.cancellation_reason,
                )
                for record in self.task_registry.records
            ),
            context_revision=int(context.generation),
            memory_revision=int(memory.revision),
            context_budget=512,
            compaction_required=bool(composition.digests),
            knowledge_references=(knowledge_reference,),
            mcp_allowlist=self.agent_mcp_allowlist,
        )
        result = pipeline.run(snapshot)
        self._agent_results.append(result)
        if isinstance(result, PlanAccepted):
            self._apply_agent_plan(
                result, audience_input, turn_id, correlation, composition
            )
        else:
            _LOGGER.debug(
                "agent_plan_rejected turn=%s reason=%s", turn_id, result.reason
            )

    async def _run_async_agent_plan(
        self,
        pipeline: AsyncAgentPipeline,
        audience_input: BrainAudienceInput,
        turn_id: TurnId,
        correlation: EventCorrelation,
    ) -> None:
        composition = self.interaction_ingress.data.context.compose(
            _BRAIN_CONTEXT_MODEL, _BRAIN_CONTEXT_POLICY
        )
        context = composition.snapshot
        memory = self.interaction_ingress.data.memory.snapshot
        retrieval = self.interaction_ingress.data.retrieval.snapshot
        presentation = self.interaction_ingress.reducer.presentation_state
        ppt_deck_id = (
            presentation[0] if presentation is not None else self._planned_ppt_deck_id
        )
        ppt_page = (
            presentation[2] if presentation is not None else self._planned_ppt_page
        )
        corpus_version = f"{retrieval.corpus_id}@{retrieval.corpus_revision}"
        index_version = f"{retrieval.index_id}@{retrieval.index_revision}"
        snapshot = BrainStateSnapshot(
            session_id=str(self.scheduler.snapshot.session_id),
            turn_id=str(turn_id),
            revision=int(self.scheduler.snapshot.revision),
            cancellation_epoch=int(self.cancellation_epoch),
            input=audience_input,
            context_summary=context.summary,
            recent_context=tuple(entry.text for entry in composition.entries),
            memory_markdown=render_markdown_memory(
                memory, self.scheduler.snapshot.session_id
            ),
            capabilities=self.agent_capabilities,
            frontend_caption=self._frontend_caption,
            frontend_animation=self._frontend_animation,
            ppt_deck_id=ppt_deck_id,
            ppt_page=ppt_page,
            tasks=tuple(
                TaskSnapshot(
                    task_id=str(record.request.task_id),
                    kind=record.request.kind.value,
                    lane=record.request.kind.value,
                    status=record.state.value,
                    deadline_ms=int(record.request.deadline_ms),
                    owner_turn_id=str(record.request.turn_id),
                    cancellation_reason=record.cancellation_reason,
                )
                for record in self.task_registry.records
            ),
            context_revision=int(context.generation),
            memory_revision=int(memory.revision),
            context_budget=512,
            compaction_required=bool(composition.digests),
            knowledge_references=(
                f"本地知识库: corpus={corpus_version}, index={index_version}",
            ),
            mcp_allowlist=self.agent_mcp_allowlist,
        )
        result = await pipeline.run(snapshot)
        self._agent_results.append(result)
        if isinstance(result, PlanAccepted):
            self._apply_agent_plan(
                result, audience_input, turn_id, correlation, composition
            )
        else:
            _LOGGER.debug(
                "agent_plan_rejected turn=%s reason=%s", turn_id, result.reason
            )

    def _apply_agent_plan(
        self,
        accepted: PlanAccepted,
        audience_input: BrainAudienceInput,
        turn_id: TurnId,
        correlation: EventCorrelation,
        composition: ContextComposition,
    ) -> None:
        """Commit only reducer-accepted Brain state operations.

        Media/frontend effects remain with their dedicated adapters; this
        reducer path owns context and scheduler tasks so late provider output
        cannot write either one directly.
        """
        provenance = ContextProvenance(
            session_id=SessionId(audience_input.session_id),
            turn_id=turn_id,
            segment_id=SegmentId(f"agent-{turn_id}"),
            sequence=ContextSequence(audience_input.sequence),
            source_id=ContextSourceId(audience_input.trace_id),
        )
        self._apply_context_compaction(accepted, composition)
        self.interaction_ingress.data.consider_context(
            FinalizedInput(provenance, audience_input.text)
        )
        for index, observation in enumerate(accepted.observations):
            if observation.request.kind != "knowledge":
                continue
            self.interaction_ingress.data.consider_context(
                ToolObservation(
                    ContextProvenance(
                        session_id=provenance.session_id,
                        turn_id=provenance.turn_id,
                        segment_id=provenance.segment_id,
                        sequence=ContextSequence(audience_input.sequence),
                        source_id=ContextSourceId(
                            f"{audience_input.trace_id}:tool:{index}"
                        ),
                    ),
                    observation.text,
                )
            )
        if accepted.plan.response_text != "":
            self.interaction_ingress.data.consider_context(
                AcceptedOutput(provenance, accepted.plan.response_text)
            )
        self._apply_memory_patch(accepted, correlation, turn_id)
        for index, operation in enumerate(accepted.plan.state_operations):
            if operation.kind == "create_task":
                self._create_brain_task(
                    operation.payload,
                    accepted.plan.response_text,
                    turn_id,
                    correlation,
                    index,
                )
            elif operation.kind == "cancel_task":
                task_id = operation.payload.get("task_id")
                if isinstance(task_id, str):
                    _ = self.cancel_task(TaskId(task_id), correlation)
        self._dispatch_agent_effects(accepted, turn_id)

    def _dispatch_agent_effects(self, accepted: PlanAccepted, turn_id: TurnId) -> None:
        self._materialize_frontend_operations(accepted.plan.frontend_operations)
        dispatcher = self.agent_effect_dispatcher
        if dispatcher is None:
            return
        session_id = self.scheduler.snapshot.session_id
        for operation in accepted.plan.media_operations:
            dispatcher.dispatch_media(
                operation,
                session_id=session_id,
                turn_id=turn_id,
                cancellation_epoch=self.cancellation_epoch,
            )
        for operation in accepted.plan.frontend_operations:
            dispatcher.dispatch_frontend(
                operation,
                session_id=session_id,
                turn_id=turn_id,
            )

    def _materialize_frontend_operations(
        self, operations: tuple[FrontendOperation, ...]
    ) -> None:
        for operation in operations:
            if operation.kind == "caption" and isinstance(operation.value, str):
                self._frontend_caption = operation.value
            elif operation.kind == "animation" and isinstance(operation.value, str):
                self._frontend_animation = operation.value
            elif operation.kind == "ppt.load" and isinstance(operation.value, str):
                self._planned_ppt_deck_id = operation.value
                self._planned_ppt_page = 1
            elif operation.kind == "ppt.navigate" and isinstance(operation.value, int):
                self._planned_ppt_page = operation.value

    def _apply_context_compaction(
        self, accepted: PlanAccepted, composition: ContextComposition
    ) -> None:
        for operation in accepted.plan.state_operations:
            if operation.kind != "context.compact":
                continue
            summary = operation.payload.get("summary")
            if not isinstance(summary, str):
                continue
            try:
                _ = self.interaction_ingress.data.context.compact(
                    composition, summary=summary
                )
            except ContextCompactionError:
                return

    def _apply_memory_patch(
        self,
        accepted: PlanAccepted,
        correlation: EventCorrelation,
        turn_id: TurnId,
    ) -> None:
        if not accepted.plan.memory_patches:
            return
        patch = accepted.plan.memory_patches[0]
        key = patch.get("id")
        operation = patch.get("op")
        if not isinstance(key, str) or not isinstance(operation, str):
            return
        if operation == "delete":
            self.interaction_ingress.data.delete_memory(MemoryKey(key))
            return
        value = patch.get("value")
        confidence = patch.get("confidence")
        base_revision = patch.get("base_revision")
        if (
            operation not in {"add", "update"}
            or patch.get("category") != "preference"
            or not isinstance(value, str)
            or type(confidence) is not int
            or type(base_revision) is not int
        ):
            return
        _ = self.interaction_ingress.data.reduce_memory(
            MemoryProposal(
                key=MemoryKey(key),
                value=value,
                category=MemoryCategory.ORDINARY_PREFERENCE,
                confidence=MemoryConfidence(confidence),
                base_revision=ProposalRevision(base_revision),
                provenance=MemoryProvenance(
                    source=MemorySource.AGENT_PROPOSAL,
                    trace_id=correlation.trace_id,
                    session_id=correlation.session_id,
                    turn_id=turn_id,
                    evidence_id=f"brain:{correlation.trace_id}:{correlation.sequence}",
                ),
            )
        )

    def _create_brain_task(
        self,
        payload: dict[str, object],
        response_text: str,
        turn_id: TurnId,
        correlation: EventCorrelation,
        index: int,
    ) -> None:
        kind = _brain_task_kind(payload.get("task_kind"))
        if kind is None:
            return
        deadline_ms = payload.get("deadline_ms")
        deadline = (
            self.clock() + 30_000
            if type(deadline_ms) is not int or deadline_ms <= self.clock()
            else deadline_ms
        )
        task_id = TaskId(f"brain-{turn_id}-{index}")
        outcome = self.schedule_task(
            TaskRequest(
                task_id=task_id,
                session_id=self.scheduler.snapshot.session_id,
                turn_id=turn_id,
                parent_task_id=None,
                deadline_ms=TaskDeadlineMs(deadline),
                snapshot_revision=self.scheduler.snapshot.revision,
                idempotency_key=IdempotencyKey(str(task_id)),
                kind=kind,
            ),
            correlation,
        )
        if outcome.accepted and payload.get("task_kind") == "tts":
            self._agent_tts_text[task_id] = response_text

    async def run_agent_tts_for_turn(
        self,
        turn_id: TurnId,
        synthesize: Callable[[str], Awaitable[bool]],
        correlation: EventCorrelation,
    ) -> bool:
        """Run one reducer-approved TTS task and commit only successful output.

        The caller owns the transport adapter; this runtime retains task lifecycle
        validation so an unaccepted or stale plan can never reach TTS.
        """
        for task_id, text in tuple(self._agent_tts_text.items()):
            record = self.task_registry.task(task_id)
            if (
                record is None
                or record.request.turn_id != turn_id
                or record.state is not TaskState.PENDING
            ):
                continue
            if not text.strip():
                _ = self.cancel_task(task_id, correlation)
                _ = self._agent_tts_text.pop(task_id, None)
                return False
            try:
                emitted = await synthesize(text)
            except (OSError, ValueError):
                emitted = False
                _LOGGER.exception("agent_tts_execution_failed task=%s", task_id)
            if not emitted:
                _ = self.cancel_task(task_id, correlation)
                _ = self._agent_tts_text.pop(task_id, None)
                return False
            committed = self.reduce_task(
                TaskResult(
                    task_id,
                    record.request.session_id,
                    record.request.turn_id,
                    record.request.snapshot_revision,
                    TaskEffect("tts.emitted", text[:240]),
                ),
                correlation,
            ).accepted
            _ = self._agent_tts_text.pop(task_id, None)
            return committed
        return False

    def receive_control(self, raw_message: str) -> bool:
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

    def receive_session_control(  # noqa: PLR0911
        self, control: SessionControl
    ) -> RuntimeOutcome:
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

            case ContextResetControl(correlation=correlation):
                self.interaction_ingress.data.reset_context()
                return self._interaction_outcome(
                    correlation, "context_reset", accepted=True, task_id=None
                )

            case MemoryDeleteControl(key=key, correlation=correlation):
                self.interaction_ingress.data.delete_memory(key)
                return self._interaction_outcome(
                    correlation, "memory_deleted", accepted=True, task_id=None
                )

            case SessionEndControl(correlation=correlation):
                return self.end_session(correlation)

    async def receive_session_control_async(
        self, control: SessionControl
    ) -> RuntimeOutcome:
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

        plan = build_presentation_mcp_plan(
            PresentationMcpPlanInput(
                proposal=proposal,
                snapshot=self.scheduler.snapshot,
                turn_id=turn_id,
                data_snapshot=self._task_data_snapshot,
                deadline_ms=self.clock() + 5_000,
            )
        )

        scheduled = self.schedule_deck_task(plan.intent, plan.request, correlation)

        if not scheduled.accepted:
            self.interaction_ingress.reducer.cancel_presentation(proposal.command_id)

            return scheduled

        _ = await self.run_deck_worker_async(
            now_ms=self.clock(), correlation=correlation
        )

        return scheduled

    def receive_asr_final(
        self,
        event: ASRAudienceEvent,
        correlation: EventCorrelation,
        gate: AsrSemanticGate | None = None,
    ) -> RuntimeOutcome:
        if correlation in self._correlations:
            return self._reject(correlation, "duplicate_correlation")
        if self._ended:
            return self._reject(correlation, "session_ended")

        audience_input = BrainAudienceInput(
            session_id=str(correlation.session_id),
            trace_id=str(correlation.trace_id),
            sequence=int(correlation.sequence),
            source=BrainAudienceSource.ASR,
            received_at_ms=event.received_at_ms,
            text=event.text,
        )
        pipeline = self.agent_pipeline
        if (
            pipeline is not None
            and pipeline.submit(audience_input) is GateDecision.DISCARD
        ):
            return self._reject(correlation, "agent_gate_discarded")

        if (
            pipeline is None
            and gate is not None
            and gate.evaluate(event.text) is AsrGateDecision.DISCARD
        ):
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

                self._run_agent_plan(
                    audience_input, accepted_event.turn_id, correlation
                )

                return RuntimeOutcome(
                    accepted=True,
                    correlation=correlation,
                    turn_id=accepted_event.turn_id,
                )

            case _:
                return self._reject(correlation, "scheduler_rejected")

    async def receive_asr_final_async(
        self,
        event: ASRAudienceEvent,
        correlation: EventCorrelation,
    ) -> RuntimeOutcome:
        """Run finalized ASR through the same async Gate and Brain as comments."""
        pipeline = self.async_agent_pipeline
        if pipeline is None:
            return self.receive_asr_final(event, correlation)
        if correlation in self._correlations:
            return self._reject(correlation, "duplicate_correlation")
        if self._ended:
            return self._reject(correlation, "session_ended")
        audience_input = BrainAudienceInput(
            session_id=str(correlation.session_id),
            trace_id=str(correlation.trace_id),
            sequence=int(correlation.sequence),
            source=BrainAudienceSource.ASR,
            received_at_ms=event.received_at_ms,
            text=event.text,
        )
        if await pipeline.submit(audience_input) is GateDecision.DISCARD:
            return self._reject(correlation, "agent_gate_discarded")
        transition = self.scheduler.apply(
            StartTurn(
                expected_revision=self.scheduler.snapshot.revision,
                event=SchedulerEvent(event_type="asr.final", correlation=correlation),
            )
        )
        if not isinstance(transition, TransitionAccepted):
            return self._reject(correlation, "scheduler_rejected")
        accepted_turn = transition.accepted_event.turn_id
        self._correlations.add(correlation)
        self._journal.dispatches.append(RuntimeDispatch(correlation, accepted_turn))
        await self._run_async_agent_plan(
            pipeline, audience_input, accepted_turn, correlation
        )
        return RuntimeOutcome(
            accepted=True, correlation=correlation, turn_id=accepted_turn
        )

    def enroll_profile(
        self, enrollment: ProfileEnrollment, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        _ = self.interaction_ingress.data.enroll_profile(enrollment)

        return self._interaction_outcome(
            correlation, "profile_enrolled", accepted=True, task_id=None
        )

    def end_session(self, correlation: EventCorrelation) -> RuntimeOutcome:
        """Cancel session work and erase session-owned durable and transient state."""
        if self._ended:
            return self._reject(correlation, "session_ended")
        _ = self.task_registry.cancel_pending(reason="session_ended")
        for command_id in tuple(self._active_deck_tasks.values()):
            _ = self.deck_dispatcher.cancel(command_id)
        self._deck_intents.clear()
        self._active_deck_tasks.clear()
        self._voice_evidence_ranges.clear()
        self.interaction_ingress.data.clear_session()
        self.interaction_ingress.clear_session_data()
        self._ended = True
        return self._interaction_outcome(
            correlation, "session_ended", accepted=True, task_id=None
        )

    def revoke_profile_consent(
        self, profile_id: VoiceProfileId, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        self.interaction_ingress.data.revoke_profile_consent(profile_id)

        return self._interaction_outcome(
            correlation, "profile_revoked", accepted=True, task_id=None
        )

    def receive_action(
        self, proposal: ActionProposal, correlation: EventCorrelation
    ) -> RuntimeOutcome:
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

    def schedule_deck_task(
        self,
        intent: DeckDispatchIntent,
        request: TaskRequest,
        correlation: EventCorrelation,
    ) -> RuntimeOutcome:
        outcome = self.schedule_task(request, correlation)

        if outcome.accepted:
            self._deck_intents[request.task_id] = intent

        self.operational_journal.append(
            interaction_record(
                correlation,
                self.scheduler.snapshot,
                "deck_task",
                outcome.accepted,
                request.task_id,
            )
        )

        return outcome

    def run_deck_worker(
        self, *, now_ms: int, correlation: EventCorrelation
    ) -> DeckDispatchOutcome:
        request = self.next_task(now_ms=now_ms)

        if request is None:
            self._discard_terminal_deck_intents()

            return DeckDispatchOutcome(accepted=False)

        intent = self._deck_intents.pop(request.task_id, None)

        if intent is None:
            return DeckDispatchOutcome(accepted=False)

        outcome = self.deck_dispatcher.dispatch(intent, now_ms=now_ms)

        if not outcome.accepted:
            self._deck_intents[request.task_id] = intent

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

    async def run_deck_worker_async(
        self, *, now_ms: int, correlation: EventCorrelation
    ) -> DeckDispatchOutcome:
        request = self.next_task(now_ms=now_ms)

        if request is None:
            self._discard_terminal_deck_intents()

            return DeckDispatchOutcome(accepted=False)

        intent = self._deck_intents.pop(request.task_id, None)

        if intent is None:
            return DeckDispatchOutcome(accepted=False)

        self._active_deck_tasks[request.task_id] = intent.command.command_id

        try:
            outcome = await self.deck_dispatcher.dispatch_async(intent, now_ms=now_ms)

        finally:
            _ = self._active_deck_tasks.pop(request.task_id, None)

        self.operational_journal.append(
            interaction_record(
                correlation,
                self.scheduler.snapshot,
                "deck_dispatch",
                outcome.accepted,
                request.task_id,
            )
        )

        if not outcome.accepted:
            self._deck_intents[request.task_id] = intent

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

    def reconcile_deck_worker(
        self,
        task_id: TaskId,
        *,
        now_ms: int,
        correlation: EventCorrelation,
    ) -> DeckDispatchOutcome:
        intent = self._deck_intents.get(task_id)

        if intent is None:
            return DeckDispatchOutcome(accepted=False)

        if now_ms > intent.deadline_ms:
            _ = self.task_registry.timeout(task_id)

            _ = self._deck_intents.pop(task_id)

            self._cancel_pending_deck_presentation(intent.command.command_id)

            _ = self.deck_dispatcher.reconcile(intent.command.command_id, now_ms=now_ms)

            return DeckDispatchOutcome(accepted=False)

        outcome = self.deck_dispatcher.reconcile(
            intent.command.command_id, now_ms=now_ms
        )

        if not outcome.accepted:
            return outcome

        request = self.task_registry.task(task_id)

        if request is None:
            return DeckDispatchOutcome(accepted=False)

        _ = self._deck_intents.pop(task_id)

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
        cancelled = self.task_registry.cancel(task_id, reason="cancelled")

        if cancelled is not None:
            intent = self._deck_intents.pop(task_id, None)

            if intent is not None:
                _ = self.deck_dispatcher.cancel(intent.command.command_id)

                self._cancel_pending_deck_presentation(intent.command.command_id)

            active_command = self._active_deck_tasks.get(task_id)

            if active_command is not None:
                _ = self.deck_dispatcher.cancel(active_command)

                self._cancel_pending_deck_presentation(active_command)

        return self._interaction_outcome(
            correlation,
            "task_cancelled",
            cancelled is not None,
            task_id,
        )

    def _discard_terminal_deck_intents(self) -> None:
        for task_id in tuple(self._deck_intents):
            record = self.task_registry.task(task_id)

            if record is None or record.state is not TaskState.PENDING:
                intent = self._deck_intents.pop(task_id)

                self._cancel_pending_deck_presentation(intent.command.command_id)

    def _cancel_pending_deck_presentation(self, command_id: CommandId) -> None:
        self.interaction_ingress.reducer.cancel_presentation(command_id)
        _ = self._presentation_correlations.pop(command_id, None)

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
        return self.executor.next(now_ms=now_ms)

    def reduce_task(
        self, result: TaskResult, correlation: EventCorrelation
    ) -> RuntimeOutcome:
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
        self._journal.rejections.append(RuntimeRejection(correlation, reason))

        return RuntimeOutcome(accepted=False, correlation=correlation)

    def _interaction_outcome(
        self,
        correlation: EventCorrelation,
        stage: str,
        accepted: bool,
        task_id: TaskId | None,
    ) -> RuntimeOutcome:
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
