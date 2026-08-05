import asyncio
import hashlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field, replace
from time import monotonic_ns
from typing import Protocol

from cryptography.exceptions import InvalidTag

from orchestrator.agent_state import AgentState, TurnCoordinator, TurnPhase
from orchestrator.brain_contracts import (
    AudienceInput as BrainAudienceInput,
)
from orchestrator.brain_contracts import (
    AudienceSource as BrainAudienceSource,
)
from orchestrator.brain_contracts import (
    BrainStateSnapshot,
    GateDecision,
    TaskSnapshot,
)
from orchestrator.caption_timeline import CaptionTimelineCommand
from orchestrator.context_compactor import AsyncContextCompactor
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
from orchestrator.execution_envelope import ExecutionEnvelope
from orchestrator.identity import (
    ProfileEnrollment,
    ProfileRecognition,
    ProfileRecognitionKnown,
    RecognitionConfidence,
    VoiceProfileId,
)
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
    TestLocalDeckEffectExecutor,
)
from orchestrator.memory import (
    MemoryCategory,
    MemoryProposal,
    MemoryProvenance,
    MemorySource,
    ProposalRevision,
)
from orchestrator.memory_extractor import (
    AsyncMemoryCandidateExtractor,
    parse_memory_candidate,
)
from orchestrator.memory_store import render_markdown_memory
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.operational_journal import OperationalJournal, OperationalRecord
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.response_contracts import ResponseProposal, parse_inline_cues
from orchestrator.response_coordinator import (
    AsyncResponseCoordinator,
    CoordinatedResponse,
)
from orchestrator.response_execution_mode import ResponseExecutionMode
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
    TaskRecord,
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
from orchestrator.voice_templates import (
    DecryptedVoiceTemplate,
    VoiceTemplateError,
    VoiceTemplateProtector,
    match_voice,
    modular_intervals_overlap,
)

_LOGGER = logging.getLogger(__name__)

type AgentTtsSynthesize = Callable[[str, Callable[[], bool]], Awaitable[bool]]


class _ResponseProviderCancelledError(Exception):
    """A fenced response task had its owned provider coroutine cancelled."""


class _TtsProviderCancelledError(Exception):
    """A pre-output TTS provider coroutine was fenced and cancelled."""

def _monotonic_ms() -> int:
    return monotonic_ns() // 1_000_000


def _bounded_observation_summary(observation: str) -> str:
    normalized = " ".join(observation.split())
    digest = hashlib.sha256(observation.encode("utf-8")).hexdigest()
    return f"status=success digest=sha256:{digest} text={normalized[:512]}"


class AsyncAudienceGate(Protocol):
    async def evaluate(
        self,
        audience_input: BrainAudienceInput,
        *,
        active_summary: str,
        recent_turn_context: tuple[str, ...] = (),
    ) -> GateDecision: ...


_BRAIN_CONTEXT_MODEL = ModelId("agent-brain")

_BRAIN_CONTEXT_POLICY = StaticContextBudgetPolicy(
    model_id=_BRAIN_CONTEXT_MODEL,
    budget=ModelContextBudget(input_tokens=TokenBudget(512)),
)

_VOICE_EVIDENCE_CAPACITY = 128
_VOICE_EVIDENCE_MINIMUM_QUALITY = 0.5
_VOICE_EVIDENCE_MINIMUM_SPEECH_MS = 500
_AUDIENCE_QUEUE_CAPACITY = 16

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


@dataclass(frozen=True, slots=True)
class _PendingResponseCommit:
    provenance: ContextProvenance
    input_text: str
    spoken_text: str
    marked_text: str
    observation: str | None


@dataclass(frozen=True, slots=True)
class _AgentTtsExecution:
    task_id: TaskId
    text: str
    record: TaskRecord
    correlation: EventCorrelation
    output_started: Callable[[], None] | None


@dataclass(frozen=True, slots=True)
class _PendingAudienceAdmission:
    source: BrainAudienceSource
    correlation: EventCorrelation
    run: Callable[[], Coroutine[None, None, RuntimeOutcome]]
    result: asyncio.Future[RuntimeOutcome]
    admission_valid: Callable[[], bool]


@dataclass(frozen=True, slots=True)
class _SpeakerIdentity:
    profile_id: VoiceProfileId
    preferred_name: str
    confidence: float


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

    # This is the sole logical-state owner for the minimal response path.
    # SessionScheduler remains authoritative for ids/revisions and
    # SchedulerOutputFence remains authoritative for Sound lease cutover.
    turn_coordinator: TurnCoordinator = field(default_factory=TurnCoordinator)

    async_agent_gate: AsyncAudienceGate | None = None

    async_response_coordinator: AsyncResponseCoordinator | None = None

    response_execution_mode: ResponseExecutionMode = ResponseExecutionMode.NEW_EXECUTE

    memory_candidate_extractor: AsyncMemoryCandidateExtractor | None = None

    context_compactor: AsyncContextCompactor | None = None

    agent_capabilities: frozenset[str] = frozenset()

    agent_mcp_allowlist: frozenset[str] = frozenset()

    # Transport owns raw provider coroutines; the scheduler tells it only when
    # a TTS task is still pre-output and therefore safe to stop.
    preoutput_tts_cancellation: Callable[[TurnId], None] | None = None

    response_task_timeout_ms: int = 30_000

    clock: Callable[[], int] = _monotonic_ms

    cancellation_epoch: CancellationEpoch = field(
        default_factory=lambda: CancellationEpoch(0)
    )

    _correlations: set[EventCorrelation] = field(default_factory=set)

    _pending_correlations: set[EventCorrelation] = field(default_factory=set)

    _voice_admissions: deque[_PendingAudienceAdmission] = field(
        default_factory=deque
    )

    _comment_admissions: deque[_PendingAudienceAdmission] = field(
        default_factory=deque
    )

    _admission_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    _admission_worker: asyncio.Task[None] | None = None

    _deck_intents: dict[TaskId, DeckDispatchIntent] = field(default_factory=dict)

    _active_deck_tasks: dict[TaskId, CommandId] = field(default_factory=dict)

    _presentation_correlations: dict[CommandId, EventCorrelation] = field(
        default_factory=dict
    )

    _journal: _RuntimeJournal = field(default_factory=_RuntimeJournal)

    _agent_tts_text: dict[TaskId, str] = field(default_factory=dict)

    _pending_response_commits: dict[TaskId, _PendingResponseCommit] = field(
        default_factory=dict
    )

    _started_timeline_text: dict[TurnId, tuple[str, SegmentId]] = field(
        default_factory=dict
    )

    _maintenance_tasks: set[asyncio.Task[None]] = field(default_factory=set)

    _active_response_provider_tasks: dict[TaskId, asyncio.Task[object]] = field(
        default_factory=dict
    )

    _active_preoutput_tts_provider_tasks: dict[TaskId, asyncio.Task[bool]] = field(
        default_factory=dict
    )

    _voice_evidence_ranges: list[tuple[str, int, int]] = field(default_factory=list)

    _voice_evidence_cache: dict[str, tuple[VoiceEvidence, int]] = field(
        default_factory=dict
    )

    _voice_template_protector: VoiceTemplateProtector | None = None

    _voice_evidence_ttl_ms: int = 120_000

    _voice_match_threshold: float = 0.90

    _voice_ambiguity_margin: float = 0.05

    _speaker_identities: dict[EventCorrelation, _SpeakerIdentity] = field(
        default_factory=dict
    )

    _re_enrollment_required: set[VoiceProfileId] = field(default_factory=set)

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
        async_agent_gate: AsyncAudienceGate | None = None,
        async_response_coordinator: AsyncResponseCoordinator | None = None,
        response_execution_mode: ResponseExecutionMode = (
            ResponseExecutionMode.NEW_EXECUTE
        ),
        memory_candidate_extractor: AsyncMemoryCandidateExtractor | None = None,
        context_compactor: AsyncContextCompactor | None = None,
        agent_capabilities: frozenset[str] | None = None,
        agent_mcp_allowlist: frozenset[str] | None = None,
        response_task_timeout_ms: int = 30_000,
    ) -> "SessionRuntime":
        if response_task_timeout_ms <= 0:
            field_name = "response_task_timeout_ms"
            raise ValueError(field_name)
        scheduler = SessionScheduler(
            session_id=session_id,
            turn_id_prefix=turn_id_prefix,
        )

        interaction_ingress = SessionInteractionIngress.create(scheduler)

        task_registry = TaskRegistry(
            session_id=session_id,
            config=task_config,
            clock=clock,
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
                interaction_ingress.reducer, TestLocalDeckEffectExecutor()
            ),
            mode_policy=AdaptiveAgentPolicy(),
            clock=clock,
            async_agent_gate=async_agent_gate,
            async_response_coordinator=async_response_coordinator,
            response_execution_mode=response_execution_mode,
            memory_candidate_extractor=memory_candidate_extractor,
            context_compactor=context_compactor,
            agent_capabilities=(
                _DEFAULT_AGENT_CAPABILITIES
                if agent_capabilities is None
                else agent_capabilities
            ),
            agent_mcp_allowlist=(
                frozenset() if agent_mcp_allowlist is None else agent_mcp_allowlist
            ),
            response_task_timeout_ms=response_task_timeout_ms,
        )

    @property
    def voice_evidence_ranges(self) -> tuple[tuple[str, int, int], ...]:
        """Non-sensitive evidence correlation only; embeddings never persist."""
        return tuple(self._voice_evidence_ranges)

    def receive_voice_evidence(self, evidence: VoiceEvidence) -> bool:
        if evidence.session_id != str(self.scheduler.snapshot.session_id):
            return False
        now_ms = self.clock()
        self._expire_voice_evidence(now_ms)
        if evidence.evidence_id in self._voice_evidence_cache:
            return False
        if len(self._voice_evidence_cache) >= _VOICE_EVIDENCE_CAPACITY:
            oldest = min(
                self._voice_evidence_cache,
                key=lambda evidence_id: self._voice_evidence_cache[evidence_id][1],
            )
            _ = self._voice_evidence_cache.pop(oldest, None)
        self._voice_evidence_cache[evidence.evidence_id] = (evidence, now_ms)
        self._voice_evidence_ranges.append(
            (
                evidence.stream_id,
                evidence.rtp_start_timestamp,
                evidence.rtp_end_timestamp,
            )
        )
        return True

    def configure_voice_identity(
        self,
        key: bytes | None,
        *,
        evidence_ttl_seconds: int = 120,
        match_threshold: float = 0.90,
        ambiguity_margin: float = 0.05,
    ) -> None:
        self._voice_template_protector = (
            None if key is None else VoiceTemplateProtector(key)
        )
        self._voice_evidence_ttl_ms = evidence_ttl_seconds * 1_000
        self._voice_match_threshold = match_threshold
        self._voice_ambiguity_margin = ambiguity_margin
        if self._voice_template_protector is not None:
            self._mark_unreadable_voice_templates(self._voice_template_protector)

    def _mark_unreadable_voice_templates(
        self, protector: VoiceTemplateProtector
    ) -> None:
        profiles = self.interaction_ingress.data.profiles
        session_id = str(self.scheduler.snapshot.session_id)
        for profile_id in profiles.profile_ids:
            encrypted = profiles.encrypted_template(profile_id)
            if encrypted is None:
                continue
            try:
                _ = protector.decrypt(
                    session_id=session_id,
                    profile_id=profile_id,
                    template=encrypted,
                )
            except (InvalidTag, VoiceTemplateError, ValueError):
                self._re_enrollment_required.add(profile_id)
                try:
                    profiles.mark_re_enrollment_required(profile_id)
                except OSError:
                    _LOGGER.exception(
                        "voice_profile_status_write_failed session=%s profile=%s",
                        session_id,
                        profile_id,
                    )

    @property
    def re_enrollment_required(self) -> frozenset[VoiceProfileId]:
        return frozenset(
            self._re_enrollment_required
            | set(self.interaction_ingress.data.profiles.re_enrollment_required_ids)
        )

    def _recognize_voice(
        self, event: ASRAudienceEvent, correlation: EventCorrelation
    ) -> None:
        protector = self._voice_template_protector
        if (
            protector is None
            or event.stream_id is None
            or event.input_epoch is None
            or event.rtp_start_timestamp is None
            or event.rtp_end_timestamp is None
        ):
            return
        now_ms = self.clock()
        self._expire_voice_evidence(now_ms)
        candidates = tuple(
            (evidence, created_ms)
            for evidence, created_ms in self._voice_evidence_cache.values()
            if evidence.stream_id == event.stream_id
            and evidence.input_epoch == event.input_epoch
            and modular_intervals_overlap(
                evidence.rtp_start_timestamp,
                evidence.rtp_end_timestamp,
                event.rtp_start_timestamp,
                event.rtp_end_timestamp,
            )
        )
        if not candidates:
            return
        evidence = max(candidates, key=lambda item: item[1])[0]
        profiles = self.interaction_ingress.data.profiles
        templates: dict[VoiceProfileId, DecryptedVoiceTemplate] = {}
        for profile_id in profiles.matchable_profile_ids(now_ms=now_ms):
            encrypted = profiles.encrypted_template(profile_id)
            if encrypted is None:
                continue
            try:
                templates[profile_id] = protector.decrypt(
                    session_id=str(correlation.session_id),
                    profile_id=profile_id,
                    template=encrypted,
                )
            except (InvalidTag, VoiceTemplateError, ValueError):
                self._re_enrollment_required.add(profile_id)
                try:
                    profiles.mark_re_enrollment_required(profile_id)
                except OSError:
                    _LOGGER.exception(
                        "voice_profile_status_write_failed session=%s profile=%s",
                        correlation.session_id,
                        profile_id,
                    )
        matched = match_voice(
            evidence.embedding_model_revision,
            evidence.embedding,
            templates,
            threshold=self._voice_match_threshold,
            ambiguity_margin=self._voice_ambiguity_margin,
        )
        recognized = profiles.recognize(
            ProfileRecognition(
                matched.profile_id,
                RecognitionConfidence(round(matched.confidence * 100)),
            ),
            now_ms=now_ms,
        )
        if isinstance(recognized, ProfileRecognitionKnown):
            self._speaker_identities[correlation] = _SpeakerIdentity(
                recognized.profile_id,
                recognized.preferred_name,
                matched.confidence,
            )

    def enroll_profile_from_evidence(
        self, control: ProfileEnrollmentControl
    ) -> RuntimeOutcome:
        correlation = control.correlation
        self._expire_voice_evidence(self.clock())
        cached = self._voice_evidence_cache.pop(control.evidence_id, None)
        protector = self._voice_template_protector
        if cached is None or protector is None or not control.consented:
            return self._reject(correlation, "voice_evidence_unavailable")
        evidence = cached[0]
        if (
            evidence.quality_score < _VOICE_EVIDENCE_MINIMUM_QUALITY
            or evidence.speech_ms < _VOICE_EVIDENCE_MINIMUM_SPEECH_MS
        ):
            return self._reject(correlation, "voice_evidence_quality")
        encrypted = protector.encrypt(
            session_id=evidence.session_id,
            profile_id=control.profile_id,
            model_revision=evidence.embedding_model_revision,
            embedding=evidence.embedding,
        )
        return self.enroll_profile(
            ProfileEnrollment(
                profile_id=control.profile_id,
                preferred_name=control.preferred_name,
                encrypted_template=encrypted,
                consented=True,
            ),
            correlation,
        )

    def _expire_voice_evidence(self, now_ms: int) -> None:
        for evidence_id, (_, created_ms) in tuple(
            self._voice_evidence_cache.items()
        ):
            if now_ms - created_ms >= self._voice_evidence_ttl_ms:
                _ = self._voice_evidence_cache.pop(evidence_id, None)

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

    def _advance_turn_epoch(self) -> None:
        """Fence and cancel unfinished work before a newer turn can run."""
        state = self.turn_coordinator.state
        if (
            state.turn_id is not None
            and state.epoch == int(self.cancellation_epoch)
            and state.phase
            not in {TurnPhase.IDLE, TurnPhase.PLAYING, TurnPhase.COMPLETED}
        ):
            _ = self.turn_coordinator.cancel(
                turn_id=state.turn_id, epoch=state.epoch
            )
        self.cancellation_epoch = CancellationEpoch(int(self.cancellation_epoch) + 1)
        cancelled = self.task_registry.cancel_pending(reason="superseded_turn")
        self._cancel_preoutput_tts(cancelled)
        self._cancel_active_response_providers(cancelled)
        self._cancel_active_preoutput_tts_providers(cancelled)

    @property
    def response_turn_state(self) -> AgentState:
        """Expose an immutable lifecycle view for diagnostics and tests."""
        return self.turn_coordinator.state

    @property
    def has_active_work(self) -> bool:
        active_task = any(
            record.state
            in {
                TaskState.CREATED,
                TaskState.ADMITTED,
                TaskState.QUEUED,
                TaskState.RUNNING,
                TaskState.CANCELLING,
            }
            for record in self.task_registry.records
        )
        active_turn = self.turn_coordinator.state.phase not in {
            TurnPhase.IDLE,
            TurnPhase.COMPLETED,
        }
        return bool(
            active_task
            or active_turn
            or self.output_fence.has_active_playback
            or self._pending_response_commits
            or self._active_deck_tasks
            or self._maintenance_tasks
            or self._active_response_provider_tasks
            or self._active_preoutput_tts_provider_tasks
            or self._voice_admissions
            or self._comment_admissions
        )

    def _begin_response_turn(self, turn_id: TurnId) -> None:
        """Move an already-admitted scheduler turn into reasoning.

        A pre-existing PLAYING state represents the retained lease.  The new
        response is marked as a replacement, while actual Sound cutover stays
        fenced by ``SchedulerOutputFence`` and its flush task.
        """
        previous = self.turn_coordinator.state
        replacement = previous.phase is TurnPhase.PLAYING
        epoch = int(self.cancellation_epoch)
        _ = self.turn_coordinator.enqueue(
            turn_id=str(turn_id), epoch=epoch, replacement=replacement
        )
        _ = self.turn_coordinator.start_reasoning(
            turn_id=str(turn_id), epoch=epoch
        )

    def response_cutover_pending(self, turn_id: TurnId) -> bool:
        """Record that a prepared replacement first frame awaits Sound ACK.

        This is invoked only after the bridge has a valid first RTP frame and
        before the transport opens its reducer-owned flush task.  It neither
        emits audio nor changes the output lease; both remain owned by the
        transport's exact flush acknowledgement path.
        """
        state = self.turn_coordinator.state
        if (
            state.turn_id != str(turn_id)
            or state.epoch != int(self.cancellation_epoch)
            or state.phase is not TurnPhase.SYNTHESIZING
        ):
            return False
        transition = self.turn_coordinator.await_cutover(
            turn_id=str(turn_id), epoch=int(self.cancellation_epoch)
        )
        return transition.state.phase is TurnPhase.CUTOVER_PENDING

    def response_cutover_failed(self, turn_id: TurnId) -> bool:
        """Keep the retained lease authoritative after a failed replacement."""
        state = self.turn_coordinator.state
        if (
            state.turn_id != str(turn_id)
            or state.epoch != int(self.cancellation_epoch)
            or state.phase is not TurnPhase.CUTOVER_PENDING
        ):
            return False
        transition = self.turn_coordinator.restore_retained_playback(
            turn_id=str(turn_id), epoch=int(self.cancellation_epoch)
        )
        return transition.state.phase is TurnPhase.PLAYING

    def response_playback_finished(self) -> bool:
        """Consume a Sound-validated physical completion for the active turn.

        The caller reaches this method only after ``SchedulerOutputFence`` has
        matched stream, turn, segment and output lease epoch.  The response
        state deliberately uses the session cancellation epoch instead of the
        independent output generation, so a stale media event cannot complete
        a newer logical turn.
        """
        state = self.turn_coordinator.state
        if (
            state.turn_id is None
            or state.epoch != int(self.cancellation_epoch)
            or state.phase is not TurnPhase.PLAYING
        ):
            return False
        transition = self.turn_coordinator.playback_finished(
            turn_id=state.turn_id, epoch=int(self.cancellation_epoch)
        )
        return transition.state.phase is TurnPhase.COMPLETED

    def _cancel_active_response_providers(
        self, cancelled: tuple[TaskRecord, ...]
    ) -> None:
        """Stop provider work only after the registry has closed its result gate."""
        for record in cancelled:
            provider_task = self._active_response_provider_tasks.get(
                record.request.task_id
            )
            if provider_task is not None and not provider_task.done():
                _ = provider_task.cancel()

    def _cancel_active_preoutput_tts_providers(
        self, cancelled: tuple[TaskRecord, ...]
    ) -> None:
        """Cancel only TTS coroutines whose task has not admitted audio yet."""
        for record in cancelled:
            provider_task = self._active_preoutput_tts_provider_tasks.get(
                record.request.task_id
            )
            if provider_task is not None and not provider_task.done():
                _ = provider_task.cancel()

    def _cancel_all_active_tts_providers(self) -> None:
        """Session termination also ends synthesis that has already started output."""
        for provider_task in self._active_preoutput_tts_provider_tasks.values():
            if not provider_task.done():
                _ = provider_task.cancel()

    def set_preoutput_tts_cancellation(
        self, callback: Callable[[TurnId], None] | None
    ) -> None:
        """Bind the transport's provider-cancellation handle to this session."""
        self.preoutput_tts_cancellation = callback

    def _cancel_preoutput_tts(self, cancelled: tuple[TaskRecord, ...]) -> None:
        callback = self.preoutput_tts_cancellation
        if callback is None:
            return
        for record in cancelled:
            task_id = record.request.task_id
            if (
                task_id in self._agent_tts_text
                and task_id in self._pending_response_commits
            ):
                callback(record.request.turn_id)

    def receive_comment(self, proposal: CommentProposal) -> RuntimeOutcome:
        """Reject the retired synchronous response path.

        Gate, model, tool, and TTS work all have asynchronous task lifecycles;
        a synchronous callback cannot own their cancellation and result fences.
        """
        return self._reject(proposal.correlation, "async_response_required")

    async def receive_comment_async(
        self,
        proposal: CommentProposal,
        *,
        admission_valid: Callable[[], bool] = lambda: True,
    ) -> RuntimeOutcome:
        coordinator = self.async_response_coordinator
        gate = self.async_agent_gate
        if (
            self.response_execution_mode is ResponseExecutionMode.NEW_SHADOW
            and coordinator is None
        ):
            return self._reject(proposal.correlation, "shadow_coordinator_missing")
        if coordinator is None or gate is None:
            return self._reject(proposal.correlation, "response_coordinator_missing")
        correlation = proposal.correlation
        audience_input = BrainAudienceInput(
            session_id=str(correlation.session_id),
            trace_id=str(correlation.trace_id),
            sequence=int(correlation.sequence),
            source=BrainAudienceSource.COMMENT,
            received_at_ms=self.clock(),
            text=proposal.text,
        )
        return await self._gate_and_enqueue_audience(
            gate,
            audience_input,
            correlation,
            lambda: self._admit_comment(
                coordinator, proposal, audience_input, correlation
            ),
            admission_valid=admission_valid,
        )

    async def _admit_comment(
        self,
        coordinator: AsyncResponseCoordinator,
        proposal: CommentProposal,
        audience_input: BrainAudienceInput,
        correlation: EventCorrelation,
    ) -> RuntimeOutcome:
        outcome = self.interaction_ingress.receive_comment(
            text=proposal.text, correlation=correlation
        )
        if not isinstance(outcome, InteractionAccepted) or outcome.turn_id is None:
            return self._reject(correlation, "scheduler_rejected")
        accepted_turn = TurnId(outcome.turn_id)
        self._advance_turn_epoch()
        self._begin_response_turn(accepted_turn)
        self._correlations.add(correlation)
        self._journal.dispatches.append(RuntimeDispatch(correlation, accepted_turn))
        await self._run_async_response(
            coordinator,
            audience_input,
            accepted_turn,
            correlation,
            shadow=self.response_execution_mode is ResponseExecutionMode.NEW_SHADOW,
        )
        return RuntimeOutcome(
            accepted=True, correlation=correlation, turn_id=accepted_turn
        )

    async def _gate_and_enqueue_audience(  # noqa: C901, PLR0911
        self,
        gate: AsyncAudienceGate,
        audience_input: BrainAudienceInput,
        correlation: EventCorrelation,
        admission: Callable[[], Coroutine[None, None, RuntimeOutcome]],
        *,
        admission_valid: Callable[[], bool] = lambda: True,
    ) -> RuntimeOutcome:
        if self._ended:
            return self._reject(correlation, "session_ended")
        if (
            correlation in self._correlations
            or correlation in self._pending_correlations
        ):
            return self._reject(correlation, "duplicate_correlation")
        self._pending_correlations.add(correlation)
        expected_revision = self.scheduler.snapshot.revision
        try:
            decision = await gate.evaluate(
                audience_input,
                active_summary=self._frontend_caption,
                recent_turn_context=self.interaction_ingress.data.recent_turn_context,
            )
        except asyncio.CancelledError:
            self._pending_correlations.discard(correlation)
            raise
        except Exception:
            self._pending_correlations.discard(correlation)
            _LOGGER.exception(
                "gate_failed trace=%s session=%s seq=%s source=%s rejected",
                correlation.trace_id,
                correlation.session_id,
                correlation.sequence,
                audience_input.source,
            )
            return self._reject(correlation, "agent_gate_failed")
        if decision is GateDecision.DISCARD:
            self._pending_correlations.discard(correlation)
            return self._reject(correlation, "agent_gate_discarded")
        if (
            self._ended
            or str(correlation.session_id) != str(self.scheduler.snapshot.session_id)
            or correlation in self._correlations
            or self.scheduler.snapshot.revision != expected_revision
            or not admission_valid()
        ):
            self._pending_correlations.discard(correlation)
            return self._reject(correlation, "audience_admission_stale")

        result = asyncio.get_running_loop().create_future()
        pending = _PendingAudienceAdmission(
            audience_input.source,
            correlation,
            admission,
            result,
            admission_valid,
        )
        async with self._admission_lock:
            queued = len(self._voice_admissions) + len(self._comment_admissions)
            if queued >= _AUDIENCE_QUEUE_CAPACITY:
                if audience_input.source is BrainAudienceSource.COMMENT:
                    self._pending_correlations.discard(correlation)
                    return self._reject(correlation, "audience_queue_full")
                if not self._comment_admissions:
                    self._pending_correlations.discard(correlation)
                    return self._reject(correlation, "audience_queue_full")
                evicted = self._comment_admissions.popleft()
                self._pending_correlations.discard(evicted.correlation)
                if not evicted.result.done():
                    evicted.result.set_result(
                        self._reject(evicted.correlation, "audience_comment_evicted")
                    )
            target = (
                self._voice_admissions
                if audience_input.source is BrainAudienceSource.ASR
                else self._comment_admissions
            )
            target.append(pending)
            if self._admission_worker is None or self._admission_worker.done():
                self._admission_worker = asyncio.create_task(
                    self._run_audience_admissions(),
                    name=f"audience-admission-{self.scheduler.snapshot.session_id}",
                )
        return await asyncio.shield(result)

    async def _run_audience_admissions(self) -> None:
        while True:
            async with self._admission_lock:
                if self._voice_admissions:
                    pending = self._voice_admissions.popleft()
                elif self._comment_admissions:
                    pending = self._comment_admissions.popleft()
                else:
                    self._admission_worker = None
                    return
            try:
                if self._ended or not pending.admission_valid():
                    outcome = self._reject(
                        pending.correlation, "audience_admission_stale"
                    )
                else:
                    outcome = await pending.run()
            except Exception:
                _LOGGER.exception(
                    "admission_failed trace=%s session=%s seq=%s source=%s rejected",
                    pending.correlation.trace_id,
                    pending.correlation.session_id,
                    pending.correlation.sequence,
                    pending.source,
                )
                outcome = self._reject(
                    pending.correlation, "audience_admission_failed"
                )
            finally:
                self._pending_correlations.discard(pending.correlation)
            if not pending.result.done():
                pending.result.set_result(outcome)

    async def _run_async_response(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        coordinator: AsyncResponseCoordinator,
        audience_input: BrainAudienceInput,
        turn_id: TurnId,
        correlation: EventCorrelation,
        *,
        shadow: bool = False,
    ) -> None:
        """Run the small response contract; only this runtime creates effects."""
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
        identity = self._speaker_identities.pop(correlation, None)
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
            speaker_profile_id=(
                None if identity is None else str(identity.profile_id)
            ),
            speaker_preferred_name=(
                None if identity is None else identity.preferred_name
            ),
            speaker_confidence=(None if identity is None else identity.confidence),
        )
        envelope = ExecutionEnvelope(
            session_id=self.scheduler.snapshot.session_id,
            turn_id=turn_id,
            segment_id=SegmentId(f"agent-{turn_id}"),
            revision=self.scheduler.snapshot.revision,
            cancellation_epoch=int(self.cancellation_epoch),
            deadline_ms=self.clock() + self.response_task_timeout_ms,
            allowed_actions=frozenset(
                {"breathe", "dance", "explain_point", "speak", "wave", "nod"}
            ),
            allowed_expressions=frozenset(),
            replacement=self.turn_coordinator.state.pending_interrupt,
        )
        response_task_id = TaskId(f"response-llm-initial-{turn_id}")
        scheduled = self.schedule_task(
            TaskRequest(
                task_id=response_task_id,
                session_id=envelope.session_id,
                turn_id=envelope.turn_id,
                parent_task_id=None,
                deadline_ms=TaskDeadlineMs(envelope.deadline_ms),
                snapshot_revision=envelope.revision,
                idempotency_key=IdempotencyKey(str(response_task_id)),
                kind=TaskKind.INTERACTIVE,
                segment_id=envelope.segment_id,
            ),
            correlation,
        )
        if not scheduled.accepted or self.executor.claim(response_task_id) is None:
            _ = self.turn_coordinator.fail(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            _LOGGER.debug("response_task_not_admitted turn=%s", turn_id)
            return
        try:
            initial = await self._await_response_provider(
                response_task_id, coordinator.initial_response(snapshot)
            )
        except _ResponseProviderCancelledError:
            _ = self.turn_coordinator.cancel(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            _LOGGER.debug("initial_response_cancelled turn=%s", turn_id)
            return
        except TimeoutError:
            _ = self.turn_coordinator.fail(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            _LOGGER.debug("initial_response_timed_out turn=%s", turn_id)
            return
        except (OSError, ValueError):
            _ = self.task_registry.fail(
                response_task_id, reason="initial_response_provider_failed"
            )
            _ = self.turn_coordinator.fail(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            _LOGGER.exception("initial_response_provider_failed turn=%s", turn_id)
            return
        if not isinstance(initial, ResponseProposal):
            _ = self.task_registry.fail(
                response_task_id, reason="initial_response_provider_invalid"
            )
            _ = self.turn_coordinator.fail(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            return
        if not self._response_envelope_is_current(envelope):
            _ = self.cancel_task(response_task_id, correlation)
            _ = self.turn_coordinator.cancel(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            _LOGGER.debug("response_superseded turn=%s", turn_id)
            return
        initial_accepted = self.reduce_task(
            TaskResult(
                response_task_id,
                envelope.session_id,
                envelope.turn_id,
                envelope.revision,
                TaskEffect("llm.initial", initial.reply[:240]),
                envelope.cancellation_epoch,
                envelope.segment_id,
            ),
            correlation,
        )
        if not initial_accepted.accepted:
            _ = self.turn_coordinator.cancel(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            _LOGGER.debug("initial_response_result_rejected turn=%s", turn_id)
            return

        if shadow:
            # Shadow evaluates the new minimal contract only.  In particular it
            # never executes a selected tool or creates TTS/context/frontend/
            # memory work; the accepted LLM task and this diagnostic are its
            # entire observable footprint.
            compiled = parse_inline_cues(
                initial.reply,
                allowed_actions=envelope.allowed_actions,
                allowed_expressions=envelope.allowed_expressions,
            )
            outcome = (
                f"intent={initial.intent};fallback={initial.used_text_fallback};"
                f"cues={len(compiled.cues)};"
                f"rejected_cues={compiled.rejected_cues};"
                f"empty={not compiled.spoken_text.strip()};phase=completed"
            )
            self.operational_journal.append(
                OperationalRecord(
                    stage="response_shadow",
                    trace_id=str(correlation.trace_id),
                    session_id=str(correlation.session_id),
                    turn_id=str(turn_id),
                    segment_id=str(envelope.segment_id),
                    task_id=str(response_task_id),
                    outcome=outcome,
                )
            )
            _ = self.turn_coordinator.complete_without_output(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            return

        if initial.intent == "answer":
            self._apply_coordinated_response(
                CoordinatedResponse(initial),
                audience_input,
                envelope,
                correlation,
                response_task_id,
            )
            return

        _ = self.turn_coordinator.wait_for_tool(
            turn_id=str(turn_id), epoch=envelope.cancellation_epoch
        )
        request = coordinator.tool_request(initial, snapshot)
        observation = "工具调用未成功完成。请基于已知信息简短说明。"
        parent_task_id = response_task_id
        if request is not None:
            tool_task_id = TaskId(f"response-tool-{turn_id}")
            tool_timeout_ms = coordinator.tool_timeout_ms(initial)
            tool_deadline_ms = envelope.deadline_ms
            if tool_timeout_ms is not None:
                tool_deadline_ms = min(
                    tool_deadline_ms, self.clock() + tool_timeout_ms
                )
            tool_scheduled = self.schedule_task(
                TaskRequest(
                    task_id=tool_task_id,
                    session_id=envelope.session_id,
                    turn_id=envelope.turn_id,
                    parent_task_id=response_task_id,
                    deadline_ms=TaskDeadlineMs(tool_deadline_ms),
                    snapshot_revision=envelope.revision,
                    idempotency_key=IdempotencyKey(str(tool_task_id)),
                    kind=TaskKind.DELIBERATIVE,
                    segment_id=envelope.segment_id,
                ),
                correlation,
            )
            if (
                tool_scheduled.accepted
                and self.executor.claim(tool_task_id) is not None
            ):
                parent_task_id = tool_task_id
                try:
                    provider_output = await self._await_response_provider(
                        tool_task_id, coordinator.execute_tool(request, snapshot)
                    )
                    tool_output = (
                        provider_output if isinstance(provider_output, str) else None
                    )
                except _ResponseProviderCancelledError:
                    _ = self.turn_coordinator.cancel(
                        turn_id=str(turn_id), epoch=envelope.cancellation_epoch
                    )
                    _LOGGER.debug("response_tool_cancelled turn=%s", turn_id)
                    return
                except TimeoutError:
                    tool_output = None
                    _LOGGER.debug("response_tool_timed_out turn=%s", turn_id)
                except (OSError, ValueError):
                    tool_output = None
                    _LOGGER.exception("response_tool_provider_failed turn=%s", turn_id)
                if not self._response_envelope_is_current(envelope):
                    _ = self.cancel_task(tool_task_id, correlation)
                    _ = self.turn_coordinator.cancel(
                        turn_id=str(turn_id), epoch=envelope.cancellation_epoch
                    )
                    return
                if tool_output is None:
                    _ = self.task_registry.fail(
                        tool_task_id, reason="response_tool_provider_failed"
                    )
                else:
                    tool_accepted = self.reduce_task(
                        TaskResult(
                            tool_task_id,
                            envelope.session_id,
                            envelope.turn_id,
                            envelope.revision,
                            TaskEffect("tool.observed", tool_output[:240]),
                            envelope.cancellation_epoch,
                            envelope.segment_id,
                        ),
                        correlation,
                    )
                    if not tool_accepted.accepted:
                        _ = self.turn_coordinator.cancel(
                            turn_id=str(turn_id), epoch=envelope.cancellation_epoch
                        )
                        return
                    observation = tool_output

        _ = self.turn_coordinator.resume_reasoning(
            turn_id=str(turn_id), epoch=envelope.cancellation_epoch
        )

        final_task_id = TaskId(f"response-llm-final-{turn_id}")
        final_scheduled = self.schedule_task(
            TaskRequest(
                task_id=final_task_id,
                session_id=envelope.session_id,
                turn_id=envelope.turn_id,
                parent_task_id=parent_task_id,
                deadline_ms=TaskDeadlineMs(envelope.deadline_ms),
                snapshot_revision=envelope.revision,
                idempotency_key=IdempotencyKey(str(final_task_id)),
                kind=TaskKind.INTERACTIVE,
                segment_id=envelope.segment_id,
            ),
            correlation,
        )
        if not final_scheduled.accepted or self.executor.claim(final_task_id) is None:
            _ = self.turn_coordinator.fail(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            return
        try:
            final = await self._await_response_provider(
                final_task_id, coordinator.final_response(snapshot, observation)
            )
        except _ResponseProviderCancelledError:
            _ = self.turn_coordinator.cancel(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            _LOGGER.debug("final_response_cancelled turn=%s", turn_id)
            return
        except TimeoutError:
            _LOGGER.debug("final_response_timed_out turn=%s", turn_id)
            if self._response_envelope_is_current(envelope):
                self._apply_coordinated_response(
                    CoordinatedResponse(
                        ResponseProposal("抱歉, 我暂时无法完成这项查询。", "answer")
                    ),
                    audience_input,
                    envelope,
                    correlation,
                    parent_task_id,
                )
            return
        except (OSError, ValueError):
            _ = self.task_registry.fail(
                final_task_id, reason="final_response_provider_failed"
            )
            if self._response_envelope_is_current(envelope):
                self._apply_coordinated_response(
                    CoordinatedResponse(
                        ResponseProposal("抱歉, 我暂时无法完成这项查询。", "answer"),
                        request,
                        observation,
                    ),
                    audience_input,
                    envelope,
                    correlation,
                    response_task_id,
                )
            return
        if not isinstance(final, ResponseProposal):
            _ = self.task_registry.fail(
                final_task_id, reason="final_response_provider_invalid"
            )
            _ = self.turn_coordinator.fail(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            return
        if not self._response_envelope_is_current(envelope):
            _ = self.cancel_task(final_task_id, correlation)
            _ = self.turn_coordinator.cancel(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            return
        final_accepted = self.reduce_task(
            TaskResult(
                final_task_id,
                envelope.session_id,
                envelope.turn_id,
                envelope.revision,
                TaskEffect("llm.final", final.reply[:240]),
                envelope.cancellation_epoch,
                envelope.segment_id,
            ),
            correlation,
        )
        if not final_accepted.accepted:
            _ = self.turn_coordinator.cancel(
                turn_id=str(turn_id), epoch=envelope.cancellation_epoch
            )
            return
        self._apply_coordinated_response(
            CoordinatedResponse(final, request, observation),
            audience_input,
            envelope,
            correlation,
            final_task_id,
        )

    def _response_envelope_is_current(self, envelope: ExecutionEnvelope) -> bool:
        return envelope.is_current(
            session_id=self.scheduler.snapshot.session_id,
            revision=self.scheduler.snapshot.revision,
            cancellation_epoch=int(self.cancellation_epoch),
            now_ms=self.clock(),
            session_ended=self._ended,
        )

    async def _await_response_provider(
        self,
        task_id: TaskId,
        operation: Awaitable[ResponseProposal | str | None],
    ) -> ResponseProposal | str | None:
        """Bind one provider coroutine to its registry task and absolute deadline."""
        record = self.task_registry.task(task_id)
        if record is None or record.state is not TaskState.RUNNING:
            raise _ResponseProviderCancelledError
        remaining_ms = int(record.request.deadline_ms) - self.clock()
        if remaining_ms <= 0:
            _ = self.task_registry.timeout(task_id)
            raise TimeoutError
        provider_task = asyncio.ensure_future(operation)
        self._active_response_provider_tasks[task_id] = provider_task
        try:
            return await asyncio.wait_for(provider_task, timeout=remaining_ms / 1_000)
        except TimeoutError:
            _ = self.task_registry.timeout(task_id)
            raise
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            raise _ResponseProviderCancelledError from None
        finally:
            _ = self._active_response_provider_tasks.pop(task_id, None)
            if not provider_task.done():
                _ = provider_task.cancel()

    @property
    def active_response_provider_task_ids(self) -> frozenset[TaskId]:
        """Expose active provider ownership without leaking mutable task handles."""
        return frozenset(self._active_response_provider_tasks)

    @property
    def active_preoutput_tts_provider_task_ids(self) -> frozenset[TaskId]:
        """Expose pre-output TTS ownership without leaking mutable task handles."""
        return frozenset(self._active_preoutput_tts_provider_tasks)

    def _apply_coordinated_response(
        self,
        response: CoordinatedResponse,
        audience_input: BrainAudienceInput,
        envelope: ExecutionEnvelope,
        correlation: EventCorrelation,
        parent_task_id: TaskId,
    ) -> None:
        parsed = parse_inline_cues(
            response.proposal.reply,
            allowed_actions=envelope.allowed_actions,
            allowed_expressions=envelope.allowed_expressions,
        )
        self.operational_journal.append(
            OperationalRecord(
                stage="response_compiled",
                trace_id=str(correlation.trace_id),
                session_id=str(correlation.session_id),
                turn_id=str(envelope.turn_id),
                segment_id=str(envelope.segment_id),
                task_id=str(parent_task_id),
                outcome=(
                    f"intent={response.proposal.intent};"
                    f"fallback={response.proposal.used_text_fallback};"
                    f"cues={len(parsed.cues)};"
                    f"rejected_cues={parsed.rejected_cues};"
                    f"empty={not parsed.spoken_text.strip()};"
                    f"tool={response.tool_request is not None}"
                ),
            )
        )
        if not parsed.spoken_text.strip():
            _ = self.turn_coordinator.fail(
                turn_id=str(envelope.turn_id), epoch=envelope.cancellation_epoch
            )
            _LOGGER.debug("response_rejected_empty turn=%s", envelope.turn_id)
            return
        _ = self.turn_coordinator.start_synthesizing(
            turn_id=str(envelope.turn_id), epoch=envelope.cancellation_epoch
        )
        provenance = ContextProvenance(
            session_id=SessionId(audience_input.session_id),
            turn_id=envelope.turn_id,
            segment_id=envelope.segment_id,
            sequence=ContextSequence(audience_input.sequence),
            source_id=ContextSourceId(audience_input.trace_id),
        )
        task_id = TaskId(f"response-tts-{envelope.turn_id}")
        self._pending_response_commits[task_id] = _PendingResponseCommit(
            provenance,
            audience_input.text,
            parsed.spoken_text,
            parsed.marked_text,
            response.observation if response.tool_request is not None else None,
        )
        # Accepted audience input and the validated Brain reply belong in
        # transient context even when synthesis later fails.  Commit them
        # before capturing the TTS task's data snapshot; doing this after task
        # admission makes the task stale against its own context write and
        # rejects output immediately after the first RTP frame.
        pending_commit = self._commit_response_context(task_id)
        outcome = self.schedule_task(
            TaskRequest(
                task_id=task_id,
                session_id=self.scheduler.snapshot.session_id,
                turn_id=envelope.turn_id,
                parent_task_id=parent_task_id,
                deadline_ms=TaskDeadlineMs(envelope.deadline_ms),
                snapshot_revision=envelope.revision,
                idempotency_key=IdempotencyKey(str(task_id)),
                kind=TaskKind.INTERACTIVE,
                segment_id=envelope.segment_id,
            ),
            correlation,
        )
        if outcome.accepted and self.executor.claim(task_id) is not None:
            self._agent_tts_text[task_id] = parsed.spoken_text
            if pending_commit is not None:
                self._schedule_memory_extraction(pending_commit, task_id, correlation)
                self._schedule_context_compaction(pending_commit, task_id, correlation)
        else:
            _ = self.cancel_task(task_id, correlation)
            _ = self.turn_coordinator.fail(
                turn_id=str(envelope.turn_id), epoch=envelope.cancellation_epoch
            )

    def _commit_response_context(
        self, task_id: TaskId
    ) -> _PendingResponseCommit | None:
        pending = self._pending_response_commits.pop(task_id, None)
        if pending is None:
            return None
        self.interaction_ingress.data.consider_context(
            FinalizedInput(pending.provenance, pending.input_text)
        )
        if pending.observation is not None:
            self.interaction_ingress.data.consider_context(
                ToolObservation(
                    ContextProvenance(
                        session_id=pending.provenance.session_id,
                        turn_id=pending.provenance.turn_id,
                        segment_id=pending.provenance.segment_id,
                        sequence=pending.provenance.sequence,
                        source_id=ContextSourceId(
                            f"{pending.provenance.source_id}:tool"
                        ),
                    ),
                    _bounded_observation_summary(pending.observation),
                )
            )
        self.interaction_ingress.data.consider_context(
            AcceptedOutput(pending.provenance, pending.spoken_text)
        )
        self._started_timeline_text[pending.provenance.turn_id] = (
            pending.marked_text,
            pending.provenance.segment_id,
        )
        return pending

    def _schedule_context_compaction(
        self,
        pending: _PendingResponseCommit,
        parent_task_id: TaskId,
        correlation: EventCorrelation,
    ) -> None:
        compactor = self.context_compactor
        if compactor is None:
            return
        composition = self.interaction_ingress.data.context.compose(
            _BRAIN_CONTEXT_MODEL, _BRAIN_CONTEXT_POLICY
        )
        if not composition.digests:
            return
        task_id = TaskId(f"context-compact-{pending.provenance.turn_id}")
        outcome = self.schedule_task(
            TaskRequest(
                task_id=task_id,
                session_id=pending.provenance.session_id,
                turn_id=pending.provenance.turn_id,
                parent_task_id=parent_task_id,
                deadline_ms=TaskDeadlineMs(self.clock() + 10_000),
                snapshot_revision=self.scheduler.snapshot.revision,
                idempotency_key=IdempotencyKey(str(task_id)),
                kind=TaskKind.MAINTENANCE,
                segment_id=pending.provenance.segment_id,
            ),
            correlation,
        )
        if not outcome.accepted or self.executor.claim(task_id) is None:
            return
        task = asyncio.create_task(
            self._run_context_compaction(
                compactor, task_id, composition, correlation
            )
        )
        self._maintenance_tasks.add(task)
        task.add_done_callback(self._maintenance_tasks.discard)

    async def _run_context_compaction(
        self,
        compactor: AsyncContextCompactor,
        task_id: TaskId,
        composition: ContextComposition,
        correlation: EventCorrelation,
    ) -> None:
        try:
            summary = await compactor.compact(composition)
        except (OSError, TimeoutError, ValueError):
            _ = self.task_registry.fail(task_id, reason="context_compactor_failed")
            _LOGGER.exception("context_compactor_failed task=%s", task_id)
            return
        if summary is None:
            _ = self.task_registry.fail(task_id, reason="context_compaction_rejected")
            return
        record = self.task_registry.task(task_id)
        if record is None:
            return
        accepted = self.reduce_task(
            TaskResult(
                task_id,
                record.request.session_id,
                record.request.turn_id,
                record.request.snapshot_revision,
                TaskEffect("context.compaction", summary[:240]),
                record.request.cancellation_epoch,
                record.request.segment_id,
            ),
            correlation,
        )
        if not accepted.accepted:
            return
        try:
            _ = self.interaction_ingress.data.context.compact(
                composition, summary=summary
            )
        except ContextCompactionError:
            _LOGGER.debug("context_compaction_stale task=%s", task_id)

    def _schedule_memory_extraction(
        self,
        pending: _PendingResponseCommit,
        parent_task_id: TaskId,
        correlation: EventCorrelation,
    ) -> None:
        extractor = self.memory_candidate_extractor
        if extractor is None:
            return
        task_id = TaskId(f"memory-extract-{pending.provenance.turn_id}")
        outcome = self.schedule_task(
            TaskRequest(
                task_id=task_id,
                session_id=pending.provenance.session_id,
                turn_id=pending.provenance.turn_id,
                parent_task_id=parent_task_id,
                deadline_ms=TaskDeadlineMs(self.clock() + 10_000),
                snapshot_revision=self.scheduler.snapshot.revision,
                idempotency_key=IdempotencyKey(str(task_id)),
                kind=TaskKind.MAINTENANCE,
                segment_id=pending.provenance.segment_id,
            ),
            correlation,
        )
        if not outcome.accepted or self.executor.claim(task_id) is None:
            return
        memory_revision = ProposalRevision(
            self.interaction_ingress.data.memory.snapshot.revision
        )
        task = asyncio.create_task(
            self._run_memory_extraction(
                extractor,
                task_id,
                pending,
                memory_revision,
                correlation,
            )
        )
        self._maintenance_tasks.add(task)
        task.add_done_callback(self._maintenance_tasks.discard)

    async def _run_memory_extraction(
        self,
        extractor: AsyncMemoryCandidateExtractor,
        task_id: TaskId,
        pending: _PendingResponseCommit,
        memory_revision: ProposalRevision,
        correlation: EventCorrelation,
    ) -> None:
        try:
            raw = await extractor.extract(
                user_text=pending.input_text, reply_text=pending.spoken_text
            )
        except (OSError, TimeoutError, ValueError):
            _ = self.task_registry.fail(task_id, reason="memory_extractor_failed")
            _LOGGER.exception("memory_extractor_failed task=%s", task_id)
            return
        candidate = None if raw is None else parse_memory_candidate(raw)
        if candidate is None:
            _ = self.task_registry.fail(task_id, reason="memory_candidate_rejected")
            return
        record = self.task_registry.task(task_id)
        if record is None:
            return
        accepted = self.reduce_task(
            TaskResult(
                task_id,
                pending.provenance.session_id,
                pending.provenance.turn_id,
                record.request.snapshot_revision,
                TaskEffect("memory.candidate", str(candidate.key)),
                record.request.cancellation_epoch,
                record.request.segment_id,
            ),
            correlation,
        )
        if not accepted.accepted:
            return
        _ = self.interaction_ingress.data.reduce_memory(
            MemoryProposal(
                key=candidate.key,
                value=candidate.value,
                category=MemoryCategory.ORDINARY_PREFERENCE,
                confidence=candidate.confidence,
                base_revision=memory_revision,
                provenance=MemoryProvenance(
                    source=MemorySource.AGENT_PROPOSAL,
                    trace_id=correlation.trace_id,
                    session_id=pending.provenance.session_id,
                    turn_id=pending.provenance.turn_id,
                    evidence_id=(
                        f"memory-extract:{pending.provenance.source_id}:{task_id}"
                    ),
                ),
            )
        )

    def take_started_timeline(
        self, turn_id: TurnId, *, audio_stream_id: str
    ) -> CaptionTimelineCommand | None:
        started = self._take_started_timeline(turn_id, audio_stream_id=audio_stream_id)
        return None if started is None else started[0]

    def _take_started_timeline(
        self, turn_id: TurnId, *, audio_stream_id: str
    ) -> tuple[CaptionTimelineCommand, SegmentId] | None:
        started = self._started_timeline_text.pop(turn_id, None)
        if started is None:
            return None
        marked_text, segment_id = started
        return CaptionTimelineCommand(
            timeline_id=f"agent-{turn_id}",
            marked_text=marked_text,
            audio_stream_id=audio_stream_id,
            cancellation_epoch=int(self.cancellation_epoch),
            start_rtp_timestamp=96_000,
        ), segment_id

    def schedule_started_timeline(
        self,
        turn_id: TurnId,
        *,
        audio_stream_id: str,
        correlation: EventCorrelation,
    ) -> tuple[TaskId, CaptionTimelineCommand] | None:
        """Claim a short-lived, reducer-fenced frontend timeline delivery.

        Timeline delivery begins only after the media adapter has admitted the
        first RTP frame.  It is nevertheless an externally-visible asynchronous
        effect, so it gets the same session/turn/epoch fencing as provider work.
        """
        started = self._take_started_timeline(turn_id, audio_stream_id=audio_stream_id)
        if started is None:
            return None
        timeline, segment_id = started
        return self.schedule_caption_timeline_delivery(
            turn_id, timeline, correlation=correlation, segment_id=segment_id
        )

    def schedule_caption_timeline_delivery(
        self,
        turn_id: TurnId,
        timeline: CaptionTimelineCommand,
        *,
        correlation: EventCorrelation,
        segment_id: SegmentId | None = None,
    ) -> tuple[TaskId, CaptionTimelineCommand] | None:
        """Register a media-admitted timeline delivery for the current turn."""
        task_id = TaskId(f"caption-timeline-{turn_id}")
        outcome = self.schedule_task(
            TaskRequest(
                task_id=task_id,
                session_id=self.scheduler.snapshot.session_id,
                turn_id=turn_id,
                parent_task_id=None,
                deadline_ms=TaskDeadlineMs(self.clock() + 5_000),
                snapshot_revision=self.scheduler.snapshot.revision,
                idempotency_key=IdempotencyKey(str(task_id)),
                kind=TaskKind.INTERACTIVE,
                segment_id=segment_id,
            ),
            correlation,
        )
        if not outcome.accepted or self.executor.claim(task_id) is None:
            _ = self.cancel_task(task_id, correlation)
            return None
        return task_id, timeline

    def caption_timeline_delivery_is_current(self, task_id: TaskId) -> bool:
        """Check a claimed timeline task immediately before frontend I/O."""
        record = self.task_registry.task(task_id)
        if record is None or record.state is not TaskState.RUNNING:
            return False
        request = record.request
        return (
            request.session_id == self.scheduler.snapshot.session_id
            and request.turn_id == self.scheduler.snapshot.active_turn_id
            and request.snapshot_revision == self.scheduler.snapshot.revision
            and request.data_snapshot == self._task_data_snapshot
            and request.cancellation_epoch == int(self.cancellation_epoch)
            and int(request.deadline_ms) >= self.clock()
            and request.capability_snapshot.issubset(self.agent_capabilities)
        )

    def complete_caption_timeline_delivery(
        self, task_id: TaskId, correlation: EventCorrelation
    ) -> bool:
        """Commit a successfully delivered timeline through the result gate."""
        record = self.task_registry.task(task_id)
        if record is None:
            return False
        return self.reduce_task(
            TaskResult(
                task_id=task_id,
                session_id=record.request.session_id,
                turn_id=record.request.turn_id,
                snapshot_revision=record.request.snapshot_revision,
                effect=TaskEffect("caption.timeline.delivered", str(task_id)),
                cancellation_epoch=record.request.cancellation_epoch,
                segment_id=record.request.segment_id,
            ),
            correlation,
        ).accepted

    def fail_caption_timeline_delivery(self, task_id: TaskId, *, reason: str) -> None:
        """Close a delivery task that could not reach the frontend."""
        _ = self.task_registry.fail(task_id, reason=reason)

    def schedule_sound_flush(
        self,
        turn_id: TurnId,
        segment_id: SegmentId,
        *,
        request_id: str,
        correlation: EventCorrelation,
    ) -> TaskId | None:
        """Register a replacement cutover before any Sound control I/O.

        The transport adapter owns the wire exchange, but it cannot decide
        whether a late acknowledgement may switch playback.  This short-lived
        interactive task is the scheduler-owned fence for that exchange.
        """
        task_id = TaskId(f"sound-flush-{request_id}")
        outcome = self.schedule_task(
            TaskRequest(
                task_id=task_id,
                session_id=self.scheduler.snapshot.session_id,
                turn_id=turn_id,
                parent_task_id=None,
                deadline_ms=TaskDeadlineMs(self.clock() + 3_000),
                snapshot_revision=self.scheduler.snapshot.revision,
                idempotency_key=IdempotencyKey(str(task_id)),
                kind=TaskKind.INTERACTIVE,
                segment_id=segment_id,
            ),
            correlation,
        )
        if not outcome.accepted or self.executor.claim(task_id) is None:
            _ = self.cancel_task(task_id, correlation)
            return None
        return task_id

    def sound_flush_is_current(self, task_id: TaskId) -> bool:
        """Fence a flush immediately before it can admit replacement output."""
        record = self.task_registry.task(task_id)
        if record is None or record.state is not TaskState.RUNNING:
            return False
        request = record.request
        if int(request.deadline_ms) < self.clock():
            _ = self.task_registry.timeout(task_id)
            return False
        return (
            request.session_id == self.scheduler.snapshot.session_id
            and request.turn_id == self.scheduler.snapshot.active_turn_id
            and request.snapshot_revision == self.scheduler.snapshot.revision
            and request.data_snapshot == self._task_data_snapshot
            and request.cancellation_epoch == int(self.cancellation_epoch)
            and request.capability_snapshot.issubset(self.agent_capabilities)
        )

    def complete_sound_flush(
        self, task_id: TaskId, correlation: EventCorrelation
    ) -> bool:
        """Commit a Sound-acknowledged cutover through the result reducer."""
        record = self.task_registry.task(task_id)
        if record is None:
            return False
        return self.reduce_task(
            TaskResult(
                task_id=task_id,
                session_id=record.request.session_id,
                turn_id=record.request.turn_id,
                snapshot_revision=record.request.snapshot_revision,
                effect=TaskEffect("sound.flush.admitted", str(task_id)),
                cancellation_epoch=record.request.cancellation_epoch,
                segment_id=record.request.segment_id,
            ),
            correlation,
        ).accepted

    def fail_sound_flush(self, task_id: TaskId, *, reason: str) -> None:
        """Close a rejected, cancelled, or stale replacement cutover task."""
        record = self.task_registry.task(task_id)
        if record is None or record.state is not TaskState.RUNNING:
            return
        if int(record.request.deadline_ms) < self.clock():
            _ = self.task_registry.timeout(task_id)
            _ = self.response_cutover_failed(record.request.turn_id)
            return
        _ = self.task_registry.fail(task_id, reason=reason)
        _ = self.response_cutover_failed(record.request.turn_id)

    async def run_agent_tts_for_turn(
        self,
        turn_id: TurnId,
        synthesize: AgentTtsSynthesize,
        correlation: EventCorrelation,
        output_started: Callable[[], None] | None = None,
    ) -> bool:
        """Run one reducer-approved TTS task and commit admitted output.

        The caller owns the transport adapter; this runtime retains task lifecycle
        validation so an unaccepted or stale plan can never reach TTS.  The
        adapter invokes ``output_started`` after synthesis and output admission,
        but before its paced RTP playback.  Playback may outlive the active turn;
        task completion must not, or stale results strand pending tasks.
        """
        for task_id, text in tuple(self._agent_tts_text.items()):
            record = self.task_registry.task(task_id)
            if (
                record is None
                or record.request.turn_id != turn_id
                or record.state is not TaskState.RUNNING
            ):
                continue
            return await self._run_agent_tts_task(
                _AgentTtsExecution(
                    task_id, text, record, correlation, output_started
                ),
                synthesize,
            )
        return False

    async def _run_agent_tts_task(
        self,
        execution: _AgentTtsExecution,
        synthesize: AgentTtsSynthesize,
    ) -> bool:
        task_id = execution.task_id
        text = execution.text
        record = execution.record
        correlation = execution.correlation
        if not text.strip():
            _ = self.cancel_task(task_id, correlation)
            _ = self._agent_tts_text.pop(task_id, None)
            return False
        committed = False

        def accept_output_started() -> bool:
            nonlocal committed
            if committed:
                return True
            committed = self.reduce_task(
                TaskResult(
                    task_id,
                    record.request.session_id,
                    record.request.turn_id,
                    record.request.snapshot_revision,
                    TaskEffect("tts.emitted", text[:240]),
                    record.request.cancellation_epoch,
                    record.request.segment_id,
                ),
                correlation,
            ).accepted
            if committed:
                _ = self.turn_coordinator.playback_started(
                    turn_id=str(record.request.turn_id),
                    epoch=record.request.cancellation_epoch,
                )
                if execution.output_started is not None:
                    execution.output_started()
            return committed

        try:
            emitted = await self._await_preoutput_tts_provider(
                task_id, synthesize(text, accept_output_started)
            )
        except _TtsProviderCancelledError:
            emitted = False
            _LOGGER.debug("agent_tts_cancelled task=%s", task_id)
        except TimeoutError:
            emitted = False
            _LOGGER.debug("agent_tts_timed_out task=%s", task_id)
        except (OSError, ValueError):
            emitted = False
            _LOGGER.exception("agent_tts_execution_failed task=%s", task_id)
        if not emitted or not committed:
            _ = self.cancel_task(task_id, correlation)
            _ = self._agent_tts_text.pop(task_id, None)
            _ = self._pending_response_commits.pop(task_id, None)
            _ = self.turn_coordinator.fail(
                turn_id=str(record.request.turn_id),
                epoch=record.request.cancellation_epoch,
            )
            return False
        _ = self._agent_tts_text.pop(task_id, None)
        return True

    async def _await_preoutput_tts_provider(
        self, task_id: TaskId, operation: Awaitable[bool]
    ) -> bool:
        """Run TTS under the task deadline until audio crosses its result gate."""
        record = self.task_registry.task(task_id)
        if record is None or record.state is not TaskState.RUNNING:
            raise _TtsProviderCancelledError
        remaining_ms = int(record.request.deadline_ms) - self.clock()
        if remaining_ms <= 0:
            _ = self.task_registry.timeout(task_id)
            raise TimeoutError
        provider_task = asyncio.ensure_future(operation)
        self._active_preoutput_tts_provider_tasks[task_id] = provider_task
        try:
            return await asyncio.wait_for(provider_task, timeout=remaining_ms / 1_000)
        except TimeoutError:
            _ = self.task_registry.timeout(task_id)
            raise
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            raise _TtsProviderCancelledError from None
        finally:
            _ = self._active_preoutput_tts_provider_tasks.pop(task_id, None)
            if not provider_task.done():
                _ = provider_task.cancel()

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
                correlation=correlation
            ):
                _ = correlation
                return self.enroll_profile_from_evidence(control)

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
    ) -> RuntimeOutcome:
        _ = event
        return self._reject(correlation, "async_response_required")

    async def receive_asr_final_async(
        self,
        event: ASRAudienceEvent,
        correlation: EventCorrelation,
        *,
        admission_valid: Callable[[], bool] = lambda: True,
    ) -> RuntimeOutcome:
        """Run finalized ASR through the same async Gate and Brain as comments."""
        coordinator = self.async_response_coordinator
        gate = self.async_agent_gate
        if (
            self.response_execution_mode is ResponseExecutionMode.NEW_SHADOW
            and coordinator is None
        ):
            return self._reject(correlation, "shadow_coordinator_missing")
        if coordinator is None or gate is None:
            return self._reject(correlation, "response_coordinator_missing")
        audience_input = BrainAudienceInput(
            session_id=str(correlation.session_id),
            trace_id=str(correlation.trace_id),
            sequence=int(correlation.sequence),
            source=BrainAudienceSource.ASR,
            received_at_ms=event.received_at_ms,
            text=event.text,
        )
        return await self._gate_and_enqueue_audience(
            gate,
            audience_input,
            correlation,
            lambda: self._admit_asr_final(
                coordinator, event, audience_input, correlation
            ),
            admission_valid=admission_valid,
        )

    async def _admit_asr_final(
        self,
        coordinator: AsyncResponseCoordinator,
        event: ASRAudienceEvent,
        audience_input: BrainAudienceInput,
        correlation: EventCorrelation,
    ) -> RuntimeOutcome:
        self._recognize_voice(event, correlation)
        transition = self.scheduler.apply(
            StartTurn(
                expected_revision=self.scheduler.snapshot.revision,
                event=SchedulerEvent(event_type="asr.final", correlation=correlation),
            )
        )
        if not isinstance(transition, TransitionAccepted):
            return self._reject(correlation, "scheduler_rejected")
        accepted_turn = transition.accepted_event.turn_id
        self._advance_turn_epoch()
        self._begin_response_turn(accepted_turn)
        self._correlations.add(correlation)
        self._journal.dispatches.append(RuntimeDispatch(correlation, accepted_turn))
        await self._run_async_response(
            coordinator,
            audience_input,
            accepted_turn,
            correlation,
            shadow=self.response_execution_mode is ResponseExecutionMode.NEW_SHADOW,
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
        self._ended = True
        worker = self._admission_worker
        if worker is not None and not worker.done():
            _ = worker.cancel()
        self._admission_worker = None
        for pending in (*self._voice_admissions, *self._comment_admissions):
            if not pending.result.done():
                pending.result.set_result(
                    self._reject(pending.correlation, "session_ended")
                )
        self._voice_admissions.clear()
        self._comment_admissions.clear()
        self._pending_correlations.clear()
        cancelled = self.task_registry.cancel_pending(reason="session_ended")
        self._cancel_preoutput_tts(cancelled)
        self._cancel_active_response_providers(cancelled)
        self._cancel_active_preoutput_tts_providers(cancelled)
        self._cancel_all_active_tts_providers()
        for command_id in tuple(self._active_deck_tasks.values()):
            _ = self.deck_dispatcher.cancel(command_id)
        self._deck_intents.clear()
        self._active_deck_tasks.clear()
        self._agent_tts_text.clear()
        self._pending_response_commits.clear()
        self._started_timeline_text.clear()
        for task in tuple(self._maintenance_tasks):
            _ = task.cancel()
        self._maintenance_tasks.clear()
        self._voice_evidence_ranges.clear()
        self._voice_evidence_cache.clear()
        self._speaker_identities.clear()
        self._re_enrollment_required.clear()
        self.interaction_ingress.data.clear_session()
        self.interaction_ingress.clear_session_data()
        _ = self.task_registry.clear_terminal_tombstones()
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
                request.cancellation_epoch,
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
                request.cancellation_epoch,
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
                request.request.cancellation_epoch,
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

            if record is None or record.state is not TaskState.QUEUED:
                intent = self._deck_intents.pop(task_id)

                self._cancel_pending_deck_presentation(intent.command.command_id)

    def _cancel_pending_deck_presentation(self, command_id: CommandId) -> None:
        self.interaction_ingress.reducer.cancel_presentation(command_id)
        _ = self._presentation_correlations.pop(command_id, None)

    @property
    def _task_data_snapshot(self) -> TaskStateSnapshot:
        return self.interaction_ingress.data.task_snapshot

    def _admit_task(self, request: TaskRequest) -> TaskRegistrationResult:
        _ = self.task_registry.expire_terminal_tombstones(now_ms=self.clock())
        if request.cancellation_epoch != int(self.cancellation_epoch):
            return TaskRegistrationRejected(TaskRegistrationRejection.STALE_SNAPSHOT)
        rejection = scheduling_rejection(request, self.scheduler.snapshot)

        if rejection is not None:
            return TaskRegistrationRejected(rejection)

        if request.data_snapshot != self._task_data_snapshot:
            return TaskRegistrationRejected(TaskRegistrationRejection.STALE_SNAPSHOT)

        return self.task_registry.register(request)

    def schedule_task(
        self, request: TaskRequest, correlation: EventCorrelation
    ) -> RuntimeOutcome:
        task_request = replace(
            with_current_data_snapshot(request, self._task_data_snapshot),
            cancellation_epoch=int(self.cancellation_epoch),
            capability_snapshot=self.agent_capabilities,
        )

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
        record = self.task_registry.task(result.task_id)
        if (
            record is not None
            and not record.request.capability_snapshot.issubset(
                self.agent_capabilities
            )
        ):
            _ = self.task_registry.cancel(result.task_id, reason="capability_revoked")
            return self._reject(correlation, "task_capability_revoked")
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
