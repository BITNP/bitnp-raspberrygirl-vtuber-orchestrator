from pathlib import Path

from orchestrator.identity import (
    EncryptedVoiceTemplate,
    ProfileEnrollment,
    ProfileRecognition,
    ProfileRecognitionUnknown,
    RecognitionConfidence,
    VoiceProfileId,
)
from orchestrator.ids import SegmentId, SessionId, TraceId, TurnId
from orchestrator.memory import (
    MemoryCategory,
    MemoryCommitAccepted,
    MemoryConfidence,
    MemoryKey,
    MemoryProposal,
    MemoryProvenance,
    MemorySource,
    ProposalRevision,
)
from orchestrator.memory_store import JsonMemoryStore
from orchestrator.retrieval import KnowledgeRef, RetrievalFixtureProvider
from orchestrator.session_data import SessionDataState
from orchestrator.sessions import (
    EventCorrelation,
    EventSequence,
    SchedulerEvent,
    SessionScheduler,
    StartTurn,
    StateRevision,
    TransitionAccepted,
)
from orchestrator.state_snapshots import (
    CorpusRevision,
    IndexRevision,
    TaskStateSnapshot,
)
from orchestrator.task_reducer import (
    TaskEffect,
    TaskResult,
    TaskResultReducer,
    TaskResultRejected,
    TaskResultRejection,
)
from orchestrator.task_registry import (
    IdempotencyKey,
    SchedulerTaskConfig,
    TaskDeadlineMs,
    TaskId,
    TaskKind,
    TaskRegistrationAccepted,
    TaskRegistry,
    TaskRequest,
)
from orchestrator.transient_context import (
    AcceptedOutput,
    ContextProvenance,
    ContextSequence,
    ContextSourceId,
    FinalizedInput,
)


def test_session_data_state_admits_only_approved_memory_and_finalized_context() -> None:
    # Given: one live session with immutable knowledge attribution.

    state = _state()

    # When: a supported ordinary preference and a finalized input are accepted.

    memory_result = state.reduce_memory(_memory_proposal())

    state.consider_context(_finalized_input("请叫我小莓"))

    # Then: both approved sources appear in the bounded prompt input snapshot.

    assert isinstance(memory_result, MemoryCommitAccepted)

    prompt_snapshot = state.prompt_snapshot(max_context_chars=1_000)

    assert prompt_snapshot.task_state.memory_revision == 1

    assert prompt_snapshot.task_state.context_generation == 1

    assert prompt_snapshot.memory_entries == ("preferred_name=小莓",)

    assert prompt_snapshot.context_entries == ("请叫我小莓",)


def test_session_data_state_reset_and_profile_deletion_invalidate_prior_work() -> None:
    # Given: work captured after a consented profile and finalized context exist.

    state = _state()

    profile_id = VoiceProfileId("profile-1")

    _ = state.enroll_profile(
        ProfileEnrollment(
            profile_id=profile_id,
            preferred_name="小莓",
            encrypted_template=EncryptedVoiceTemplate(b"ciphertext"),
            consented=True,
        )
    )

    state.consider_context(_finalized_input("介绍产品"))

    captured = state.task_snapshot

    # When: the session resets context and the user deletes the profile.

    state.reset_context()

    state.delete_profile(profile_id)

    # Then: queued work is stale and recognition cannot personalize after deletion.

    assert state.is_current(captured) is False

    assert (
        state.recognize_profile(
            ProfileRecognition(profile_id, RecognitionConfidence(99))
        )
        == ProfileRecognitionUnknown()
    )


def test_recent_turn_context_contains_only_the_latest_accepted_exchange() -> None:
    state = _state()
    first = _finalized_input("介绍产品")
    state.consider_context(first)
    state.consider_context(AcceptedOutput(first.provenance, "这是产品介绍。"))

    latest = _finalized_input("它是否支持语音")
    state.consider_context(latest)
    state.consider_context(AcceptedOutput(latest.provenance, "支持实时语音交互。"))

    assert state.recent_turn_context == (
        "用户 - 它是否支持语音",
        "智能体 - 支持实时语音交互。",
    )


def test_memory_delete_rejects_previously_admitted_task_without_effect() -> None:
    # Given: a task admitted against the current session data snapshot.

    state = _state()

    scheduler = SessionScheduler(
        session_id=SessionId("session-1"), turn_id_prefix="turn"
    )

    transition = scheduler.apply(
        StartTurn(
            expected_revision=StateRevision(0),
            event=SchedulerEvent(
                "audience.input",
                EventCorrelation(
                    TraceId("trace-1"),
                    SessionId("session-1"),
                    EventSequence(1),
                ),
            ),
        )
    )

    assert isinstance(transition, TransitionAccepted)

    task_snapshot = state.task_snapshot

    registry = TaskRegistry(
        session_id=SessionId("session-1"),
        config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )

    request = _task_request(scheduler, task_snapshot)

    assert isinstance(registry.register(request), TaskRegistrationAccepted)

    # When: deletion advances the persisted memory revision before task completion.

    state.delete_memory(MemoryKey("preferred_name"))

    result = TaskResultReducer(registry).reduce(
        _task_result(request),
        snapshot=scheduler.snapshot,
        data_snapshot=state.task_snapshot,
        now_ms=0,
    )

    # Then: the stale worker result cannot complete or emit its proposed effect.

    assert result == TaskResultRejected(TaskResultRejection.STALE_DATA_SNAPSHOT)


def test_memory_store_persists_human_readable_approved_preference(
    tmp_path: Path,
) -> None:
    # Given: a real scheduler-owned file store for a new session.

    path = tmp_path / "ordinary-preferences.json"

    state = SessionDataState.create(
        session_id=SessionId("session-1"),
        retrieval=RetrievalFixtureProvider(refs=()),
        memory_store=JsonMemoryStore(path),
    )

    # When: policy accepts a supported ordinary preference proposal.

    result = state.reduce_memory(_memory_proposal())

    # Then: durable human-readable state contains only approved preference metadata.

    assert isinstance(result, MemoryCommitAccepted)

    document = path.read_text(encoding="utf-8")

    assert '"revision": 1' in document

    assert '"value": "小莓"' in document

    assert "ciphertext" not in document


def _state() -> SessionDataState:

    return SessionDataState.create(
        session_id=SessionId("session-1"),
        retrieval=RetrievalFixtureProvider(
            refs=(
                KnowledgeRef(
                    "kb-1",
                    "产品",
                    "BitNet 产品介绍",
                    "corpus-1",
                    CorpusRevision(2),
                    "index-1",
                    IndexRevision(3),
                ),
            )
        ),
    )


def _memory_proposal() -> MemoryProposal:

    return MemoryProposal(
        key=MemoryKey("preferred_name"),
        value="小莓",
        category=MemoryCategory.ORDINARY_PREFERENCE,
        confidence=MemoryConfidence(95),
        base_revision=ProposalRevision(0),
        provenance=MemoryProvenance(
            source=MemorySource.AGENT_PROPOSAL,
            trace_id=TraceId("trace-1"),
            session_id=SessionId("session-1"),
            turn_id=TurnId("turn-1"),
            evidence_id="finalized-input-1",
        ),
    )


def _finalized_input(text: str) -> FinalizedInput:

    return FinalizedInput(
        ContextProvenance(
            session_id=SessionId("session-1"),
            turn_id=TurnId("turn-1"),
            segment_id=SegmentId("segment-1"),
            sequence=ContextSequence(1),
            source_id=ContextSourceId("finalized-input-1"),
        ),
        text,
    )


def _task_request(
    scheduler: SessionScheduler,
    data_snapshot: TaskStateSnapshot,
) -> TaskRequest:

    assert scheduler.snapshot.active_turn_id is not None

    return TaskRequest(
        task_id=TaskId("task-1"),
        session_id=SessionId("session-1"),
        turn_id=scheduler.snapshot.active_turn_id,
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(100),
        snapshot_revision=scheduler.snapshot.revision,
        idempotency_key=IdempotencyKey("answer-1"),
        kind=TaskKind.INTERACTIVE,
        data_snapshot=data_snapshot,
    )


def _task_result(request: TaskRequest) -> TaskResult:

    return TaskResult(
        task_id=request.task_id,
        session_id=request.session_id,
        turn_id=request.turn_id,
        snapshot_revision=request.snapshot_revision,
        effect=TaskEffect("answer", "stale"),
    )
