"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
from orchestrator.scheduler_tasks import SchedulerTaskFacade
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
    TaskRequest,
)
from orchestrator.transient_context import (
    ContextProvenance,
    ContextSequence,
    ContextSourceId,
    FinalizedInput,
)


def test_session_data_state_admits_only_approved_memory_and_finalized_context() -> None:
    # Given: one live session with immutable knowledge attribution.

    """函数契约说明.

    功能: 验证 session data state admits
    only approved memory and finalized
    context 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    state = _state()

    # When: a supported ordinary preference and a finalized input are accepted.

    memory_result = state.reduce_memory(_memory_proposal())

    state.consider_context(_finalized_input("请叫我小莓"))

    # Then: both approved sources appear in the bounded prompt input snapshot.

    assert isinstance(memory_result, MemoryCommitAccepted)

    prompt_snapshot = state.prompt_snapshot(max_context_chars=1_000)

    assert prompt_snapshot.task_state.memory_revision == 1

    assert prompt_snapshot.task_state.context_generation == 0

    assert prompt_snapshot.memory_entries == ("preferred_name=小莓",)

    assert prompt_snapshot.context_entries == ("请叫我小莓",)


def test_session_data_state_reset_and_profile_deletion_invalidate_prior_work() -> None:
    # Given: work captured after a consented profile and finalized context exist.

    """函数契约说明.

    功能: 验证 session data state reset and
    profile deletion invalidate prior
    work 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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


def test_memory_delete_rejects_previously_admitted_task_without_effect() -> None:
    # Given: a task admitted against the current session data snapshot.

    """函数契约说明.

    功能: 验证 memory delete rejects
    previously admitted task without
    effect 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    facade = SchedulerTaskFacade.create(
        scheduler,
        SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        data_snapshot_provider=lambda: state.task_snapshot,
    )

    request = _task_request(scheduler, task_snapshot)

    assert isinstance(facade.schedule(request), TaskRegistrationAccepted)

    # When: deletion advances the persisted memory revision before task completion.

    state.delete_memory(MemoryKey("preferred_name"))

    result = facade.reduce(_task_result(request), now_ms=0)

    # Then: the stale worker result cannot complete or emit its proposed effect.

    assert result == TaskResultRejected(TaskResultRejection.STALE_DATA_SNAPSHOT)


def test_memory_store_persists_human_readable_approved_preference(
    tmp_path: Path,
) -> None:
    # Given: a real scheduler-owned file store for a new session.

    """函数契约说明.

    功能: 验证 memory store persists human
    readable approved preference
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

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
    """函数契约说明.

    功能: 执行 _state 的同步逻辑,并协调 create,
    SessionId, RetrievalFixtureProvider,
    KnowledgeRef。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `SessionDataState`。
    """

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
    """函数契约说明.

    功能: 执行 _memory_proposal 的同步逻辑,并协调
    MemoryProposal, MemoryKey,
    MemoryConfidence, ProposalRevision。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `MemoryProposal`。
    """

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
    """函数契约说明.

    功能: 执行 _finalized_input 的同步逻辑,并协调
    FinalizedInput, ContextProvenance,
    SessionId, TurnId。
    参数: text: str。 必填。
    契约: 同步调用。 返回 `FinalizedInput`。
    """

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
    """函数契约说明.

    功能: 执行 _task_request 的同步逻辑,并协调
    TaskRequest, TaskId, SessionId,
    TaskDeadlineMs。
    参数: scheduler: SessionScheduler。 必填。
    data_snapshot: TaskStateSnapshot。
    必填。
    契约: 同步调用。 返回 `TaskRequest`。
    """

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
    """函数契约说明.

    功能: 执行 _task_result 的同步逻辑,并协调
    TaskResult, TaskEffect。
    参数: request: TaskRequest。 必填。
    契约: 同步调用。 返回 `TaskResult`。
    """

    return TaskResult(
        task_id=request.task_id,
        session_id=request.session_id,
        turn_id=request.turn_id,
        snapshot_revision=request.snapshot_revision,
        effect=TaskEffect("answer", "stale"),
    )
