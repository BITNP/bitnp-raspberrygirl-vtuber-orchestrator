from dataclasses import replace

import pytest

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.scheduler_tasks import SchedulerTaskFacade
from orchestrator.sessions import (
    EventCorrelation,
    EventSequence,
    SchedulerEvent,
    SessionScheduler,
    SessionSnapshot,
    StartTurn,
    StateRevision,
    TransitionAccepted,
    TransitionRejected,
)
from orchestrator.state_snapshots import (
    ConsentRevision,
    ContextGeneration,
    CorpusRevision,
    IndexRevision,
    MemoryRevision,
    ProfileRevision,
    TaskStateSnapshot,
)
from orchestrator.task_reducer import (
    TaskEffect,
    TaskResult,
    TaskResultAccepted,
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
    TaskRegistrationDuplicate,
    TaskRegistrationRejected,
    TaskRegistrationRejection,
    TaskRequest,
)


def test_scheduler_facade_admits_only_current_turn_result_through_reducer() -> None:
    # Given: scheduler-owned work registered against its accepted turn snapshot.
    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )
    first_turn = _start_turn(scheduler, StateRevision(0), _event("trace-1", 1))
    facade = SchedulerTaskFacade.create(
        scheduler,
        SchedulerTaskConfig(
            allowed_kinds=frozenset({TaskKind.INTERACTIVE}),
            max_children_per_task=1,
        ),
    )
    request = TaskRequest(
        task_id=TaskId("task-1"),
        session_id=SessionId("session-1"),
        turn_id=first_turn,
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(200),
        snapshot_revision=scheduler.snapshot.revision,
        idempotency_key=IdempotencyKey("answer-1"),
        kind=TaskKind.INTERACTIVE,
    )
    match facade.schedule(request):
        case TaskRegistrationAccepted():
            pass
        case TaskRegistrationDuplicate() | TaskRegistrationRejected():
            pytest.fail("current scheduler task was not registered")

    # When: a newer scheduler turn supersedes the task's captured snapshot.
    _ = _start_turn(scheduler, scheduler.snapshot.revision, _event("trace-2", 2))
    outcome = facade.reduce(_result(first_turn), now_ms=100)

    # Then: only the facade's reducer path rejects the stale effect proposal.
    match outcome:
        case TaskResultRejected(reason=TaskResultRejection.STALE_REVISION):
            pass
        case TaskResultAccepted():
            pytest.fail("stale scheduler task result was admitted")
        case TaskResultRejected():
            pytest.fail("stale scheduler task result had the wrong rejection")


def test_scheduler_facade_rejects_stale_snapshot_before_registry_mutation() -> None:
    # Given: an active scheduler turn at revision one.
    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )
    turn_id = _start_turn(scheduler, StateRevision(0), _event("trace-1", 1))
    facade = SchedulerTaskFacade.create(
        scheduler,
        SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )
    stale = _request(
        task_id="task-1",
        key="answer-1",
        snapshot=replace(scheduler.snapshot, revision=StateRevision(0)),
    )

    # When: scheduler-facing registration receives stale snapshot work.
    scheduled = facade.schedule(stale)
    later_result = facade.reduce(_result(turn_id), now_ms=100)

    # Then: rejection occurs before retaining work or admitting a later effect.
    match scheduled:
        case TaskRegistrationRejected(TaskRegistrationRejection.STALE_SNAPSHOT):
            pass
        case TaskRegistrationAccepted() | TaskRegistrationDuplicate():
            pytest.fail("stale snapshot task was registered")
        case TaskRegistrationRejected():
            pytest.fail("stale snapshot task had the wrong rejection")
    assert facade.registry.records == ()
    match later_result:
        case TaskResultRejected(TaskResultRejection.TASK_NOT_FOUND):
            pass
        case TaskResultAccepted():
            pytest.fail("unregistered stale task admitted an effect")
        case TaskResultRejected():
            pytest.fail("unregistered stale task had the wrong reduction")


def test_scheduler_facade_rejects_inactive_turn_before_registry_mutation() -> None:
    # Given: an active scheduler turn and a request targeting another turn.
    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )
    _ = _start_turn(scheduler, StateRevision(0), _event("trace-1", 1))
    facade = SchedulerTaskFacade.create(
        scheduler,
        SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
    )
    inactive = replace(
        _request(task_id="task-2", key="answer-2", snapshot=scheduler.snapshot),
        turn_id=TurnId("turn-inactive"),
    )

    # When: scheduler-facing registration receives an inactive-turn task.
    scheduled = facade.schedule(inactive)

    # Then: the live turn guard rejects it before task storage changes.
    match scheduled:
        case TaskRegistrationRejected(TaskRegistrationRejection.ACTIVE_TURN_MISMATCH):
            pass
        case TaskRegistrationAccepted() | TaskRegistrationDuplicate():
            pytest.fail("inactive turn task was registered")
        case TaskRegistrationRejected():
            pytest.fail("inactive turn task had the wrong rejection")
    assert facade.registry.records == ()


def test_scheduler_facade_rejects_task_with_stale_data_snapshot() -> None:
    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )
    _ = _start_turn(scheduler, StateRevision(0), _event("trace-1", 1))
    facade = SchedulerTaskFacade.create(
        scheduler,
        SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 1),
        _data_snapshot(memory=2),
    )
    request = _request(task_id="task-3", key="answer-3", snapshot=scheduler.snapshot)

    match facade.schedule(request):
        case TaskRegistrationRejected(TaskRegistrationRejection.STALE_SNAPSHOT):
            pass
        case TaskRegistrationAccepted() | TaskRegistrationDuplicate():
            pytest.fail("task with stale data snapshot was registered")
        case TaskRegistrationRejected():
            pytest.fail("stale data snapshot had the wrong rejection")
    assert facade.registry.records == ()


def _start_turn(
    scheduler: SessionScheduler,
    revision: StateRevision,
    event: SchedulerEvent,
) -> TurnId:
    transition = scheduler.apply(
        StartTurn(
            expected_revision=revision,
            event=event,
        )
    )
    match transition:
        case TransitionAccepted(snapshot=snapshot):
            assert snapshot.active_turn_id is not None
            return snapshot.active_turn_id
        case TransitionRejected():
            pytest.fail("scheduler turn setup was rejected")


def _event(trace_id: str, sequence: int) -> SchedulerEvent:
    return SchedulerEvent(
        event_type="audience.input",
        correlation=EventCorrelation(
            trace_id=TraceId(trace_id),
            session_id=SessionId("session-1"),
            sequence=EventSequence(sequence),
        ),
    )


def _result(turn_id: TurnId) -> TaskResult:
    return TaskResult(
        task_id=TaskId("task-1"),
        session_id=SessionId("session-1"),
        turn_id=turn_id,
        snapshot_revision=StateRevision(1),
        effect=TaskEffect(effect_type="answer", payload="accepted"),
    )


def _request(
    *,
    task_id: str,
    key: str,
    snapshot: SessionSnapshot,
) -> TaskRequest:
    assert snapshot.active_turn_id is not None
    return TaskRequest(
        task_id=TaskId(task_id),
        session_id=SessionId("session-1"),
        turn_id=snapshot.active_turn_id,
        parent_task_id=None,
        deadline_ms=TaskDeadlineMs(200),
        snapshot_revision=snapshot.revision,
        idempotency_key=IdempotencyKey(key),
        kind=TaskKind.INTERACTIVE,
    )


def _data_snapshot(*, memory: int) -> TaskStateSnapshot:
    return TaskStateSnapshot(
        memory_revision=MemoryRevision(memory),
        context_generation=ContextGeneration(0),
        profile_revision=ProfileRevision(0),
        consent_revision=ConsentRevision(0),
        corpus_revision=CorpusRevision(0),
        index_revision=IndexRevision(0),
    )
