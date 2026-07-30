"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import asdict

import pytest

from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.sessions import (
    EventCorrelation,
    EventSequence,
    SchedulerEvent,
    SessionScheduler,
    StartTurn,
    StateRevision,
    TransitionAccepted,
    TransitionRejected,
    TransitionRejection,
)


def test_start_turn_rejects_a_stale_snapshot_revision() -> None:
    # Given: a scheduler whose initial revision has already accepted one event.

    """函数契约说明.

    功能: 验证 start turn rejects a stale
    snapshot revision 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    scheduler = SessionScheduler(
        session_id=SessionId("session-local"),
        turn_id_prefix="turn",
    )

    first = scheduler.apply(
        StartTurn(
            expected_revision=StateRevision(0),
            event=SchedulerEvent(
                event_type="audience.input",
                correlation=EventCorrelation(
                    trace_id=TraceId("trace-1"),
                    session_id=SessionId("session-local"),
                    sequence=EventSequence(1),
                ),
            ),
        ),
    )

    # When: a concurrent transition still targets the old state revision.

    stale = scheduler.apply(
        StartTurn(
            expected_revision=StateRevision(0),
            event=SchedulerEvent(
                event_type="audience.input",
                correlation=EventCorrelation(
                    trace_id=TraceId("trace-2"),
                    session_id=SessionId("session-local"),
                    sequence=EventSequence(2),
                ),
            ),
        ),
    )

    # Then: the stale request is rejected without changing the accepted snapshot.

    match stale:
        case TransitionRejected(
            snapshot=snapshot,
            reason=TransitionRejection.STALE_REVISION,
        ):
            assert snapshot == first.snapshot

        case TransitionAccepted():
            pytest.fail("stale transition was accepted")

        case TransitionRejected():
            pytest.fail("stale transition was rejected for the wrong reason")


def test_accepted_events_preserve_correlation_and_allocate_monotonic_turns() -> None:
    # Given: an empty local scheduler and two correlated audience events.

    """函数契约说明.

    功能: 验证 accepted events preserve
    correlation and allocate monotonic
    turns 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    session_id = SessionId("session-local")

    scheduler = SessionScheduler(session_id=session_id, turn_id_prefix="turn")

    first = scheduler.apply(
        StartTurn(
            expected_revision=StateRevision(0),
            event=SchedulerEvent(
                event_type="audience.input",
                correlation=EventCorrelation(
                    trace_id=TraceId("trace-1"),
                    session_id=session_id,
                    sequence=EventSequence(1),
                ),
            ),
        ),
    )

    # When: the second event targets the revision committed by the first event.

    match first:
        case TransitionAccepted() as accepted_first:
            pass

        case TransitionRejected():
            pytest.fail("initial transition was rejected")

    second = scheduler.apply(
        StartTurn(
            expected_revision=accepted_first.snapshot.revision,
            event=SchedulerEvent(
                event_type="audience.input",
                correlation=EventCorrelation(
                    trace_id=TraceId("trace-2"),
                    session_id=session_id,
                    sequence=EventSequence(2),
                ),
            ),
        ),
    )

    # Then: immutable history retains correlation and serialized results expose it.

    match second:
        case TransitionAccepted() as accepted_second:
            pass

        case TransitionRejected():
            pytest.fail("current-revision transition was rejected")

    assert accepted_first.snapshot.active_turn_id == TurnId("turn-0001")

    assert accepted_second.snapshot.active_turn_id == TurnId("turn-0002")

    assert accepted_second.snapshot.revision == StateRevision(2)

    assert scheduler.event_history == (
        accepted_first.accepted_event,
        accepted_second.accepted_event,
    )

    assert asdict(accepted_second) == {
        "snapshot": {
            "session_id": "session-local",
            "revision": 2,
            "active_turn_id": "turn-0002",
        },
        "accepted_event": {
            "event": {
                "event_type": "audience.input",
                "correlation": {
                    "trace_id": "trace-2",
                    "session_id": "session-local",
                    "sequence": 2,
                },
            },
            "turn_id": "turn-0002",
        },
    }


def test_start_turn_rejects_events_for_another_session() -> None:
    # Given: a scheduler with an empty session-local history.

    """函数契约说明.

    功能: 验证 start turn rejects events for
    another session 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    scheduler = SessionScheduler(
        session_id=SessionId("session-local"),
        turn_id_prefix="turn",
    )

    initial_snapshot = scheduler.snapshot

    # When: an event carries a correlation for a different session.

    result = scheduler.apply(
        StartTurn(
            expected_revision=StateRevision(0),
            event=SchedulerEvent(
                event_type="audience.input",
                correlation=EventCorrelation(
                    trace_id=TraceId("trace-1"),
                    session_id=SessionId("another-session"),
                    sequence=EventSequence(1),
                ),
            ),
        ),
    )

    # Then: no snapshot, turn identity, or history entry can be created.

    match result:
        case TransitionRejected(
            snapshot=snapshot,
            reason=TransitionRejection.SESSION_MISMATCH,
        ):
            assert snapshot == initial_snapshot

        case TransitionAccepted():
            pytest.fail("cross-session event was accepted")

        case TransitionRejected():
            pytest.fail("cross-session event was rejected for the wrong reason")

    assert scheduler.event_history == ()
