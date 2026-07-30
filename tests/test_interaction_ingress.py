"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from orchestrator.ids import SessionId, TraceId
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.interactions import InteractionAccepted
from orchestrator.sessions import EventCorrelation, EventSequence, SessionScheduler


def test_production_ingress_routes_comment_through_scheduler_reducer() -> None:
    # Given: one production ingress composed with its session scheduler controls.

    """函数契约说明.

    功能: 验证 production ingress routes
    comment through scheduler reducer
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )

    ingress = SessionInteractionIngress.create(scheduler)

    # When: comments ingress submits a normalized, correlated audience proposal.

    outcome = ingress.receive_comment(
        text="解释量化",
        correlation=EventCorrelation(
            trace_id=TraceId("trace-1"),
            session_id=SessionId("session-1"),
            sequence=EventSequence(1),
        ),
    )

    # Then: the real reducer opens the scheduler's first monotonic turn.

    assert outcome == InteractionAccepted(turn_id="turn-0001")

    assert scheduler.event_history[0].event.event_type == "audience.input"


def test_control_envelope_routes_comments_but_leaves_media_for_transport() -> None:
    # Given: a live ingress sharing one session scheduler with transport.

    """函数契约说明.

    功能: 验证 control envelope routes
    comments but leaves media for
    transport 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )

    ingress = SessionInteractionIngress.create(scheduler)

    # When: a comments envelope and a media envelope reach the common listener.

    comment = (
        '{"event_type":"audience.input","source":"comments","trace_id":"trace-1",'
        '"session_id":"session-1","seq":1,"data":{"text":"解释量化"}}'
    )

    media = '{"event_type":"media.rtp.source.ready","source":"mic","data":{}}'

    # Then: only the typed comments proposal opens a reducer-controlled turn.

    assert ingress.receive_control(comment) is True

    assert ingress.receive_control(media) is False

    assert scheduler.snapshot.active_turn_id is not None


def test_duplicate_correlated_comment_opens_only_one_turn() -> None:
    # Given: one live comments frame and its exact transport replay.

    """函数契约说明.

    功能: 验证 duplicate correlated comment
    opens only one turn 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )

    ingress = SessionInteractionIngress.create(scheduler)

    comment = (
        '{"event_type":"audience.input","source":"comments","trace_id":"trace-1",'
        '"session_id":"session-1","seq":1,"data":{"text":"解释量化"}}'
    )

    # When: the same trace/session/sequence envelope is received twice.

    first = ingress.receive_control(comment)

    duplicate = ingress.receive_control(comment)

    # Then: exactly one reducer-controlled turn is materialized.

    assert (first, duplicate) == (True, True)

    assert [event.turn_id for event in scheduler.event_history] == ["turn-0001"]
