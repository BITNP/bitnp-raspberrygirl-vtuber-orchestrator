"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from orchestrator.comment_ingress import (
    AuthenticatedCommentIngress,
    CommentAccessToken,
    CommentIngressConfig,
    CommentIngressRejection,
    CommentTokenValue,
)
from orchestrator.ids import SessionId
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.sessions import SessionScheduler


def test_authenticated_comment_is_queued_then_reduced_once() -> None:
    # Given: an authenticated bounded ingress for one session.

    """函数契约说明.

    功能: 验证 authenticated comment is
    queued then reduced once
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    scheduler = SessionScheduler(
        session_id=SessionId("session-1"), turn_id_prefix="turn"
    )

    ingress = AuthenticatedCommentIngress(
        SessionInteractionIngress.create(scheduler),
        CommentIngressConfig(
            CommentAccessToken(CommentTokenValue("comments-token"), 1_000), 2, 512, 4
        ),
    )

    # When: a trusted Comments envelope is received and the scheduler drains it.

    received = ingress.receive(
        _comment(sequence=1), "Bearer comments-token", now_ms=100
    )

    reduced = ingress.reduce_next()

    # Then: exactly one scheduler event materializes from the typed proposal.

    assert received.accepted is True

    assert reduced.accepted is True

    assert [event.turn_id for event in scheduler.event_history] == ["turn-0001"]


def test_invalid_expired_replayed_and_backpressured_comments_have_no_effect() -> None:
    # Given: a small queue and a token that expires at a deterministic instant.

    """函数契约说明.

    功能: 验证 invalid expired replayed and
    backpressured comments have no
    effect 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    scheduler = SessionScheduler(
        session_id=SessionId("session-1"), turn_id_prefix="turn"
    )

    ingress = AuthenticatedCommentIngress(
        SessionInteractionIngress.create(scheduler),
        CommentIngressConfig(
            CommentAccessToken(CommentTokenValue("comments-token"), 100), 1, 512, 1
        ),
    )

    # When: unauthorized, expired, oversized, replayed, and saturated frames arrive.

    unauthorized = ingress.receive(_comment(sequence=1), "Bearer wrong", now_ms=10)

    expired = ingress.receive(_comment(sequence=1), "Bearer comments-token", now_ms=101)

    oversized = ingress.receive(
        _comment(sequence=1, text="x" * 513), "Bearer comments-token", now_ms=10
    )

    accepted = ingress.receive(_comment(sequence=1), "Bearer comments-token", now_ms=10)

    replayed = ingress.receive(_comment(sequence=1), "Bearer comments-token", now_ms=10)

    saturated = ingress.receive(
        _comment(sequence=2), "Bearer comments-token", now_ms=10
    )

    # Then: only the queued valid frame could reach the scheduler.

    assert unauthorized.rejection is CommentIngressRejection.UNAUTHORIZED

    assert expired.rejection is CommentIngressRejection.EXPIRED_CREDENTIAL

    assert oversized.rejection is CommentIngressRejection.PAYLOAD_TOO_LARGE

    assert accepted.accepted is True

    assert replayed.rejection is CommentIngressRejection.DUPLICATE

    assert saturated.rejection is CommentIngressRejection.BACKPRESSURE

    assert scheduler.event_history == ()


def _comment(sequence: int, text: str = "解释量化") -> str:
    """函数契约说明.

    功能: 执行 _comment 的同步逻辑,并维持签名契约。
    参数: sequence: int。 必填。 text: str。
    可省略。
    契约: 同步调用。 返回 `str`。
    """

    return (
        '{"event_type":"audience.input","source":"comments","trace_id":"trace-1",'
        f'"session_id":"session-1","seq":{sequence},"data":{{"text":"{text}"}}}}'
    )
