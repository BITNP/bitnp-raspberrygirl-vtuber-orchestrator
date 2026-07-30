"""模块契约说明.

职责: 提供 orchestrator.comment_ingress
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import hmac
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import NewType

from orchestrator.interaction_ingress import (
    SessionInteractionIngress,
    parse_comment_proposal,
)
from orchestrator.interactions import CommentProposal, InteractionAccepted
from orchestrator.sessions import EventCorrelation

CommentTokenValue = NewType("CommentTokenValue", str)


@dataclass(frozen=True, slots=True)
class CommentAccessToken:
    """类契约说明.

    职责: 保存 CommentAccessToken
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: value、expires_at_ms。
    """

    value: CommentTokenValue

    expires_at_ms: int


@dataclass(frozen=True, slots=True)
class CommentIngressConfig:
    """类契约说明.

    职责: 保存 CommentIngressConfig
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: token、replay_window、max_payl
    oad_bytes、max_pending。
    """

    token: CommentAccessToken | None

    replay_window: int

    max_payload_bytes: int

    max_pending: int


@unique
class CommentIngressRejection(StrEnum):
    """类契约说明.

    职责: 定义 CommentIngressRejection
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    UNAUTHORIZED = "unauthorized"

    EXPIRED_CREDENTIAL = "credential_expired"

    MALFORMED = "malformed"

    PAYLOAD_TOO_LARGE = "payload_too_large"

    DUPLICATE = "duplicate"

    STALE_REPLAY = "stale_replay"

    BACKPRESSURE = "backpressure"

    NO_PENDING = "no_pending"

    REDUCER_REJECTED = "reducer_rejected"


@dataclass(frozen=True, slots=True)
class CommentIngressReceipt:
    """类契约说明.

    职责: 保存 CommentIngressReceipt
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: accepted、rejection。
    """

    accepted: bool

    rejection: CommentIngressRejection | None = None


@dataclass(slots=True)
class AuthenticatedCommentIngress:
    """类契约说明.

    职责: 保存 AuthenticatedCommentIngress
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: interactions、config、_pending
    、_seen、_highest_sequence。 方法: receiv
    e、reduce_next、take_next、cancel_pendi
    ng、_authorization_rejection、_preflig
    ht。
    """

    interactions: SessionInteractionIngress

    config: CommentIngressConfig

    _pending: deque[CommentProposal] = field(default_factory=deque)

    _seen: set[EventCorrelation] = field(default_factory=set)

    _highest_sequence: int | None = None

    def receive(
        self, raw_message: str, authorization: str | None, *, now_ms: int
    ) -> CommentIngressReceipt:
        """函数契约说明.

        功能: 执行 receive 的同步逻辑,并协调
        _preflight, isinstance, add,
        max。
        参数: self 表示当前实例。 raw_message:
        str。 必填。 authorization: str |
        None。 必填。 now_ms: int。 必填。
        契约: 同步调用。 返回
        `CommentIngressReceipt`。
        """
        preflight = self._preflight(raw_message, authorization, now_ms)

        if isinstance(preflight, CommentIngressReceipt):
            return preflight

        proposal = preflight

        correlation = proposal.correlation

        if len(self._pending) == self.config.max_pending:
            return CommentIngressReceipt(
                accepted=False, rejection=CommentIngressRejection.BACKPRESSURE
            )

        self._seen.add(correlation)

        self._highest_sequence = max(
            correlation.sequence, self._highest_sequence or correlation.sequence
        )

        self._discard_expired_replays()

        self._pending.append(proposal)

        return CommentIngressReceipt(accepted=True)

    def reduce_next(self) -> CommentIngressReceipt:
        """函数契约说明.

        功能: 执行 reduce_next 的同步逻辑,并协调
        popleft, receive_comment,
        isinstance,
        CommentIngressReceipt。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `CommentIngressReceipt`。
        """
        if not self._pending:
            return CommentIngressReceipt(
                accepted=False, rejection=CommentIngressRejection.NO_PENDING
            )

        proposal = self._pending.popleft()

        outcome = self.interactions.receive_comment(
            text=proposal.text, correlation=proposal.correlation
        )

        if isinstance(outcome, InteractionAccepted):
            return CommentIngressReceipt(accepted=True)

        return CommentIngressReceipt(
            accepted=False, rejection=CommentIngressRejection.REDUCER_REJECTED
        )

    def take_next(self) -> CommentProposal | None:
        """函数契约说明.

        功能: 执行 take_next 的同步逻辑,并协调
        popleft。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `CommentProposal |
        None`。
        """
        if not self._pending:
            return None

        return self._pending.popleft()

    def cancel_pending(self) -> None:
        """函数契约说明.

        功能: 执行 cancel_pending 的同步逻辑,并协调
        clear。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self._pending.clear()

    def _authorization_rejection(
        self, authorization: str | None, now_ms: int
    ) -> CommentIngressRejection | None:
        """函数契约说明.

        功能: 执行 _authorization_rejection
        的同步逻辑,并协调 compare_digest,
        startswith, removeprefix。
        参数: self 表示当前实例。 authorization:
        str | None。 必填。 now_ms: int。 必填。
        契约: 同步调用。 返回
        `CommentIngressRejection |
        None`。
        """
        token = self.config.token

        if token is None:
            return (
                None if authorization is None else CommentIngressRejection.UNAUTHORIZED
            )

        if now_ms > token.expires_at_ms:
            return CommentIngressRejection.EXPIRED_CREDENTIAL

        prefix = "Bearer "

        if authorization is None or not authorization.startswith(prefix):
            return CommentIngressRejection.UNAUTHORIZED

        if not hmac.compare_digest(authorization.removeprefix(prefix), token.value):
            return CommentIngressRejection.UNAUTHORIZED

        return None

    def _preflight(
        self, raw_message: str, authorization: str | None, now_ms: int
    ) -> CommentProposal | CommentIngressReceipt:
        """函数契约说明.

        功能: 执行 _preflight 的同步逻辑,并协调
        _authorization_rejection,
        parse_comment_proposal,
        _outside_replay_window,
        CommentIngressReceipt。
        参数: self 表示当前实例。 raw_message:
        str。 必填。 authorization: str |
        None。 必填。 now_ms: int。 必填。
        契约: 同步调用。 返回 `CommentProposal |
        CommentIngressReceipt`。
        """
        rejection = self._authorization_rejection(authorization, now_ms)

        if rejection is not None:
            return CommentIngressReceipt(accepted=False, rejection=rejection)

        if len(raw_message.encode()) > self.config.max_payload_bytes:
            return CommentIngressReceipt(
                accepted=False, rejection=CommentIngressRejection.PAYLOAD_TOO_LARGE
            )

        proposal = parse_comment_proposal(raw_message)

        if proposal is None:
            return CommentIngressReceipt(
                accepted=False, rejection=CommentIngressRejection.MALFORMED
            )

        correlation = proposal.correlation

        if correlation in self._seen:
            return CommentIngressReceipt(
                accepted=False, rejection=CommentIngressRejection.DUPLICATE
            )

        if self._outside_replay_window(correlation.sequence):
            return CommentIngressReceipt(
                accepted=False, rejection=CommentIngressRejection.STALE_REPLAY
            )

        return proposal

    def _outside_replay_window(self, sequence: int) -> bool:
        """函数契约说明.

        功能: 执行 _outside_replay_window
        的同步逻辑,并产出 highest。
        参数: self 表示当前实例。 sequence: int。
        必填。
        契约: 同步调用。 返回 `bool`。
        """
        highest = self._highest_sequence

        return highest is not None and sequence < highest - self.config.replay_window

    def _discard_expired_replays(self) -> None:
        """函数契约说明.

        功能: 执行 _discard_expired_replays
        的同步逻辑,并产出 highest, _seen。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        highest = self._highest_sequence

        if highest is None:
            return

        self._seen = {
            correlation
            for correlation in self._seen
            if correlation.sequence >= highest - self.config.replay_window
        }
