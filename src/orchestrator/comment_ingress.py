
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

    value: CommentTokenValue

    expires_at_ms: int


@dataclass(frozen=True, slots=True)
class CommentIngressConfig:

    token: CommentAccessToken | None

    replay_window: int

    max_payload_bytes: int

    max_pending: int


@unique
class CommentIngressRejection(StrEnum):

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

    accepted: bool

    rejection: CommentIngressRejection | None = None


@dataclass(slots=True)
class AuthenticatedCommentIngress:

    interactions: SessionInteractionIngress

    config: CommentIngressConfig

    _pending: deque[CommentProposal] = field(default_factory=deque)

    _seen: set[EventCorrelation] = field(default_factory=set)

    _highest_sequence: int | None = None

    def receive(
        self, raw_message: str, authorization: str | None, *, now_ms: int
    ) -> CommentIngressReceipt:
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
        if not self._pending:
            return None

        return self._pending.popleft()

    def cancel_pending(self) -> None:
        self._pending.clear()

    def _authorization_rejection(
        self, authorization: str | None, now_ms: int
    ) -> CommentIngressRejection | None:
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
        highest = self._highest_sequence

        return highest is not None and sequence < highest - self.config.replay_window

    def _discard_expired_replays(self) -> None:
        highest = self._highest_sequence

        if highest is None:
            return

        self._seen = {
            correlation
            for correlation in self._seen
            if correlation.sequence >= highest - self.config.replay_window
        }
