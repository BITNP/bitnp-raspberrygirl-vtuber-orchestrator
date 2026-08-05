from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from orchestrator.config import TrustedLanToken

MAX_CONTROL_FRAME_BYTES: Final = 64 * 1024
MAX_SESSION_ID_LENGTH: Final = 128
SESSION_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class PeerRole(StrEnum):
    MIC = "mic"
    SOUND = "sound"
    COMMENTS = "comments"
    FRONTEND = "frontend"
    OPERATOR = "operator"


ROLE_SOURCES: Final = {
    PeerRole.MIC: "mic",
    PeerRole.SOUND: "sound",
    PeerRole.COMMENTS: "comments",
    PeerRole.FRONTEND: "frontend",
    PeerRole.OPERATOR: "orchestrator",
}

ROLE_EVENTS: Final = {
    PeerRole.MIC: frozenset(
        {"mic.input.register", "asr.partial", "asr.final", "voice.evidence"}
    ),
    PeerRole.SOUND: frozenset(
        {
            "media.rtp.sink.register",
            "media.rtp.sink.ready",
            "media.stream.state",
            "media.stream.flush.ack",
        }
    ),
    PeerRole.COMMENTS: frozenset({"audience.input"}),
    PeerRole.FRONTEND: frozenset(
        {"frontend.register", "presentation.result", "action.result"}
    ),
    PeerRole.OPERATOR: frozenset(
        {"session.end.command", "profile.enroll.command", "profile.revoke.command"}
    ),
}

SESSION_ADMISSION_EVENTS: Final = frozenset(
    {"mic.input.register", "media.rtp.sink.register", "frontend.register"}
)


@dataclass(frozen=True, slots=True)
class RoleTokens:
    mic: TrustedLanToken | None = None
    sound: TrustedLanToken | None = None
    comments: TrustedLanToken | None = None
    frontend: TrustedLanToken | None = None
    operator: TrustedLanToken | None = None

    def items(self) -> tuple[tuple[PeerRole, TrustedLanToken | None], ...]:
        return (
            (PeerRole.MIC, self.mic),
            (PeerRole.SOUND, self.sound),
            (PeerRole.COMMENTS, self.comments),
            (PeerRole.FRONTEND, self.frontend),
            (PeerRole.OPERATOR, self.operator),
        )

    def resolve(self, authorization: str | None) -> PeerRole | None:
        if authorization is None or not authorization.startswith("Bearer "):
            return None
        candidate = authorization.removeprefix("Bearer ")
        for role, token in self.items():
            if token is not None and hmac.compare_digest(candidate, token):
                return role
        return None

    def validate_unique(self) -> bool:
        configured = [str(token) for _, token in self.items() if token is not None]
        return len(configured) == len(set(configured))


def valid_session_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= MAX_SESSION_ID_LENGTH
        and SESSION_ID_PATTERN.fullmatch(value) is not None
    )


def role_allows(role: PeerRole, source: object, event_type: object) -> bool:
    return source == ROLE_SOURCES[role] and event_type in ROLE_EVENTS[role]
