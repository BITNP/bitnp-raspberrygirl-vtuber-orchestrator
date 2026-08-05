"""Model-neutral immutable inputs and trusted tool requests.

These contracts belong to the minimal response pipeline and its trusted
runtime adapters.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

MAX_AUDIENCE_TEXT_LENGTH: Final = 4_000


class AudienceInputError(ValueError):
    """Audience input has invalid correlation or text."""


class AudienceSource(StrEnum):
    ASR = "asr"
    COMMENT = "comment"


class GateDecision(StrEnum):
    ACCEPT = "accept"
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class AudienceInput:
    session_id: str
    trace_id: str
    sequence: int
    source: AudienceSource
    received_at_ms: int
    text: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.trace_id or self.sequence < 0:
            raise AudienceInputError
        if not self.text.strip() or len(self.text) > MAX_AUDIENCE_TEXT_LENGTH:
            raise AudienceInputError


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    task_id: str
    kind: str
    lane: str
    status: str
    deadline_ms: int
    owner_turn_id: str
    cancellation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PlaybackSnapshot:
    status: str = "idle"
    position_ms: int = 0
    active_audio_id: str | None = None
    replacement_audio_id: str | None = None
    replacement_first_frame_ready: bool = False
    flush_accepted: bool = False


@dataclass(frozen=True, slots=True)
class BrainStateSnapshot:
    session_id: str
    turn_id: str
    revision: int
    cancellation_epoch: int
    input: AudienceInput
    context_summary: str
    recent_context: tuple[str, ...]
    memory_markdown: str
    capabilities: frozenset[str]
    tasks: tuple[TaskSnapshot, ...] = ()
    playback: PlaybackSnapshot = PlaybackSnapshot()
    frontend_caption: str = ""
    frontend_animation: str | None = None
    ppt_deck_id: str | None = None
    ppt_page: int | None = None
    context_revision: int = 0
    memory_revision: int = 0
    context_budget: int = 0
    compaction_required: bool = False
    knowledge_references: tuple[str, ...] = ()
    mcp_allowlist: frozenset[str] = frozenset()
    speaker_profile_id: str | None = None
    speaker_preferred_name: str | None = None
    speaker_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ToolRequest:
    kind: str
    name: str
    arguments: dict[str, object]
