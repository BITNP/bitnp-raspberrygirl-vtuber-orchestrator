"""Local Orchestrator turn-pipeline contracts."""

from dataclasses import dataclass
from typing import Literal

from orchestrator.ids import SegmentId, TurnId


@dataclass(frozen=True, slots=True)
class CommentAudienceEvent:
    """Normalized comments audience.input payload consumed by Orchestrator."""

    platform: str
    source: str
    user: str
    text: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class ASRAudienceEvent:
    """Normalized ASR final payload consumed by Orchestrator."""

    text: str
    received_at_ms: int
    segment_id: str
    seq: int


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Deterministic turn-pipeline ID and queue settings."""

    queue_capacity: int
    turn_id_prefix: str
    segment_id_prefix: str


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Audio metadata from Orchestrator-owned provider synthesis."""

    sample_rate: int
    channels: int
    codec: str
    duration_ms: int
    byte_length: int


@dataclass(frozen=True, slots=True)
class MediaStreamCommand:
    """Orchestrator-local audio command aligned to a target RTP stream."""

    turn_id: TurnId
    segment_id: SegmentId
    stream_id: str
    audio: AudioMetadata
    start_at_ms: int
    event_type: Literal["media.stream.command"] = "media.stream.command"


@dataclass(frozen=True, slots=True)
class MediaStreamState:
    """Sound-service RTP playback state for an Orchestrator stream."""

    turn_id: TurnId
    segment_id: SegmentId
    stream_id: str
    state: Literal["queued", "playing", "finished", "cancelled"]
    playback_position_ms: int
    event_type: Literal["media.stream.state"] = "media.stream.state"


@dataclass(frozen=True, slots=True)
class VtuberCaptionCommand:
    """Orchestrator command registering visible caption text for a segment."""

    turn_id: TurnId
    segment_id: SegmentId
    text: str
    start_at_ms: int = 0
    event_type: Literal["vtuber.caption.command"] = "vtuber.caption.command"


@dataclass(frozen=True, slots=True)
class VtuberActionCommand:
    """Orchestrator command requesting a segment-scoped avatar action."""

    turn_id: TurnId
    segment_id: SegmentId
    action: str
    start_at_ms: int = 0
    event_type: Literal["vtuber.action.command"] = "vtuber.action.command"


@dataclass(frozen=True, slots=True)
class VtuberExpressionCommand:
    """Orchestrator command requesting a timed avatar expression."""

    turn_id: TurnId
    segment_id: SegmentId
    expression: str
    start_at_ms: int = 0
    event_type: Literal["vtuber.expression.command"] = "vtuber.expression.command"


@dataclass(frozen=True, slots=True)
class VtuberSceneCommand:
    """Orchestrator command requesting a lecture scene or slide state."""

    turn_id: TurnId
    segment_id: SegmentId
    scene: str
    slide_id: str
    slide_title: str
    slide_page: int = 1
    start_at_ms: int = 0
    event_type: Literal["vtuber.scene.command"] = "vtuber.scene.command"


@dataclass(frozen=True, slots=True)
class MockSynthesisResult:
    """Completed local synthesis with deterministic frontend controls."""

    turn_id: TurnId
    segment_id: SegmentId
    audio: AudioMetadata | None
    expression: str
    action: str
    scene: str
    slide_page: int
    offset_samples: int = 0


@dataclass(frozen=True, slots=True)
class SynthesisCueResult:
    """RTP-relative media and frontend cues for one completed synthesis."""

    media: MediaStreamCommand | None
    caption: VtuberCaptionCommand
    expression: VtuberExpressionCommand
    action: VtuberActionCommand
    scene: VtuberSceneCommand


@dataclass(frozen=True, slots=True)
class CancelCommand:
    """Orchestrator command cancelling one target's work for a segment."""

    turn_id: TurnId
    segment_id: SegmentId
    target: Literal["llm", "media_stream", "frontend"]
    reason: str
    event_type: Literal["cancel"] = "cancel"


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Result of routing one audience input through mode policy and LLM."""

    turn_id: TurnId
    segment_id: SegmentId
    answer_text: str
    used_fallback: bool


type AudienceEvent = CommentAudienceEvent | ASRAudienceEvent
