
from dataclasses import dataclass
from typing import Literal

from orchestrator.ids import SegmentId, TurnId


@dataclass(frozen=True, slots=True)
class CommentAudienceEvent:

    platform: str

    source: str

    user: str

    text: str

    timestamp: str


@dataclass(frozen=True, slots=True)
class ASRAudienceEvent:

    text: str

    received_at_ms: int

    segment_id: str

    seq: int

    stream_id: str | None = None

    input_epoch: int | None = None

    rtp_start_timestamp: int | None = None

    rtp_end_timestamp: int | None = None


@dataclass(frozen=True, slots=True)
class PipelineConfig:

    queue_capacity: int

    turn_id_prefix: str

    segment_id_prefix: str


@dataclass(frozen=True, slots=True)
class AudioMetadata:

    sample_rate: int

    channels: int

    codec: str

    duration_ms: int

    byte_length: int


@dataclass(frozen=True, slots=True)
class MediaStreamCommand:

    turn_id: TurnId

    segment_id: SegmentId

    stream_id: str

    audio: AudioMetadata

    start_at_ms: int

    event_type: Literal["media.stream.command"] = "media.stream.command"


@dataclass(frozen=True, slots=True)
class MediaStreamState:

    turn_id: TurnId

    segment_id: SegmentId

    stream_id: str

    state: Literal["queued", "playing", "finished", "cancelled"]

    playback_position_ms: int

    event_type: Literal["media.stream.state"] = "media.stream.state"


@dataclass(frozen=True, slots=True)
class VtuberCaptionCommand:

    turn_id: TurnId

    segment_id: SegmentId

    text: str

    start_at_ms: int = 0

    event_type: Literal["vtuber.caption.command"] = "vtuber.caption.command"


@dataclass(frozen=True, slots=True)
class VtuberActionCommand:

    turn_id: TurnId

    segment_id: SegmentId

    action: str

    start_at_ms: int = 0

    event_type: Literal["vtuber.action.command"] = "vtuber.action.command"


@dataclass(frozen=True, slots=True)
class VtuberExpressionCommand:

    turn_id: TurnId

    segment_id: SegmentId

    expression: str

    start_at_ms: int = 0

    event_type: Literal["vtuber.expression.command"] = "vtuber.expression.command"


@dataclass(frozen=True, slots=True)
class VtuberSceneCommand:

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

    media: MediaStreamCommand | None

    caption: VtuberCaptionCommand

    expression: VtuberExpressionCommand

    action: VtuberActionCommand

    scene: VtuberSceneCommand


@dataclass(frozen=True, slots=True)
class CancelCommand:

    turn_id: TurnId

    segment_id: SegmentId

    target: Literal["llm", "media_stream", "frontend"]

    reason: str

    event_type: Literal["cancel"] = "cancel"


@dataclass(frozen=True, slots=True)
class TurnResult:

    turn_id: TurnId

    segment_id: SegmentId

    answer_text: str

    used_fallback: bool


type AudienceEvent = CommentAudienceEvent | ASRAudienceEvent
