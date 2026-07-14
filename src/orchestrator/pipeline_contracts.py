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
    """Audio metadata mirrored from TTS and routed to sound."""

    sample_rate: int
    channels: int
    codec: str
    duration_ms: int
    byte_length: int


@dataclass(frozen=True, slots=True)
class TTSCommand:
    """Orchestrator command requesting speech synthesis for one segment."""

    turn_id: TurnId
    segment_id: SegmentId
    request_id: str
    text: str
    voice: str
    event_type: Literal["tts.request"] = "tts.request"


@dataclass(frozen=True, slots=True)
class TTSChunkEvent:
    """TTS chunk observation accepted by Orchestrator."""

    turn_id: TurnId
    segment_id: SegmentId
    chunk_id: str
    audio: AudioMetadata
    uri: str
    event_type: Literal["tts.chunk"] = "tts.chunk"


@dataclass(frozen=True, slots=True)
class TTSDoneEvent:
    """TTS completion observation accepted by Orchestrator."""

    turn_id: TurnId
    segment_id: SegmentId
    event_type: Literal["tts.done"] = "tts.done"


@dataclass(frozen=True, slots=True)
class SoundPlayCommand:
    """Orchestrator command requesting playback for one audio chunk."""

    turn_id: TurnId
    segment_id: SegmentId
    command_id: str
    uri: str
    audio: AudioMetadata
    event_type: Literal["sound.play.command"] = "sound.play.command"


@dataclass(frozen=True, slots=True)
class VtuberCaptionCommand:
    """Orchestrator command registering visible caption text for a segment."""

    turn_id: TurnId
    segment_id: SegmentId
    text: str
    event_type: Literal["vtuber.caption.command"] = "vtuber.caption.command"


@dataclass(frozen=True, slots=True)
class VtuberActionCommand:
    """Orchestrator command requesting a segment-scoped avatar action."""

    turn_id: TurnId
    segment_id: SegmentId
    action: Literal["speak"]
    event_type: Literal["vtuber.action.command"] = "vtuber.action.command"


@dataclass(frozen=True, slots=True)
class VtuberSegmentCommands:
    """Commands emitted after TTS completion makes a segment presentable."""

    caption: VtuberCaptionCommand
    action: VtuberActionCommand


@dataclass(frozen=True, slots=True)
class CancelCommand:
    """Orchestrator command cancelling one target's work for a segment."""

    turn_id: TurnId
    segment_id: SegmentId
    target: Literal["llm", "tts", "sound", "vtuber"]
    reason: str
    event_type: Literal["cancel"] = "cancel"


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Result of routing one audience input through mode policy and LLM."""

    turn_id: TurnId
    segment_id: SegmentId
    tts_command: TTSCommand
    used_fallback: bool


type AudienceEvent = CommentAudienceEvent | ASRAudienceEvent
type TTSObservation = TTSChunkEvent | TTSDoneEvent
