"""Canonical, media-bound caption timeline commands for the future frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from orchestrator.response_contracts import CueParseResult

_REVEAL_UNIT_MS: Final = 180


@dataclass(frozen=True, slots=True)
class CaptionTimelineCommand:
    timeline_id: str
    marked_text: str
    audio_stream_id: str
    cancellation_epoch: int
    start_rtp_timestamp: int
    reveal_unit_ms: int = _REVEAL_UNIT_MS
    marker_grammar: str = "inline-cue/v1"

    @classmethod
    def from_cues(
        cls,
        *,
        timeline_id: str,
        parsed: CueParseResult,
        audio_stream_id: str,
        cancellation_epoch: int,
        start_rtp_timestamp: int,
    ) -> CaptionTimelineCommand:
        return cls(
            timeline_id=timeline_id,
            marked_text=parsed.marked_text,
            audio_stream_id=audio_stream_id,
            cancellation_epoch=cancellation_epoch,
            start_rtp_timestamp=start_rtp_timestamp,
        )

    def payload(self) -> dict[str, object]:
        return {
            "timeline_id": self.timeline_id,
            "marked_text": self.marked_text,
            "audio_stream_id": self.audio_stream_id,
            "cancellation_epoch": self.cancellation_epoch,
            "start_rtp_timestamp": self.start_rtp_timestamp,
            "reveal_unit_ms": self.reveal_unit_ms,
            "marker_grammar": self.marker_grammar,
        }


@dataclass(frozen=True, slots=True)
class CaptionTimelineCancel:
    timeline_id: str
    audio_stream_id: str
    cancellation_epoch: int
    reason: str

    def payload(self) -> dict[str, object]:
        return {
            "timeline_id": self.timeline_id,
            "audio_stream_id": self.audio_stream_id,
            "cancellation_epoch": self.cancellation_epoch,
            "reason": self.reason,
        }
