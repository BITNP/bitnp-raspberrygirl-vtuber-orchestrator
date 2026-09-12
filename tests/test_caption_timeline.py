import pytest

from orchestrator.caption_timeline import CaptionTimelineCommand
from orchestrator.response_contracts import parse_inline_cues


@pytest.mark.parametrize("expression", ["nod", "shake_head", "wink"])
def test_caption_timeline_preserves_validated_markers_but_not_tts_text(
    expression: str,
) -> None:
    marked_text = f'请看<action name="hello"/>这里<expression name="{expression}"/>。'
    parsed = parse_inline_cues(
        marked_text,
        allowed_actions=frozenset({"hello"}),
        allowed_expressions=frozenset({"nod", "shake_head", "wink"}),
    )
    timeline = CaptionTimelineCommand.from_cues(
        timeline_id="timeline-1",
        parsed=parsed,
        audio_stream_id="agent-turn-1",
        cancellation_epoch=3,
        start_rtp_timestamp=96000,
    )
    assert parsed.spoken_text == "请看这里。"
    assert timeline.payload()["marked_text"] == marked_text
    assert timeline.payload()["marker_grammar"] == "inline-cue/v1"
