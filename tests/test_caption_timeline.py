from orchestrator.caption_timeline import CaptionTimelineCommand
from orchestrator.response_contracts import parse_inline_cues


def test_caption_timeline_preserves_validated_markers_but_not_tts_text() -> None:
    parsed = parse_inline_cues(
        '请看<action name="hello"/>这里。',
        allowed_actions=frozenset({"hello"}),
        allowed_expressions=frozenset(),
    )
    timeline = CaptionTimelineCommand.from_cues(
        timeline_id="timeline-1",
        parsed=parsed,
        audio_stream_id="agent-turn-1",
        cancellation_epoch=3,
        start_rtp_timestamp=96000,
    )
    assert parsed.spoken_text == "请看这里。"
    assert timeline.payload()["marked_text"] == '请看<action name="hello"/>这里。'
    assert timeline.payload()["marker_grammar"] == "inline-cue/v1"
