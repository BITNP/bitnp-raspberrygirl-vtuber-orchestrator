import asyncio

from orchestrator.caption_timeline import CaptionTimelineCancel, CaptionTimelineCommand
from orchestrator.frontend_effects import send_caption_timeline
from orchestrator.ids import SessionId, TurnId
from orchestrator.json_boundary import parse_json_value


def test_caption_timeline_keeps_validated_markers() -> None:
    sent: list[str] = []

    async def send(raw: str) -> None:
        sent.append(raw)

    async def run() -> None:
        await send_caption_timeline(
            send,
            CaptionTimelineCommand(
                timeline_id="timeline-1",
                marked_text='欢迎<action name="hello"/>大家',
                audio_stream_id="agent-turn-1",
                cancellation_epoch=2,
                start_rtp_timestamp=96000,
            ),
            SessionId("session-1"),
            TurnId("turn-1"),
        )

    asyncio.run(run())
    envelope = parse_json_value(sent[0])
    assert isinstance(envelope, dict)
    assert envelope["event_type"] == "vtuber.caption.timeline.command"
    assert isinstance(envelope["data"], dict)
    assert envelope["data"]["marked_text"] == '欢迎<action name="hello"/>大家'


def test_caption_timeline_cancel_keeps_media_correlation() -> None:
    sent: list[str] = []

    async def send(raw: str) -> None:
        sent.append(raw)

    async def run() -> None:
        await send_caption_timeline(
            send,
            CaptionTimelineCancel(
                timeline_id="timeline-1",
                audio_stream_id="agent-turn-1",
                cancellation_epoch=2,
                reason="replaced",
            ),
            SessionId("session-1"),
            TurnId("turn-1"),
        )

    asyncio.run(run())
    envelope = parse_json_value(sent[0])
    assert isinstance(envelope, dict)
    assert envelope["event_type"] == "vtuber.caption.timeline.cancel"
    assert envelope["data"] == {
        "timeline_id": "timeline-1",
        "audio_stream_id": "agent-turn-1",
        "cancellation_epoch": 2,
        "reason": "replaced",
    }
