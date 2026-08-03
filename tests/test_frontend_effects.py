import asyncio

from orchestrator.agent_pipeline import FrontendOperation
from orchestrator.caption_timeline import CaptionTimelineCommand
from orchestrator.frontend_effects import (
    FrontendEffectDispatcher,
    send_caption_timeline,
    send_frontend_operation,
)
from orchestrator.ids import SessionId, TurnId
from orchestrator.json_boundary import parse_json_value


def test_frontend_dispatcher_sends_accepted_caption() -> None:
    sent: list[str] = []

    async def sender(
        event_type: str,
        operation: FrontendOperation,
        session_id: SessionId,
        turn_id: TurnId,
    ) -> None:
        async def send(raw: str) -> None:
            sent.append(raw)

        await send_frontend_operation(
            send, event_type, operation, session_id, turn_id
        )

    async def run() -> None:
        FrontendEffectDispatcher(sender).dispatch_frontend(
            FrontendOperation("caption", "欢迎来到活动"),
            session_id=SessionId("session-1"),
            turn_id=TurnId("turn-1"),
        )
        await asyncio.sleep(0)

    asyncio.run(run())

    assert len(sent) == 1
    envelope = parse_json_value(sent[0])
    assert isinstance(envelope, dict)
    assert envelope["event_type"] == "vtuber.caption.command"
    assert isinstance(envelope["data"], dict)
    assert envelope["data"]["text"] == "欢迎来到活动"
    assert envelope["data"]["audio_stream_id"] == "agent-turn-1"
    assert envelope["segment_id"] == "agent-turn-1"


def test_frontend_dispatcher_uses_brain_validated_ppt_deck_id() -> None:
    sent: list[str] = []

    async def send(raw: str) -> None:
        sent.append(raw)

    async def run() -> None:
        await send_frontend_operation(
            send,
            "presentation.navigate.command",
            FrontendOperation("ppt.navigate", 3, deck_id="launch-deck"),
            SessionId("session-1"),
            TurnId("turn-1"),
        )

    asyncio.run(run())

    envelope = parse_json_value(sent[0])
    assert isinstance(envelope, dict)
    data = envelope["data"]
    assert isinstance(data, dict)
    assert isinstance(data["command_id"], str)
    assert data["deck_id"] == "launch-deck"
    assert data["page"] == 3


def test_caption_timeline_keeps_validated_markers() -> None:
    sent: list[str] = []

    async def send(raw: str) -> None:
        sent.append(raw)

    async def run() -> None:
        await send_caption_timeline(
            send,
            CaptionTimelineCommand(
                timeline_id="timeline-1",
                marked_text='欢迎<action name="wave"/>大家',
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
    assert envelope["data"]["marked_text"] == '欢迎<action name="wave"/>大家'
