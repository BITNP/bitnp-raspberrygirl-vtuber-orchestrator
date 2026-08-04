"""Caption timeline delivery over the control WebSocket.

The old generic frontend-operation dispatcher was retired. A
timeline is emitted only after the response media gate has admitted audio.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from orchestrator.caption_timeline import CaptionTimelineCancel, CaptionTimelineCommand

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from orchestrator.ids import SessionId, TurnId

async def send_caption_timeline(
    connection_send: Callable[[str], Awaitable[None]],
    command: CaptionTimelineCommand | CaptionTimelineCancel,
    session_id: SessionId,
    turn_id: TurnId,
) -> None:
    event_type = (
        "vtuber.caption.timeline.command"
        if isinstance(command, CaptionTimelineCommand)
        else "vtuber.caption.timeline.cancel"
    )
    envelope = {
        "schema_version": "1.1.0",
        "event_type": event_type,
        "event_id": str(uuid4()),
        "source": "orchestrator",
        "time": datetime.now(UTC).isoformat(),
        "trace_id": f"agent-{turn_id}",
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "segment_id": f"agent-{turn_id}",
        "seq": 0,
        "data": command.payload(),
    }
    payload = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    await connection_send(payload)
