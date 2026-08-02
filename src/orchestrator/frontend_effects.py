"""Reducer-approved frontend effects delivered over the control WebSocket."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from orchestrator.agent_pipeline import FrontendOperation, MediaOperation
from orchestrator.ids import SessionId, TurnId

if TYPE_CHECKING:
    from orchestrator.streaming_contracts import CancellationEpoch

type FrontendSender = Callable[
    [str, FrontendOperation, SessionId, TurnId], Awaitable[None]
]


@dataclass(frozen=True, slots=True)
class FrontendEffectDispatcher:
    """Schedules only already-accepted frontend operations on the live loop."""

    sender: FrontendSender

    def dispatch_media(
        self,
        operation: MediaOperation,
        *,
        session_id: SessionId,
        turn_id: TurnId,
        cancellation_epoch: CancellationEpoch,
    ) -> None:
        # Media has its own Sound/RTP boundary; this adapter never converts a
        # frontend dispatch into an audio side effect.
        _ = operation, session_id, turn_id, cancellation_epoch

    def dispatch_frontend(
        self,
        operation: FrontendOperation,
        *,
        session_id: SessionId,
        turn_id: TurnId,
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        event_type = _event_type(operation)
        coroutine = _dispatch(
            self.sender,
            event_type,
            operation,
            session_id,
            turn_id,
        )
        task = loop.create_task(coroutine)
        task.add_done_callback(_consume_task_failure)


async def send_frontend_operation(
    connection_send: Callable[[str], Awaitable[None]],
    event_type: str,
    operation: FrontendOperation,
    session_id: SessionId,
    turn_id: TurnId,
) -> None:
    envelope = _envelope(event_type, operation, session_id, turn_id)
    payload = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    await connection_send(payload)


async def _dispatch(
    sender: FrontendSender,
    event_type: str,
    operation: FrontendOperation,
    session_id: SessionId,
    turn_id: TurnId,
) -> None:
    await sender(event_type, operation, session_id, turn_id)


def _event_type(operation: FrontendOperation) -> str:
    return {
        "caption": "vtuber.caption.command",
        "animation": "vtuber.action.command",
        "ppt.load": "presentation.load.command",
        "ppt.navigate": "presentation.navigate.command",
    }[operation.kind]


def _envelope(
    event_type: str,
    operation: FrontendOperation,
    session_id: SessionId,
    turn_id: TurnId,
) -> dict[str, object]:
    segment_id = f"agent-{turn_id}"
    data: dict[str, object]
    if operation.kind == "caption":
        data = {"text": operation.value}
    elif operation.kind == "animation":
        data = {
            "action_id": str(uuid4()),
            "action": operation.value,
            "audio_stream_id": f"agent-{turn_id}",
            "start_at_ms": 0,
            "end_at_ms": 1,
            "start_rtp_timestamp": 0,
            "end_rtp_timestamp": 1,
        }
    elif operation.kind == "ppt.load":
        data = {"command_id": str(uuid4()), "deck_id": operation.value, "page": 1}
    else:
        data = {
            "command_id": str(uuid4()),
            "deck_id": operation.deck_id,
            "page": operation.value,
        }
    if operation.kind == "caption":
        data = {
            "caption_id": str(uuid4()),
            "text": operation.value,
            "audio_stream_id": f"agent-{turn_id}",
            "start_at_ms": 0,
            "end_at_ms": 1,
            "start_rtp_timestamp": 0,
            "end_rtp_timestamp": 1,
        }
    return {
        "schema_version": "1.1.0",
        "event_type": event_type,
        "event_id": str(uuid4()),
        "source": "orchestrator",
        "time": datetime.now(UTC).isoformat(),
        "trace_id": f"agent-{turn_id}",
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "segment_id": segment_id,
        "seq": 0,
        "data": data,
    }


def _consume_task_failure(task: asyncio.Task[None]) -> None:
    try:
        _ = task.result()
    except (OSError, RuntimeError):
        return
