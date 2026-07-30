"""Bounded, stream-keyed asynchronous actors for onsite RTP processing."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, final

if TYPE_CHECKING:
    from orchestrator.streaming_contracts import StreamKey

INPUT_FRAME_CAPACITY = 100
ENDPOINTED_UTTERANCE_CAPACITY = 4
ANSWER_TURN_CAPACITY = 2
TTS_CHUNK_CAPACITY = 16


class InputFrameHandler(Protocol):
    """Processes one stream-local input frame."""

    async def __call__(self, stream: StreamKey, frame: bytes) -> None:
        """Process one stream-local input frame."""
        ...


@dataclass(frozen=True, slots=True)
class PipelineDropCounts:
    """Typed counts of unstarted items displaced by bounded mailboxes."""

    input_frames: int = 0
    endpointed_utterances: int = 0
    answer_turns: int = 0
    tts_chunks: int = 0


@dataclass(slots=True)
class _StreamActor:
    stream: StreamKey
    mailbox: deque[bytes] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


@final
class StreamPipelineActors:
    """Own one bounded input actor per stream and a shared provider limiter."""

    def __init__(
        self,
        handler: InputFrameHandler,
        *,
        input_capacity: int = INPUT_FRAME_CAPACITY,
        global_capacity: int = 8,
    ) -> None:
        """Create bounded stream mailboxes sharing one provider capacity limit."""
        self._handler = handler
        self._input_capacity = input_capacity
        self._limiter = asyncio.Semaphore(global_capacity)
        self._actors: dict[StreamKey, _StreamActor] = {}
        self._input_frame_drops = 0
        self._closed = False

    @property
    def actor_count(self) -> int:
        """Return the currently retained stream actor count."""
        return len(self._actors)

    @property
    def streams(self) -> tuple[StreamKey, ...]:
        """Return the stream keys whose actor state is currently retained."""
        return tuple(self._actors)

    @property
    def drop_counts(self) -> PipelineDropCounts:
        """Return typed mailbox displacement counts."""
        return PipelineDropCounts(input_frames=self._input_frame_drops)

    def submit(self, stream: StreamKey, frame: bytes) -> None:
        """Queue one frame, displacing the oldest unstarted frame when full."""
        if self._closed:
            return
        actor = self._actors.setdefault(stream, _StreamActor(stream))
        if len(actor.mailbox) == self._input_capacity:
            _ = actor.mailbox.popleft()
            self._input_frame_drops += 1
        actor.mailbox.append(frame)
        _ = actor.wake.set()
        task = actor.task
        if task is None or task.done():
            actor.task = asyncio.create_task(self._run(actor))

    async def disconnect(self, stream: StreamKey) -> None:
        """Cancel active route work and remove all state owned by its stream."""
        task = self.discard(stream)
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task

    def discard(self, stream: StreamKey) -> asyncio.Task[None] | None:
        """Synchronously remove one actor for synchronous route invalidation."""
        actor = self._actors.pop(stream, None)
        if actor is None:
            return None
        _ = actor.mailbox.clear()
        task = actor.task
        if task is not None:
            _ = task.cancel()
        return task

    async def wait_quiescent(self) -> None:
        """Wait until all work accepted before this call has completed."""
        tasks = tuple(
            actor.task for actor in self._actors.values() if actor.task is not None
        )
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """Cancel all active provider work and release every stream actor."""
        self._closed = True
        _ = await asyncio.gather(
            *(self.disconnect(stream) for stream in tuple(self._actors)),
        )

    async def _run(self, actor: _StreamActor) -> None:
        while True:
            _ = await actor.wake.wait()
            while actor.mailbox:
                frame = actor.mailbox.popleft()
                async with self._limiter:
                    await self._handler(actor.stream, frame)
            _ = actor.wake.clear()
            if not actor.mailbox:
                return
