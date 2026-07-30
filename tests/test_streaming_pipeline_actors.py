from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from orchestrator.streaming_contracts import StreamKey
from orchestrator.streaming_pipeline_actors import StreamPipelineActors


@dataclass(slots=True)
class _Handler:
    started: list[bytes] = field(default_factory=list)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    first_started: asyncio.Event = field(default_factory=asyncio.Event)
    two_started: asyncio.Event = field(default_factory=asyncio.Event)

    async def __call__(self, stream: StreamKey, frame: bytes) -> None:
        _ = stream
        self.started.append(frame)
        _ = self.first_started.set()
        if len(self.started) == 2:
            _ = self.two_started.set()
        _ = await self.release.wait()


def test_actors_drop_oldest_unstarted_frame_and_preserve_active_work() -> None:
    asyncio.run(_drop_oldest_unstarted_frame_proof())


async def _drop_oldest_unstarted_frame_proof() -> None:
    # Given: one stream whose active frame is deliberately held while its mailbox fills.
    handler = _Handler()
    actors = StreamPipelineActors(handler, input_capacity=2, global_capacity=1)
    stream = StreamKey("session-a", "stream-a")
    actors.submit(stream, b"active")
    _ = await handler.first_started.wait()

    # When: three newer frames arrive before the active provider work completes.
    actors.submit(stream, b"oldest-unstarted")
    actors.submit(stream, b"newer")
    actors.submit(stream, b"newest")
    handler.release.set()
    await actors.wait_quiescent()

    # Then: the active frame and the two newest queued frames run in FIFO order.
    assert handler.started == [b"active", b"newer", b"newest"]
    assert actors.drop_counts.input_frames == 1
    assert actors.actor_count == 1
    await actors.aclose()


def test_actors_allow_different_streams_to_start_without_bridge_wide_serialization(
) -> None:
    asyncio.run(_cross_stream_fairness_proof())


async def _cross_stream_fairness_proof() -> None:
    # Given: two streams and a global capacity that admits one active item from each.
    handler = _Handler()
    actors = StreamPipelineActors(handler, global_capacity=2)
    first = StreamKey("session-a", "stream-a")
    second = StreamKey("session-b", "stream-b")

    # When: both streams receive work before either provider call may finish.
    actors.submit(first, b"first")
    actors.submit(second, b"second")
    _ = await handler.two_started.wait()

    # Then: independent streams have both entered their provider stage.
    assert set(handler.started) == {b"first", b"second"}
    handler.release.set()
    await actors.wait_quiescent()
    await actors.aclose()


def test_disconnect_cancels_active_work_discards_mailbox_and_leaves_no_actor() -> None:
    asyncio.run(_disconnect_proof())


async def _disconnect_proof() -> None:
    # Given: an actor whose provider call waits at its semantic cancellation boundary.
    handler = _Handler()
    actors = StreamPipelineActors(handler)
    stream = StreamKey("session-a", "stream-a")
    actors.submit(stream, b"active")
    _ = await handler.first_started.wait()
    actors.submit(stream, b"queued")

    # When: the authenticated stream route disconnects.
    await actors.disconnect(stream)

    # Then: queued work cannot run and the stream actor is fully removed.
    assert handler.started == [b"active"]
    assert actors.actor_count == 0
    await actors.aclose()


def test_shutdown_cancels_all_stream_actors_and_reaches_quiescence() -> None:
    asyncio.run(_shutdown_proof())


async def _shutdown_proof() -> None:
    # Given: active provider work for two independent streams.
    handler = _Handler()
    actors = StreamPipelineActors(handler, global_capacity=2)
    actors.submit(StreamKey("session-a", "stream-a"), b"a")
    actors.submit(StreamKey("session-b", "stream-b"), b"b")
    _ = await handler.two_started.wait()

    # When: the runtime begins shutdown.
    await actors.aclose()

    # Then: every actor task has settled and no actor remains retained.
    assert actors.actor_count == 0
