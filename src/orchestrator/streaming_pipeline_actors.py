"""模块契约说明.

职责: 提供
orchestrator.streaming_pipeline_actors
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 声明 InputFrameHandler
    协议接口,约束实现方必须提供的行为。
    契约: 方法: __call__。
    """

    async def __call__(self, stream: StreamKey, frame: bytes) -> None:
        """函数契约说明.

        功能: 执行 __call__ 的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 frame: bytes。 必填。
        契约: 异步调用。 返回 `None`。
        """
        ...


@dataclass(frozen=True, slots=True)
class PipelineDropCounts:
    """类契约说明.

    职责: 保存 PipelineDropCounts
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: input_frames、endpointed_utte
    rances、answer_turns、tts_chunks。
    """

    input_frames: int = 0

    endpointed_utterances: int = 0

    answer_turns: int = 0

    tts_chunks: int = 0


@dataclass(slots=True)
class _StreamActor:
    """类契约说明.

    职责: 保存 _StreamActor
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、mailbox、wake、task。
    """

    stream: StreamKey

    mailbox: deque[bytes] = field(default_factory=deque)

    wake: asyncio.Event = field(default_factory=asyncio.Event)

    task: asyncio.Task[None] | None = None


@final
class StreamPipelineActors:
    """类契约说明.

    职责: 定义 StreamPipelineActors
    的状态、行为和对外协作边界。
    契约: 方法: __init__、actor_count、streams
    、drop_counts、submit、disconnect。
    """

    def __init__(
        self,
        handler: InputFrameHandler,
        *,
        input_capacity: int = INPUT_FRAME_CAPACITY,
        global_capacity: int = 8,
    ) -> None:
        """函数契约说明.

        功能: 初始化 StreamPipelineActors
        的字段并建立实例不变式。
        参数: self 表示当前实例。 handler:
        InputFrameHandler。 必填。
        input_capacity: int。 可省略。
        global_capacity: int。 可省略。
        契约: 同步调用。 返回 `None`。
        """
        self._handler = handler

        self._input_capacity = input_capacity

        self._limiter = asyncio.Semaphore(global_capacity)

        self._actors: dict[StreamKey, _StreamActor] = {}

        self._input_frame_drops = 0

        self._closed = False

    @property
    def actor_count(self) -> int:
        """函数契约说明.

        功能: 执行 actor_count 的同步逻辑,并协调
        len。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `int`。
        """
        return len(self._actors)

    @property
    def streams(self) -> tuple[StreamKey, ...]:
        """函数契约说明.

        功能: 执行 streams 的同步逻辑,并协调 tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[StreamKey,
        ...]`。
        """
        return tuple(self._actors)

    @property
    def drop_counts(self) -> PipelineDropCounts:
        """函数契约说明.

        功能: 执行 drop_counts 的同步逻辑,并协调
        PipelineDropCounts。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `PipelineDropCounts`。
        """
        return PipelineDropCounts(input_frames=self._input_frame_drops)

    def submit(self, stream: StreamKey, frame: bytes) -> None:
        """函数契约说明.

        功能: 执行 submit 的同步逻辑,并协调
        setdefault, append, set,
        _StreamActor。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 frame: bytes。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 disconnect 的异步逻辑,并协调
        discard, suppress。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        task = self.discard(stream)

        if task is not None:
            with suppress(asyncio.CancelledError):
                await task

    def discard(self, stream: StreamKey) -> asyncio.Task[None] | None:
        """函数契约说明.

        功能: 执行 discard 的同步逻辑,并协调 pop,
        clear, cancel。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回 `asyncio.Task[None]
        | None`。
        """
        actor = self._actors.pop(stream, None)

        if actor is None:
            return None

        _ = actor.mailbox.clear()

        task = actor.task

        if task is not None:
            _ = task.cancel()

        return task

    async def wait_quiescent(self) -> None:
        """函数契约说明.

        功能: 执行 wait_quiescent 的异步逻辑,并协调
        tuple, gather, values。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        tasks = tuple(
            actor.task for actor in self._actors.values() if actor.task is not None
        )

        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """函数契约说明.

        功能: 执行 aclose 的异步逻辑,并协调 gather,
        disconnect, tuple。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        self._closed = True

        _ = await asyncio.gather(
            *(self.disconnect(stream) for stream in tuple(self._actors)),
        )

    async def _run(self, actor: _StreamActor) -> None:
        """函数契约说明.

        功能: 执行 _run 的异步逻辑,并协调 clear,
        wait, popleft, _handler。
        参数: self 表示当前实例。 actor:
        _StreamActor。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        while True:
            _ = await actor.wake.wait()

            while actor.mailbox:
                frame = actor.mailbox.popleft()

                async with self._limiter:
                    await self._handler(actor.stream, frame)

            _ = actor.wake.clear()

            if not actor.mailbox:
                return
