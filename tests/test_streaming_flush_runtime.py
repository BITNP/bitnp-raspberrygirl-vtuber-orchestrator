"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.config import load_fake_config
from orchestrator.ids import SessionId
from orchestrator.json_boundary import parse_json_value
from orchestrator.observability import OnsiteObservability
from orchestrator.scheduler_reflex import SchedulerOutputFence
from orchestrator.sessions import SessionScheduler
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushRequestId,
    GeneratedSsrc,
    SegmentId,
    StreamFlush,
    StreamKey,
    TurnId,
)
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_control import EnvelopeCorrelation
from orchestrator.transport_runtime import TransportRuntime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from http import HTTPStatus

    from websockets.http11 import Response

    from orchestrator.json_boundary import JsonValue
    from orchestrator.transport_hub import RtpHub
    from orchestrator.transport_runtime import ControlHandler


@dataclass
class _Clock:
    """类契约说明.

    职责: 保存 _Clock 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: now_ms。 方法: advance。
    """

    now_ms: int = 0

    def advance(self, milliseconds: int) -> None:
        """函数契约说明.

        功能: 执行 advance 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 milliseconds:
        int。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.now_ms += milliseconds


@dataclass
class _DatagramTransport:
    """类契约说明.

    职责: 保存 _DatagramTransport
    不可变数据结构,用类型标注表达字段契约。
    契约: 方法: sendto、close。
    """

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 data: bytes。
        必填。 addr: tuple[str, int]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = data

        _ = addr

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return


@dataclass
class _ControlServer:
    """类契约说明.

    职责: 保存 _ControlServer
    不可变数据结构,用类型标注表达字段契约。
    契约: 方法: close、wait_closed。
    """

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return

    async def wait_closed(self) -> None:
        """函数契约说明.

        功能: 执行 wait_closed
        的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """

        return


@dataclass
class _WssConnection:
    """类契约说明.

    职责: 保存 _WssConnection
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: peer_ip、incoming、sent。 方法: r
    emote_address、__aiter__、__anext__、se
    nd、respond。
    """

    peer_ip: str

    incoming: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)

    sent: list[str] = field(default_factory=list)

    @property
    def remote_address(self) -> tuple[str, int]:
        """函数契约说明.

        功能: 执行 remote_address
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[str, int]`。
        """

        return (self.peer_ip, 443)

    def __aiter__(self) -> AsyncIterator[str]:
        """函数契约说明.

        功能: 执行 __aiter__ 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `AsyncIterator[str]`。
        """

        return self

    async def __anext__(self) -> str:
        """函数契约说明.

        功能: 执行 __anext__ 的异步逻辑,并协调 get。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `str`。
        """

        message = await self.incoming.get()

        if message is None:
            raise StopAsyncIteration

        return message

    async def send(self, message: str) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 message: str。
        必填。
        契约: 异步调用。 返回 `None`。
        """

        self.sent.append(message)

    def respond(self, status: HTTPStatus, text: str) -> Response:
        """函数契约说明.

        功能: 执行 respond 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 status:
        HTTPStatus。 必填。 text: str。 必填。
        契约: 同步调用。 返回 `Response`。
        """

        _ = status

        _ = text

        raise AssertionError


def test_runtime_admits_replacement_only_after_matching_sound_ack() -> None:
    """函数契约说明.

    功能: 验证 runtime admits replacement
    only after matching sound ack
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_matching_ack_proof())


def test_runtime_rejects_invalid_ack_and_missing_ack_timeout() -> None:
    """函数契约说明.

    功能: 验证 runtime rejects invalid ack
    and missing ack timeout 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_rejection_proof())


async def _matching_ack_proof() -> None:
    # Given: live Mic and Sound WSS sessions registered on one runtime-owned stream.

    """函数契约说明.

    功能: 执行 _matching_ack_proof 的异步逻辑,并协调
    _Clock, OnsiteObservability,
    TransportRuntime, set_observability。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    clock = _Clock()

    observability = OnsiteObservability(load_fake_config())

    runtime = TransportRuntime(
        _config(),
        datagram_listener=_datagram_listener,
        control_listener=_control_listener,
        clock=clock,
    )

    runtime.set_observability(observability)

    source, sink, tasks = await _registered_runtime(runtime)

    fence = SchedulerOutputFence(
        SessionScheduler(
            session_id=SessionId("session-001"), turn_id_prefix="turn-reflex"
        )
    )

    runtime.set_output_fence(fence)

    stream = StreamKey(session_id="session-001", stream_id="stream-001")

    correlation = EnvelopeCorrelation("trace-source-001", "session-001", 29)

    _ = fence.activate(
        stream=stream,
        segment_id=SegmentId("segment-active"),
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
        correlation=correlation,
    )

    replacement, flush = fence.interrupt(
        stream=stream,
        segment_id=SegmentId("segment-replacement"),
        correlation=correlation,
    )

    # When: runtime sends a flush and Sound returns the exact acknowledgement over WSS.

    gate_started_at = clock.now_ms

    await runtime.request_stream_flush(flush)

    assert fence.can_emit(stream, replacement.cancellation_epoch) is False

    await sink.incoming.put(_acknowledgement(flush, session_id="stale-session"))

    await asyncio.sleep(0)

    assert fence.can_emit(stream, replacement.cancellation_epoch) is False

    await sink.incoming.put(_acknowledgement(flush))

    await asyncio.sleep(0)

    admitted = await runtime.admit_replacement(flush)

    mismatched = await runtime.admit_replacement(
        StreamFlush(
            stream=flush.stream,
            turn_id=flush.turn_id,
            segment_id=SegmentId("segment-other"),
            cancellation_epoch=flush.cancellation_epoch,
            request_id=FlushRequestId("flush-other"),
            target_generated_ssrc=flush.target_generated_ssrc,
        )
    )

    # Then: only the matching ack permits a second generated stream command.

    assert clock.now_ms - gate_started_at <= 20

    assert _event_types(sink.sent).count("media.stream.flush") == 1

    assert _event_types(source.sent).count("media.stream.flush") == 1

    assert admitted is True

    assert mismatched is False

    assert fence.can_emit(stream, replacement.cancellation_epoch) is True

    assert _event_types(sink.sent).count("media.stream.command") == 2

    flush_envelope = next(
        _envelope_value(message)
        for message in sink.sent
        if _envelope_value(message)["event_type"] == "media.stream.flush"
    )

    replacement = _envelope_value(sink.sent[-1])

    replacement_data = replacement["data"]

    assert isinstance(replacement_data, dict)

    assert _correlation(flush_envelope) == ("trace-source-001", "session-001", 29)

    assert _correlation(replacement) == ("trace-source-001", "session-001", 29)

    assert (
        replacement["turn_id"],
        replacement["segment_id"],
        replacement_data["cancellation_epoch"],
    ) == (str(flush.turn_id), str(flush.segment_id), int(flush.cancellation_epoch))

    assert observability.records[-1].stage == "flush_ack"

    assert (
        observability.records[-1].trace_id,
        observability.records[-1].session_id,
        observability.records[-1].seq,
        observability.records[-1].turn_id,
        observability.records[-1].segment_id,
        observability.records[-1].cancellation_epoch,
    ) == (
        "trace-source-001",
        "session-001",
        29,
        str(flush.turn_id),
        str(flush.segment_id),
        int(flush.cancellation_epoch),
    )

    await _close_runtime(runtime, source, sink, tasks)


async def _rejection_proof() -> None:
    # Given: a registered Sound control peer and a pending generated-media flush.

    """函数契约说明.

    功能: 执行 _rejection_proof 的异步逻辑,并协调
    _Clock, TransportRuntime, _flush,
    advance。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    clock = _Clock()

    runtime = TransportRuntime(
        _config(),
        datagram_listener=_datagram_listener,
        control_listener=_control_listener,
        clock=clock,
    )

    source, sink, tasks = await _registered_runtime(runtime)

    flush = _flush()

    # When: Sound sends a stale-session acknowledgement, then time reaches 750ms.

    await runtime.request_stream_flush(flush)

    await sink.incoming.put(_acknowledgement(flush, session_id="stale-session"))

    await asyncio.sleep(0)

    clock.advance(250)

    await runtime.advance_flush_admission()

    clock.advance(500)

    await runtime.advance_flush_admission()

    # Then: retry occurs once and invalid/missing acknowledgement blocks replacement.

    assert _event_types(sink.sent).count("media.stream.flush") == 2

    assert await runtime.admit_replacement(flush) is False

    assert [failure.reason for failure in runtime.flush_failures] == [
        "invalid_ack",
        "timeout",
    ]

    await _close_runtime(runtime, source, sink, tasks)


async def _registered_runtime(
    runtime: TransportRuntime,
) -> tuple[
    _WssConnection, _WssConnection, tuple[asyncio.Task[None], asyncio.Task[None]]
]:
    """函数契约说明.

    功能: 执行 _registered_runtime 的异步逻辑,并协调
    _WssConnection, create_task, start,
    handle_control。
    参数: runtime: TransportRuntime。 必填。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回
    `tuple[_WssConnection,
    _WssConnection,
    tuple[asyncio.Task[None],
    asyncio.Task[None]]]`。
    """

    await runtime.start()

    source = _WssConnection(peer_ip="192.0.2.10")

    sink = _WssConnection(peer_ip="192.0.2.11")

    source_task = asyncio.create_task(runtime.handle_control(source))

    sink_task = asyncio.create_task(runtime.handle_control(sink))

    await source.incoming.put(_source_registration())

    await sink.incoming.put(_sink_registration())

    await asyncio.sleep(0)

    return source, sink, (source_task, sink_task)


async def _close_runtime(
    runtime: TransportRuntime,
    source: _WssConnection,
    sink: _WssConnection,
    tasks: tuple[asyncio.Task[None], asyncio.Task[None]],
) -> None:
    """函数契约说明.

    功能: 执行 _close_runtime 的异步逻辑,并协调 put,
    close。
    参数: runtime: TransportRuntime。 必填。
    source: _WssConnection。 必填。 sink:
    _WssConnection。 必填。 tasks:
    tuple[asyncio.Task[None],
    asyncio.Task[None]]。 必填。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

    await source.incoming.put(None)

    await sink.incoming.put(None)

    await tasks[0]

    await tasks[1]

    await runtime.close()


def _flush() -> StreamFlush:
    """函数契约说明.

    功能: 执行 _flush 的同步逻辑,并协调 StreamFlush,
    StreamKey, TurnId, SegmentId。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `StreamFlush`。
    """

    return StreamFlush(
        stream=StreamKey(session_id="session-001", stream_id="stream-001"),
        turn_id=TurnId("turn-001"),
        segment_id=SegmentId("segment-001"),
        cancellation_epoch=CancellationEpoch(3),
        request_id=FlushRequestId("flush-request-001"),
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
    )


def _source_registration() -> str:
    """函数契约说明.

    功能: 执行 _source_registration
    的同步逻辑,并协调 _envelope,
    _EnvelopeFields, _codec。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `str`。
    """

    return _envelope(
        _EnvelopeFields(
            event_type="media.rtp.source.register",
            source="mic",
            data={
                "stream_id": "stream-001",
                "ssrc": 0x1234_5678,
                "codec": _codec(),
                "rtp_endpoint": {"host": "192.0.2.10", "port": 5004},
            },
            trace_id="trace-source-001",
            seq=29,
        )
    )


def _sink_registration() -> str:
    """函数契约说明.

    功能: 执行 _sink_registration 的同步逻辑,并协调
    _envelope, _EnvelopeFields, _codec。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `str`。
    """

    return _envelope(
        _EnvelopeFields(
            event_type="media.rtp.sink.register",
            source="sound",
            data={
                "stream_id": "stream-001",
                "codec": _codec(),
                "rtp_endpoint": {"host": "192.0.2.11", "port": 5006},
            },
        )
    )


def _acknowledgement(flush: StreamFlush, *, session_id: str = "session-001") -> str:
    """函数契约说明.

    功能: 执行 _acknowledgement 的同步逻辑,并协调
    _envelope, _EnvelopeFields, str,
    int。
    参数: flush: StreamFlush。 必填。
    session_id: str。 可省略。
    契约: 同步调用。 返回 `str`。
    """

    return _envelope(
        _EnvelopeFields(
            event_type="media.stream.flush.ack",
            source="sound",
            session_id=session_id,
            turn_id=str(flush.turn_id),
            segment_id=str(flush.segment_id),
            data={
                "stream_id": flush.stream.stream_id,
                "cancellation_epoch": int(flush.cancellation_epoch),
                "request_id": str(flush.request_id),
                "target_generated_ssrc": int(flush.target_generated_ssrc),
            },
            trace_id="trace-source-001",
            seq=29,
        )
    )


@dataclass(frozen=True)
class _EnvelopeFields:
    """类契约说明.

    职责: 保存 _EnvelopeFields
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: data、event_type、source、sessi
    on_id、turn_id、segment_id。
    """

    data: dict[str, JsonValue]

    event_type: str

    source: str

    session_id: str = "session-001"

    turn_id: str | None = None

    segment_id: str | None = None

    trace_id: str = "trace-001"

    seq: int = 1


def _envelope(fields: _EnvelopeFields) -> str:
    """函数契约说明.

    功能: 执行 _envelope 的同步逻辑,并协调 dumps。
    参数: fields: _EnvelopeFields。 必填。
    契约: 同步调用。 返回 `str`。
    """

    envelope: dict[str, JsonValue] = {
        "schema_version": "1.0.0",
        "event_type": fields.event_type,
        "event_id": fields.event_type,
        "source": fields.source,
        "time": "2026-07-28T00:00:00Z",
        "trace_id": fields.trace_id,
        "session_id": fields.session_id,
        "seq": fields.seq,
        "data": fields.data,
    }

    if fields.turn_id is not None:
        envelope["turn_id"] = fields.turn_id

    if fields.segment_id is not None:
        envelope["segment_id"] = fields.segment_id

    return json.dumps(envelope)


def _codec() -> dict[str, JsonValue]:
    """函数契约说明.

    功能: 执行 _codec 的同步逻辑,并维持签名契约。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """

    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _event_types(messages: list[str]) -> list[str]:
    """函数契约说明.

    功能: 执行 _event_types 的同步逻辑,并协调 loads。
    参数: messages: list[str]。 必填。
    契约: 同步调用。 返回 `list[str]`。
    """

    return [json.loads(message)["event_type"] for message in messages]


def _correlation(envelope: dict[str, JsonValue]) -> tuple[str, str, int]:
    """函数契约说明.

    功能: 执行 _correlation 的同步逻辑,并协调
    isinstance。
    参数: envelope: dict[str, JsonValue]。
    必填。
    契约: 同步调用。 返回 `tuple[str, str, int]`。
    """

    trace_id = envelope["trace_id"]

    session_id = envelope["session_id"]

    seq = envelope["seq"]

    assert isinstance(trace_id, str)

    assert isinstance(session_id, str)

    assert isinstance(seq, int)

    return trace_id, session_id, seq


def _envelope_value(message: str) -> dict[str, JsonValue]:
    """函数契约说明.

    功能: 执行 _envelope_value 的同步逻辑,并协调
    parse_json_value, isinstance。
    参数: message: str。 必填。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """

    value = parse_json_value(message)

    assert isinstance(value, dict)

    return value


def _config() -> TransportConfig:
    """函数契约说明.

    功能: 执行 _config 的同步逻辑,并协调
    TransportConfig。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `TransportConfig`。
    """

    return TransportConfig(
        "127.0.0.1",
        8765,
        "127.0.0.1",
        5004,
        "127.0.0.1",
        8765,
        5004,
        "ws",
        None,
        None,
        None,
    )


async def _datagram_listener(
    _host: str, _port: int, _hub: RtpHub
) -> _DatagramTransport:
    """函数契约说明.

    功能: 执行 _datagram_listener 的异步逻辑,并协调
    _DatagramTransport。
    参数: _host: str。 必填。 _port: int。 必填。
    _hub: RtpHub。 必填。
    契约: 异步调用。 返回 `_DatagramTransport`。
    """

    return _DatagramTransport()


async def _control_listener(
    _config: TransportConfig, _handler: ControlHandler
) -> _ControlServer:
    """函数契约说明.

    功能: 执行 _control_listener 的异步逻辑,并协调
    _ControlServer。
    参数: _config: TransportConfig。 必填。
    _handler: ControlHandler。 必填。
    契约: 异步调用。 返回 `_ControlServer`。
    """

    return _ControlServer()
