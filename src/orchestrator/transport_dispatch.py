"""模块契约说明.

职责: 提供 orchestrator.transport_dispatch
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, final

from orchestrator.ids import ConnectionId
from orchestrator.observability import (
    OnsiteObservability,
    OnsiteStage,
    StageCorrelation,
)
from orchestrator.streaming_contracts import (
    FlushAcknowledgement,
    FlushAdmission,
    FlushClock,
    FlushFailure,
    StreamFlush,
    StreamKey,
)
from orchestrator.transport_control import (
    ControlEvent,
    EnvelopeCorrelation,
    SinkRegistration,
    SourceRegistration,
    StreamReady,
    StreamState,
    parse_control_event,
)

if TYPE_CHECKING:
    from orchestrator.scheduler_reflex import SchedulerOutputFence


_CODEC = {
    "format": "L16",
    "clock_rate_hz": 16_000,
    "channels": 1,
    "payload_type": 96,
    "samples_per_frame": 320,
}


class ControlPeer(Protocol):
    """类契约说明.

    职责: 声明 ControlPeer
    协议接口,约束实现方必须提供的行为。
    契约: 方法: send。
    """

    async def send(self, message: str) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 message: str。
        必填。
        契约: 异步调用。 返回 `None`。
        """


class RouteRegistry(Protocol):
    """类契约说明.

    职责: 声明 RouteRegistry
    协议接口,约束实现方必须提供的行为。
    契约: 方法: register_control、remove_conn
    ection、remove_stream、output_ssrc、cor
    relation。
    """

    def register_control(
        self,
        raw_message: ControlEvent | str,
        peer_ip: str,
        owner: ConnectionId | None = None,
    ) -> None:
        """函数契约说明.

        功能: 执行 register_control
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 raw_message:
        ControlEvent | str。 必填。 peer_ip:
        str。 必填。 owner: ConnectionId |
        None。 可省略。
        契约: 同步调用。 返回 `None`。
        """

    def remove_connection(self, owner: ConnectionId) -> None:
        """函数契约说明.

        功能: 执行 remove_connection
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 owner:
        ConnectionId。 必填。
        契约: 同步调用。 返回 `None`。
        """

    def remove_stream(self, session_id: str, stream_id: str) -> None:
        """函数契约说明.

        功能: 执行 remove_stream
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 session_id:
        str。 必填。 stream_id: str。 必填。
        契约: 同步调用。 返回 `None`。
        """

    def output_ssrc(self, stream: StreamKey, cancellation_epoch: int = 0) -> int:
        """函数契约说明.

        功能: 执行 output_ssrc
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        cancellation_epoch: int。 可省略。
        契约: 同步调用。 返回 `int`。
        """
        ...

    def correlation(self, stream: StreamKey) -> EnvelopeCorrelation | None:
        """函数契约说明.

        功能: 执行 correlation
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回
        `EnvelopeCorrelation | None`。
        """
        ...


@dataclass(frozen=True, slots=True)
class _SourcePeer:
    """类契约说明.

    职责: 保存 _SourcePeer
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: connection、ssrc。
    """

    connection: ControlPeer

    ssrc: int


@dataclass(frozen=True, slots=True)
class _SinkPeer:
    """类契约说明.

    职责: 保存 _SinkPeer
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: connection、host、udp_port。
    """

    connection: ControlPeer

    host: str

    udp_port: int


@final
class TransportControlDispatch:
    """类契约说明.

    职责: 定义 TransportControlDispatch
    的状态、行为和对外协作边界。
    契约: 方法: __init__、register、request_st
    ream_flush、advance_flush_admission、a
    dmit_replacement、flush_failures。
    """

    def __init__(
        self,
        hub: RouteRegistry,
        *,
        clock: FlushClock | None = None,
        observability: OnsiteObservability | None = None,
    ) -> None:
        """函数契约说明.

        功能: 初始化 TransportControlDispatch
        的字段并建立实例不变式。
        参数: self 表示当前实例。 hub:
        RouteRegistry。 必填。 clock:
        FlushClock | None。 可省略。
        observability:
        OnsiteObservability | None。 可省略。
        契约: 同步调用。 返回 `None`。
        """
        self._hub: RouteRegistry = hub

        self._observability = observability

        self._sources: dict[StreamKey, _SourcePeer] = {}

        self._sinks: dict[StreamKey, _SinkPeer] = {}

        self._dispatched: set[StreamKey] = set()

        self._ready_sinks: set[StreamKey] = set()

        self._released_sources: set[StreamKey] = set()

        self._flush_outbox: list[StreamFlush] = []

        self._flush_admission = FlushAdmission(
            clock=_MonotonicFlushClock() if clock is None else clock,
            sender=self,
        )

        self._output_fence: SchedulerOutputFence | None = None

    async def register(
        self, raw_message: str, peer_ip: str, connection: ControlPeer
    ) -> None:
        """函数契约说明.

        功能: 执行 register 的异步逻辑,并协调
        parse_control_event,
        register_control, isinstance,
        _connection_id。
        参数: self 表示当前实例。 raw_message:
        str。 必填。 peer_ip: str。 必填。
        connection: ControlPeer。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        event = parse_control_event(raw_message)

        self._hub.register_control(event, peer_ip, _connection_id(connection))

        if isinstance(event, StreamState):
            self._record_playback(event)

        match event:
            case SourceRegistration(
                session_id=session_id, stream_id=stream_id, ssrc=ssrc
            ):
                self._sources[StreamKey(session_id, stream_id)] = _SourcePeer(
                    connection, ssrc
                )

            case SinkRegistration(
                session_id=session_id, stream_id=stream_id, udp_port=port
            ):
                self._sinks[StreamKey(session_id, stream_id)] = _SinkPeer(
                    connection, peer_ip, port
                )

            case StreamReady(session_id=session_id, stream_id=stream_id):
                stream = StreamKey(session_id, stream_id)

                sink = self._sinks.get(stream)

                if sink is None or sink.connection is not connection:
                    return

                self._ready_sinks.add(stream)

                await self._release_source(stream)

                return

            case StreamState(
                session_id=session_id,
                stream_id=stream_id,
                state="cancelled" | "finished" | "error",
            ):
                self._discard(StreamKey(session_id, stream_id))

            case StreamState():
                return

            case FlushAcknowledgement() as acknowledgement:
                self._record_flush_acknowledgement(acknowledgement)

                admitted = self._flush_admission.acknowledge(acknowledgement)

                output_fence = self._output_fence

                if admitted and output_fence is not None:
                    _ = output_fence.acknowledge(acknowledgement)

                return

            case _:
                return

        await self._dispatch_start(StreamKey(event.session_id, event.stream_id))

    async def request_stream_flush(self, flush: StreamFlush) -> None:
        """函数契约说明.

        功能: 执行 request_stream_flush
        的异步逻辑,并协调 replace,
        _record_flush, begin,
        correlation。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        correlation = flush.correlation or self._hub.correlation(flush.stream)

        if correlation is None:
            return

        flush = replace(flush, correlation=correlation)

        self._record_flush(flush)

        self._flush_admission.begin(flush)

        await self._deliver_flushes()

    async def advance_flush_admission(self) -> None:
        """函数契约说明.

        功能: 执行 advance_flush_admission
        的异步逻辑,并协调 advance,
        _deliver_flushes。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        self._flush_admission.advance()

        await self._deliver_flushes()

    async def admit_replacement(self, flush: StreamFlush) -> bool:
        """函数契约说明.

        功能: 执行 admit_replacement
        的异步逻辑,并协调 get, admitted, send,
        _stream_command_envelope。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `bool`。
        """
        if not self._flush_admission.admitted(flush):
            return False

        source = self._sources.get(flush.stream)

        sink = self._sinks.get(flush.stream)

        if source is None or sink is None:
            return False

        await sink.connection.send(
            _stream_command_envelope(
                flush.stream,
                self._hub.output_ssrc(flush.stream, int(flush.cancellation_epoch)),
                sink,
                self._hub.correlation(flush.stream),
                flush,
            )
        )

        return True

    @property
    def flush_failures(self) -> tuple[FlushFailure, ...]:
        """函数契约说明.

        功能: 执行 flush_failures 的同步逻辑,并协调
        tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `tuple[FlushFailure, ...]`。
        """
        return tuple(self._flush_admission.failures)

    def send_flush(self, flush: StreamFlush) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._flush_outbox.append(flush)

    async def cancel_stream(self, session_id: str, stream_id: str) -> None:
        """函数契约说明.

        功能: 执行 cancel_stream 的异步逻辑,并协调
        StreamKey,
        _record_transport_transition,
        get, correlation。
        参数: self 表示当前实例。 session_id:
        str。 必填。 stream_id: str。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        stream = StreamKey(session_id, stream_id)

        self._record_transport_transition("cancellation", stream)

        sink = self._sinks.get(stream)

        correlation = self._hub.correlation(stream)

        self._hub.remove_stream(session_id, stream_id)

        self._discard(stream)

        if sink is not None and correlation is not None:
            await sink.connection.send(_cancel_envelope(stream_id, correlation))

    def clear(self) -> None:
        """函数契约说明.

        功能: 执行 clear 的同步逻辑,并协调 clear。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self._sources.clear()

        self._sinks.clear()

        self._dispatched.clear()

        self._ready_sinks.clear()

        self._released_sources.clear()

        self._flush_outbox.clear()

    def set_observability(self, observability: OnsiteObservability) -> None:
        """函数契约说明.

        功能: 执行 set_observability
        的同步逻辑,并产出 _observability。
        参数: self 表示当前实例。 observability:
        OnsiteObservability。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._observability = observability

    def set_output_fence(self, output_fence: SchedulerOutputFence) -> None:
        """函数契约说明.

        功能: 执行 set_output_fence
        的同步逻辑,并产出 _output_fence。
        参数: self 表示当前实例。 output_fence:
        SchedulerOutputFence。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._output_fence = output_fence

    def remove_connection(self, connection: ControlPeer) -> None:
        """函数契约说明.

        功能: 执行 remove_connection
        的同步逻辑,并协调 remove_connection,
        tuple, _connection_id, items。
        参数: self 表示当前实例。 connection:
        ControlPeer。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._hub.remove_connection(_connection_id(connection))

        for stream, source in tuple(self._sources.items()):
            if source.connection is connection:
                del self._sources[stream]

                self._dispatched.discard(stream)

                self._ready_sinks.discard(stream)

                self._released_sources.discard(stream)

        for stream, sink in tuple(self._sinks.items()):
            if sink.connection is connection:
                del self._sinks[stream]

                self._dispatched.discard(stream)

                self._ready_sinks.discard(stream)

                self._released_sources.discard(stream)

    async def _dispatch_start(self, stream: StreamKey) -> None:
        """函数契约说明.

        功能: 执行 _dispatch_start 的异步逻辑,并协调
        get, add, correlation, send。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        source = self._sources.get(stream)

        sink = self._sinks.get(stream)

        if source is None or sink is None or stream in self._dispatched:
            return

        self._dispatched.add(stream)

        correlation = self._hub.correlation(stream)

        if correlation is None:
            return

        await sink.connection.send(
            _stream_command_envelope(
                stream, self._hub.output_ssrc(stream), sink, correlation
            )
        )

    async def _release_source(self, stream: StreamKey) -> None:
        """函数契约说明.

        功能: 执行 _release_source 的异步逻辑,并协调
        get, correlation, add, send。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        source = self._sources.get(stream)

        if (
            source is None
            or stream not in self._ready_sinks
            or stream in self._released_sources
        ):
            return

        correlation = self._hub.correlation(stream)

        if correlation is None:
            return

        self._released_sources.add(stream)

        await source.connection.send(
            _source_ready_envelope(stream, source.ssrc, correlation)
        )

    async def _deliver_flushes(self) -> None:
        """函数契约说明.

        功能: 执行 _deliver_flushes
        的异步逻辑,并协调 pop, get, correlation,
        _flush_envelope。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        while self._flush_outbox:
            flush = self._flush_outbox.pop(0)

            source = self._sources.get(flush.stream)

            sink = self._sinks.get(flush.stream)

            correlation = self._hub.correlation(flush.stream)

            if correlation is not None:
                envelope = _flush_envelope(flush, correlation)

                if source is not None:
                    await source.connection.send(envelope)

                if sink is not None:
                    await sink.connection.send(envelope)

    def _discard(self, stream: StreamKey) -> None:
        """函数契约说明.

        功能: 执行 _discard 的同步逻辑,并协调 pop,
        discard。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回 `None`。
        """
        _ = self._sources.pop(stream, None)

        _ = self._sinks.pop(stream, None)

        self._dispatched.discard(stream)

        self._ready_sinks.discard(stream)

        self._released_sources.discard(stream)

    def _record_playback(self, event: StreamState) -> None:
        """函数契约说明.

        功能: 执行 _record_playback
        的同步逻辑,并协调 StreamKey,
        record_stream, StageCorrelation,
        str。
        参数: self 表示当前实例。 event:
        StreamState。 必填。
        契约: 同步调用。 返回 `None`。
        """
        observability = self._observability

        if observability is not None:
            stream = StreamKey(event.session_id, event.stream_id)

            observability.record_stream(
                "playback_state",
                stream,
                command=StageCorrelation(
                    trace_id=event.correlation.trace_id,
                    session_id=event.correlation.session_id,
                    seq=event.correlation.seq,
                    turn_id=str(event.turn_id) if event.turn_id is not None else None,
                    segment_id=(
                        str(event.segment_id) if event.segment_id is not None else None
                    ),
                    cancellation_epoch=(
                        int(event.cancellation_epoch)
                        if event.cancellation_epoch is not None
                        else None
                    ),
                ),
            )

    def _record_flush(self, flush: StreamFlush) -> None:
        """函数契约说明.

        功能: 执行 _record_flush 的同步逻辑,并协调
        correlation, record_stream,
        StageCorrelation, str。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 同步调用。 返回 `None`。
        """
        observability = self._observability

        if observability is not None:
            correlation = flush.correlation or self._hub.correlation(flush.stream)

            if correlation is not None:
                observability.record_stream(
                    "flush",
                    flush.stream,
                    command=StageCorrelation(
                        trace_id=correlation.trace_id,
                        session_id=correlation.session_id,
                        seq=correlation.seq,
                        turn_id=str(flush.turn_id),
                        segment_id=str(flush.segment_id),
                        cancellation_epoch=int(flush.cancellation_epoch),
                    ),
                )

    def _record_flush_acknowledgement(
        self, acknowledgement: FlushAcknowledgement
    ) -> None:
        """函数契约说明.

        功能: 执行
        _record_flush_acknowledgement
        的同步逻辑,并协调 record_stream,
        StageCorrelation, str, int。
        参数: self 表示当前实例。
        acknowledgement:
        FlushAcknowledgement。 必填。
        契约: 同步调用。 返回 `None`。
        """
        observability = self._observability

        correlation = acknowledgement.correlation

        if observability is not None and correlation is not None:
            observability.record_stream(
                "flush_ack",
                acknowledgement.stream,
                command=StageCorrelation(
                    trace_id=correlation.trace_id,
                    session_id=correlation.session_id,
                    seq=correlation.seq,
                    turn_id=str(acknowledgement.turn_id),
                    segment_id=str(acknowledgement.segment_id),
                    cancellation_epoch=int(acknowledgement.cancellation_epoch),
                ),
            )

    def _record_transport_transition(
        self, stage: OnsiteStage, stream: StreamKey
    ) -> None:
        """函数契约说明.

        功能: 执行
        _record_transport_transition
        的同步逻辑,并协调 record_stream。
        参数: self 表示当前实例。 stage:
        OnsiteStage。 必填。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回 `None`。
        """
        observability = self._observability

        if observability is not None:
            observability.record_stream(stage, stream)


def _source_ready_envelope(
    stream: StreamKey, ssrc: int, correlation: EnvelopeCorrelation
) -> str:
    """函数契约说明.

    功能: 执行 _source_ready_envelope
    的同步逻辑,并协调 _envelope。
    参数: stream: StreamKey。 必填。 ssrc:
    int。 必填。 correlation:
    EnvelopeCorrelation。 必填。
    契约: 同步调用。 返回 `str`。
    """
    return _envelope(
        event_type="media.rtp.source.ready",
        correlation=correlation,
        data={"stream_id": stream.stream_id, "ssrc": ssrc},
    )


def _stream_command_envelope(
    stream: StreamKey,
    ssrc: int,
    sink: _SinkPeer,
    correlation: EnvelopeCorrelation | None,
    flush: StreamFlush | None = None,
) -> str:
    """函数契约说明.

    功能: 执行 _stream_command_envelope
    的同步逻辑,并协调 _envelope, RuntimeError,
    int, str。
    参数: stream: StreamKey。 必填。 ssrc:
    int。 必填。 sink: _SinkPeer。 必填。
    correlation: EnvelopeCorrelation |
    None。 必填。 flush: StreamFlush | None。
    可省略。
    契约: 同步调用。 返回 `str`。 可能抛出
    RuntimeError。
    """
    if correlation is None:
        message = "stream correlation is required"

        raise RuntimeError(message)

    data: dict[str, object] = {
        "command_id": f"rtp-{stream.stream_id}",
        "stream_id": stream.stream_id,
        "start_rtp_timestamp": 96_000,
        "ssrc": ssrc,
        "codec": _CODEC,
        "rtp_endpoint": {"host": sink.host, "port": sink.udp_port},
    }

    if flush is not None:
        data["cancellation_epoch"] = int(flush.cancellation_epoch)

    return _envelope(
        event_type="media.stream.command",
        correlation=correlation,
        turn_id=str(flush.turn_id) if flush is not None else None,
        segment_id=str(flush.segment_id) if flush is not None else None,
        data=data,
    )


def _cancel_envelope(stream_id: str, correlation: EnvelopeCorrelation) -> str:
    """函数契约说明.

    功能: 执行 _cancel_envelope 的同步逻辑,并协调
    _envelope。
    参数: stream_id: str。 必填。 correlation:
    EnvelopeCorrelation。 必填。
    契约: 同步调用。 返回 `str`。
    """
    return _envelope(
        event_type="cancel",
        correlation=correlation,
        segment_id=stream_id,
        data={"reason": "transport_cancelled"},
    )


def _flush_envelope(flush: StreamFlush, correlation: EnvelopeCorrelation) -> str:
    """函数契约说明.

    功能: 执行 _flush_envelope 的同步逻辑,并协调
    _envelope, str, int。
    参数: flush: StreamFlush。 必填。
    correlation: EnvelopeCorrelation。
    必填。
    契约: 同步调用。 返回 `str`。
    """
    return _envelope(
        event_type="media.stream.flush",
        correlation=correlation,
        turn_id=str(flush.turn_id),
        segment_id=str(flush.segment_id),
        data={
            "stream_id": flush.stream.stream_id,
            "cancellation_epoch": int(flush.cancellation_epoch),
            "request_id": str(flush.request_id),
            "target_generated_ssrc": int(flush.target_generated_ssrc),
        },
    )


def _connection_id(connection: ControlPeer) -> ConnectionId:
    """函数契约说明.

    功能: 执行 _connection_id 的同步逻辑,并协调
    ConnectionId, str, id。
    参数: connection: ControlPeer。 必填。
    契约: 同步调用。 返回 `ConnectionId`。
    """
    return ConnectionId(str(id(connection)))


def _envelope(
    *,
    event_type: str,
    correlation: EnvelopeCorrelation,
    data: dict[str, object],
    turn_id: str | None = None,
    segment_id: str | None = None,
) -> str:
    """函数契约说明.

    功能: 执行 _envelope 的同步逻辑,并协调 dumps。
    参数: event_type: str。 必填。
    correlation: EnvelopeCorrelation。
    必填。 data: dict[str, object]。 必填。
    turn_id: str | None。 可省略。
    segment_id: str | None。 可省略。
    契约: 同步调用。 返回 `str`。
    """
    envelope: dict[str, object] = {
        "schema_version": "1.0.0",
        "event_type": event_type,
        "event_id": f"transport-{event_type}-{correlation.session_id}",
        "source": "orchestrator",
        "time": "2026-07-27T00:00:00Z",
        "trace_id": correlation.trace_id,
        "session_id": correlation.session_id,
        "seq": correlation.seq,
        "data": data,
    }

    if segment_id is not None:
        envelope["segment_id"] = segment_id

    if turn_id is not None:
        envelope["turn_id"] = turn_id

    return json.dumps(envelope, separators=(",", ":"))


@final
class _MonotonicFlushClock:
    """类契约说明.

    职责: 定义 _MonotonicFlushClock
    的状态、行为和对外协作边界。
    契约: 方法: now_ms。
    """

    @property
    def now_ms(self) -> int:
        """函数契约说明.

        功能: 执行 now_ms 的同步逻辑,并协调 int,
        monotonic。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `int`。
        """
        return int(time.monotonic() * 1_000)
