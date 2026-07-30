"""模块契约说明.

职责: 提供 orchestrator.transport_hub
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, final, override

from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    StreamFlush,
    StreamKey,
)
from orchestrator.streaming_pipeline_actors import StreamPipelineActors
from orchestrator.transport_control import (
    ControlEvent,
    EnvelopeCorrelation,
    SinkRegistration,
    SourceRegistration,
    StreamReady,
    StreamState,
    parse_control_event,
)
from orchestrator.tts_rtp import generated_ssrc

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from orchestrator.ids import ConnectionId
    from orchestrator.observability import OnsiteObservability, OnsiteStage
    from orchestrator.scheduler_reflex import SchedulerOutputFence


type PeerAddress = tuple[str, int]


RTP_HEADER_BYTES = 12

L16_FRAME_BYTES = 640

RTP_V2_HEADER = 0x80

RTP_PAYLOAD_TYPE = 96


class DatagramSender(Protocol):
    """类契约说明.

    职责: 声明 DatagramSender
    协议接口,约束实现方必须提供的行为。
    契约: 方法: sendto、close。
    """

    def sendto(self, data: bytes, addr: PeerAddress) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 data: bytes。
        必填。 addr: PeerAddress。 必填。
        契约: 同步调用。 返回 `None`。
        """

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """


class OnsiteBridge(Protocol):
    """类契约说明.

    职责: 声明 OnsiteBridge
    协议接口,约束实现方必须提供的行为。
    契约: 方法: set_output_callback、submit_m
    ic_rtp、invalidate_stream、wait_quiesc
    ent。
    """

    def set_output_callback(
        self,
        callback: Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]],
    ) -> None:
        """函数契约说明.

        功能: 执行 set_output_callback
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 callback:
        Callable[[StreamKey,
        CancellationEpoch, bytes],
        Awaitable[None]]。 必填。
        契约: 同步调用。 返回 `None`。
        """

    def submit_mic_rtp(
        self, stream: StreamKey, packet: bytes, epoch: CancellationEpoch
    ) -> None:
        """函数契约说明.

        功能: 执行 submit_mic_rtp
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 packet: bytes。
        必填。 epoch: CancellationEpoch。
        必填。
        契约: 同步调用。 返回 `None`。
        """

    def invalidate_stream(
        self, stream: StreamKey, next_epoch: CancellationEpoch
    ) -> None:
        """函数契约说明.

        功能: 执行 invalidate_stream
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 next_epoch:
        CancellationEpoch。 必填。
        契约: 同步调用。 返回 `None`。
        """

    async def wait_quiescent(self) -> None:
        """函数契约说明.

        功能: 执行 wait_quiescent
        的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """


@dataclass(frozen=True, slots=True)
class RouteKey:
    """类契约说明.

    职责: 保存 RouteKey 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: session_id、stream_id、ssrc、pe
    er_ip、udp_port。
    """

    session_id: str

    stream_id: str

    ssrc: int

    peer_ip: str

    udp_port: int


@dataclass(frozen=True, slots=True)
class PendingSource:
    """类契约说明.

    职责: 保存 PendingSource
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、ssrc、peer_ip。
    """

    stream: StreamKey

    ssrc: int

    peer_ip: str


@dataclass(frozen=True, slots=True)
class DuplicateRouteError(Exception):
    """类契约说明.

    职责: 保存 DuplicateRouteError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream。 方法: __str__。
    """

    stream: StreamKey

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"duplicate RTP route: {self.stream.session_id}/{self.stream.stream_id}"


@final
class RtpHub:
    """类契约说明.

    职责: 定义 RtpHub 的状态、行为和对外协作边界。
    契约: 方法: __init__、attach_transport、se
    t_observability、set_output_fence、rou
    te_ready、register_control。
    """

    def __init__(
        self,
        transport: DatagramSender | None = None,
        *,
        onsite_bridge: OnsiteBridge | None = None,
    ) -> None:
        """函数契约说明.

        功能: 初始化 RtpHub 的字段并建立实例不变式。
        参数: self 表示当前实例。 transport:
        DatagramSender | None。 可省略。
        onsite_bridge: OnsiteBridge |
        None。 可省略。
        契约: 同步调用。 返回 `None`。
        """
        self._transport: DatagramSender | None = transport

        self._onsite_bridge: OnsiteBridge | None = onsite_bridge

        self._output_fence: SchedulerOutputFence | None = None

        self._observability: OnsiteObservability | None = None

        self._correlations: dict[StreamKey, EnvelopeCorrelation] = {}

        self._pending_sources: dict[StreamKey, PendingSource] = {}

        self._pinned_sources: dict[RouteKey, StreamKey] = {}

        self._sinks: dict[StreamKey, PeerAddress] = {}

        self._source_owners: dict[StreamKey, ConnectionId] = {}

        self._sink_owners: dict[StreamKey, ConnectionId] = {}

        self._onsite_actors: StreamPipelineActors | None = None

        self._route_generations: dict[StreamKey, int] = {}

        if onsite_bridge is not None:
            onsite_bridge.set_output_callback(self.deliver_generated_rtp)

    def attach_transport(self, transport: DatagramSender) -> None:
        """函数契约说明.

        功能: 执行 attach_transport
        的同步逻辑,并产出 _transport。
        参数: self 表示当前实例。 transport:
        DatagramSender。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._transport = transport

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

    @property
    def route_ready(self) -> bool:
        """函数契约说明.

        功能: 执行 route_ready 的同步逻辑,并协调
        any, values。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `bool`。
        """
        source_streams = {*self._pending_sources.values()}

        return any(source.stream in self._sinks for source in source_streams) or any(
            stream in self._sinks for stream in self._pinned_sources.values()
        )

    def register_control(
        self,
        raw_message: ControlEvent | str,
        peer_ip: str,
        owner: ConnectionId | None = None,
    ) -> None:
        """函数契约说明.

        功能: 执行 register_control
        的同步逻辑,并协调 isinstance,
        parse_control_event, StreamKey,
        _register_source。
        参数: self 表示当前实例。 raw_message:
        ControlEvent | str。 必填。 peer_ip:
        str。 必填。 owner: ConnectionId |
        None。 可省略。
        契约: 同步调用。 返回 `None`。
        """
        parsed_event = (
            parse_control_event(raw_message)
            if isinstance(raw_message, str)
            else raw_message
        )

        match parsed_event:
            case SourceRegistration(
                session_id=session_id, stream_id=stream_id, ssrc=ssrc
            ):
                stream = StreamKey(session_id, stream_id)

                self._register_source(stream, ssrc, peer_ip, owner)

                self._correlations[stream] = parsed_event.correlation

                observability = self._observability

                if observability is not None:
                    observability.bind_correlation(stream, parsed_event.correlation)

            case SinkRegistration(
                session_id=session_id, stream_id=stream_id, udp_port=udp_port
            ):
                self._register_sink(
                    StreamKey(session_id, stream_id), (peer_ip, udp_port), owner
                )

            case StreamState(
                session_id=session_id,
                stream_id=stream_id,
                state="cancelled" | "finished" | "error",
            ):
                self._remove_stream(StreamKey(session_id, stream_id))

            case StreamReady() | StreamState() | StreamFlush() | FlushAcknowledgement():
                return

    def route_datagram(self, data: bytes, peer: PeerAddress) -> bool:
        """函数契约说明.

        功能: 执行 route_datagram 的同步逻辑,并协调
        _find_route, _record_rtp, get,
        sendto。
        参数: self 表示当前实例。 data: bytes。
        必填。 peer: PeerAddress。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        if not _is_canonical_rtp(data):
            return False

        stream = self._find_route(_rtp_ssrc(data), peer)

        if stream is None:
            return False

        self._record_rtp("rtp_ingress", stream)

        sink = self._sinks.get(stream)

        if sink is None or self._transport is None:
            return False

        if self._onsite_bridge is not None:
            return self._route_onsite(data, stream)

        self._transport.sendto(data, sink)

        return True

    def _route_onsite(self, data: bytes, stream: StreamKey) -> bool:
        """函数契约说明.

        功能: 执行 _route_onsite 的同步逻辑,并协调
        submit, StreamPipelineActors。
        参数: self 表示当前实例。 data: bytes。
        必填。 stream: StreamKey。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        actors = self._onsite_actors

        if actors is None:
            actors = StreamPipelineActors(self._process_onsite_frame)

            self._onsite_actors = actors

        actors.submit(stream, data)

        return False

    async def _process_onsite_frame(self, stream: StreamKey, frame: bytes) -> None:
        """函数契约说明.

        功能: 执行 _process_onsite_frame
        的异步逻辑,并协调 submit_mic_rtp,
        CancellationEpoch, get。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 frame: bytes。 必填。
        契约: 异步调用。 返回 `None`。
        """
        bridge = self._onsite_bridge

        if bridge is not None:
            bridge.submit_mic_rtp(
                stream,
                frame,
                CancellationEpoch(self._route_generations.get(stream, 0)),
            )

    async def deliver_generated_rtp(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        """函数契约说明.

        功能: 执行 deliver_generated_rtp
        的异步逻辑,并协调 get, sendto,
        _record_rtp, _is_canonical_rtp。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 epoch:
        CancellationEpoch。 必填。 packet:
        bytes。 必填。
        契约: 异步调用。 返回 `None`。
        """
        output_fence = self._output_fence

        if output_fence is None and epoch != CancellationEpoch(
            self._route_generations.get(stream, 0)
        ):
            return

        if output_fence is not None and not output_fence.can_emit(stream, epoch):
            return

        if not _is_canonical_rtp(packet):
            return

        sink = self._sinks.get(stream)

        transport = self._transport

        if sink is None or transport is None:
            return

        transport.sendto(packet, sink)

        self._record_rtp("rtp_egress", stream)

    def _record_rtp(self, stage: OnsiteStage, stream: StreamKey) -> None:
        """函数契约说明.

        功能: 执行 _record_rtp 的同步逻辑,并协调
        record_stream。
        参数: self 表示当前实例。 stage:
        OnsiteStage。 必填。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回 `None`。
        """
        observability = self._observability

        if observability is not None:
            observability.record_stream(stage, stream)

    async def wait_for_onsite_jobs(self) -> None:
        """函数契约说明.

        功能: 执行 wait_for_onsite_jobs
        的异步逻辑,并协调 wait_quiescent。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        actors = self._onsite_actors

        if actors is not None:
            await actors.wait_quiescent()

        bridge = self._onsite_bridge

        if bridge is not None:
            await bridge.wait_quiescent()

    def remove_connection(self, owner: ConnectionId) -> None:
        """函数契约说明.

        功能: 执行 remove_connection
        的同步逻辑,并协调 tuple, items,
        _remove_source, _remove_sink。
        参数: self 表示当前实例。 owner:
        ConnectionId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        for stream, route_owner in tuple(self._source_owners.items()):
            if route_owner == owner:
                self._remove_source(stream)

        for stream, route_owner in tuple(self._sink_owners.items()):
            if route_owner == owner:
                self._remove_sink(stream)

    def clear(self) -> None:
        """函数契约说明.

        功能: 执行 clear 的同步逻辑,并协调 clear,
        _invalidate_stream。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        actors = self._onsite_actors

        if actors is not None:
            for stream in actors.streams:
                self._invalidate_stream(stream)

        self._pending_sources.clear()

        self._correlations.clear()

        self._pinned_sources.clear()

        self._sinks.clear()

        self._source_owners.clear()

        self._sink_owners.clear()

    def remove_stream(self, session_id: str, stream_id: str) -> None:
        """函数契约说明.

        功能: 执行 remove_stream 的同步逻辑,并协调
        _remove_stream, StreamKey。
        参数: self 表示当前实例。 session_id:
        str。 必填。 stream_id: str。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._remove_stream(StreamKey(session_id, stream_id))

    def output_ssrc(self, stream: StreamKey, cancellation_epoch: int = 0) -> int:
        """函数契约说明.

        功能: 执行 output_ssrc 的同步逻辑,并协调
        generated_ssrc, next, items,
        CancellationEpoch。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        cancellation_epoch: int。 可省略。
        契约: 同步调用。 返回 `int`。
        """
        if self._onsite_bridge is None:
            source = next(
                (
                    pending.ssrc
                    for pending in self._pending_sources.values()
                    if pending.stream == stream
                ),
                None,
            )

            if source is not None:
                return source

            for route, route_stream in self._pinned_sources.items():
                if route_stream == stream:
                    return route.ssrc

            return 0

        return generated_ssrc(stream, CancellationEpoch(cancellation_epoch))

    def correlation(self, stream: StreamKey) -> EnvelopeCorrelation | None:
        """函数契约说明.

        功能: 执行 correlation 的同步逻辑,并协调
        get。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回
        `EnvelopeCorrelation | None`。
        """
        return self._correlations.get(stream)

    def _register_source(
        self,
        stream: StreamKey,
        ssrc: int,
        peer_ip: str,
        owner: ConnectionId | None,
    ) -> None:
        """函数契约说明.

        功能: 执行 _register_source
        的同步逻辑,并协调 any, PendingSource,
        DuplicateRouteError, values。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 ssrc: int。 必填。
        peer_ip: str。 必填。 owner:
        ConnectionId | None。 必填。
        契约: 同步调用。 返回 `None`。 可能抛出
        DuplicateRouteError。
        """
        if stream in self._pending_sources or stream in self._pinned_sources.values():
            raise DuplicateRouteError(stream)

        if any(
            source.ssrc == ssrc and source.peer_ip == peer_ip
            for source in self._pending_sources.values()
        ):
            raise DuplicateRouteError(stream)

        self._pending_sources[stream] = PendingSource(stream, ssrc, peer_ip)

        if owner is not None:
            self._source_owners[stream] = owner

    def _register_sink(
        self,
        stream: StreamKey,
        endpoint: PeerAddress,
        owner: ConnectionId | None,
    ) -> None:
        """函数契约说明.

        功能: 执行 _register_sink 的同步逻辑,并协调
        DuplicateRouteError。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 endpoint:
        PeerAddress。 必填。 owner:
        ConnectionId | None。 必填。
        契约: 同步调用。 返回 `None`。 可能抛出
        DuplicateRouteError。
        """
        if stream in self._sinks:
            raise DuplicateRouteError(stream)

        self._sinks[stream] = endpoint

        if owner is not None:
            self._sink_owners[stream] = owner

    def _remove_stream(self, stream: StreamKey) -> None:
        """函数契约说明.

        功能: 执行 _remove_stream 的同步逻辑,并协调
        _remove_source, _remove_sink。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._remove_source(stream)

        self._remove_sink(stream)

    def _remove_source(self, stream: StreamKey) -> None:
        """函数契约说明.

        功能: 执行 _remove_source 的同步逻辑,并协调
        _invalidate_stream, pop, tuple,
        items。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._invalidate_stream(stream)

        _ = self._pending_sources.pop(stream, None)

        _ = self._correlations.pop(stream, None)

        _ = self._source_owners.pop(stream, None)

        for route, route_stream in tuple(self._pinned_sources.items()):
            if route_stream == stream:
                del self._pinned_sources[route]

    def _remove_sink(self, stream: StreamKey) -> None:
        """函数契约说明.

        功能: 执行 _remove_sink 的同步逻辑,并协调
        _invalidate_stream, pop。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._invalidate_stream(stream)

        _ = self._sinks.pop(stream, None)

        _ = self._sink_owners.pop(stream, None)

    def _invalidate_stream(self, stream: StreamKey) -> None:
        """函数契约说明.

        功能: 执行 _invalidate_stream
        的同步逻辑,并协调 get,
        invalidate_stream, discard,
        CancellationEpoch。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。
        契约: 同步调用。 返回 `None`。
        """
        next_generation = self._route_generations.get(stream, 0) + 1

        self._route_generations[stream] = next_generation

        if self._onsite_bridge is not None:
            self._onsite_bridge.invalidate_stream(
                stream, CancellationEpoch(next_generation)
            )

        actors = self._onsite_actors

        if actors is not None:
            _ = actors.discard(stream)

    def _find_route(self, ssrc: int, peer: PeerAddress) -> StreamKey | None:
        """函数契约说明.

        功能: 执行 _find_route 的同步逻辑,并协调
        items, RouteKey, len, values。
        参数: self 表示当前实例。 ssrc: int。 必填。
        peer: PeerAddress。 必填。
        契约: 同步调用。 返回 `StreamKey | None`。
        """
        for route, stream in self._pinned_sources.items():
            if route.ssrc == ssrc and (route.peer_ip, route.udp_port) == peer:
                return stream

        candidates = [
            source
            for source in self._pending_sources.values()
            if source.ssrc == ssrc and source.peer_ip == peer[0]
        ]

        if len(candidates) != 1:
            return None

        source = candidates[0]

        route = RouteKey(
            source.stream.session_id,
            source.stream.stream_id,
            ssrc,
            peer[0],
            peer[1],
        )

        self._pinned_sources[route] = source.stream

        del self._pending_sources[source.stream]

        return source.stream


def _is_canonical_rtp(data: bytes) -> bool:
    """函数契约说明.

    功能: 执行 _is_canonical_rtp 的同步逻辑,并协调
    len。
    参数: data: bytes。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    return (
        len(data) == RTP_HEADER_BYTES + L16_FRAME_BYTES
        and data[0] == RTP_V2_HEADER
        and data[1] & 0x7F == RTP_PAYLOAD_TYPE
    )


def _rtp_ssrc(data: bytes) -> int:
    """函数契约说明.

    功能: 执行 _rtp_ssrc 的同步逻辑,并协调
    from_bytes。
    参数: data: bytes。 必填。
    契约: 同步调用。 返回 `int`。
    """
    return int.from_bytes(data[8:12])
