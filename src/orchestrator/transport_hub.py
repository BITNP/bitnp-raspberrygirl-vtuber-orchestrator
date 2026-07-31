
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, final, override

from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    GeneratedSsrc,
    SegmentId,
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

    def sendto(self, data: bytes, addr: PeerAddress) -> None:
        ...

    def close(self) -> None:
        ...


class OnsiteBridge(Protocol):

    def set_output_callback(
        self,
        callback: Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]],
    ) -> None:
        ...

    def set_output_authorizer(
        self, callback: Callable[[StreamKey, CancellationEpoch], bool]
    ) -> None:
        ...

    def submit_mic_rtp(
        self, stream: StreamKey, packet: bytes, epoch: CancellationEpoch
    ) -> None:
        ...

    def invalidate_stream(
        self, stream: StreamKey, next_epoch: CancellationEpoch
    ) -> None:
        ...

    def disconnect_stream(self, stream: StreamKey) -> None:
        ...

    async def wait_quiescent(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class RouteKey:

    session_id: str

    stream_id: str

    ssrc: int

    peer_ip: str

    udp_port: int


@dataclass(frozen=True, slots=True)
class PendingSource:

    stream: StreamKey

    ssrc: int

    peer_ip: str


@dataclass(frozen=True, slots=True)
class DuplicateRouteError(Exception):

    stream: StreamKey

    @override
    def __str__(self) -> str:
        return f"duplicate RTP route: {self.stream.session_id}/{self.stream.stream_id}"


@final
class RtpHub:

    def __init__(
        self,
        transport: DatagramSender | None = None,
        *,
        onsite_bridge: OnsiteBridge | None = None,
    ) -> None:
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
        self._transport = transport

    def set_observability(self, observability: OnsiteObservability) -> None:
        self._observability = observability

    def set_output_fence(self, output_fence: SchedulerOutputFence) -> None:
        self._output_fence = output_fence

        bridge = self._onsite_bridge

        if bridge is not None:
            bridge.set_output_authorizer(self.authorize_onsite_output)

    def authorize_onsite_output(
        self, stream: StreamKey, epoch: CancellationEpoch
    ) -> bool:
        """Activate a scheduler lease for a finalized onsite utterance."""
        output_fence = self._output_fence

        if output_fence is None:
            return True

        correlation = self._correlations.get(stream)

        if correlation is None:
            return False

        lease = output_fence.activate(
            stream=stream,
            segment_id=SegmentId(f"onsite-{stream.stream_id}-{int(epoch)}"),
            target_generated_ssrc=GeneratedSsrc(generated_ssrc(stream, epoch)),
            correlation=correlation,
        )

        return lease.cancellation_epoch == epoch

    @property
    def route_ready(self) -> bool:
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
        if not _is_canonical_rtp(data):
            return False

        stream = self._find_route(_rtp_ssrc(data), peer)

        if stream is None:
            return False

        self._record_rtp("rtp_ingress", stream)

        sink = self._sinks.get(stream)

        if sink is None or self._transport is None:
            return False

        if self._onsite_bridge is None:
            return False

        return self._route_onsite(data, stream)

    def _route_onsite(self, data: bytes, stream: StreamKey) -> bool:
        actors = self._onsite_actors

        if actors is None:
            actors = StreamPipelineActors(self._process_onsite_frame)

            self._onsite_actors = actors

        actors.submit(stream, data)

        return False

    async def _process_onsite_frame(self, stream: StreamKey, frame: bytes) -> None:
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
        observability = self._observability

        if observability is not None:
            observability.record_stream(stage, stream)

    async def wait_for_onsite_jobs(self) -> None:
        actors = self._onsite_actors

        if actors is not None:
            await actors.wait_quiescent()

        bridge = self._onsite_bridge

        if bridge is not None:
            await bridge.wait_quiescent()

    def remove_connection(self, owner: ConnectionId) -> None:
        for stream, route_owner in tuple(self._source_owners.items()):
            if route_owner == owner:
                self._remove_source(stream)

        for stream, route_owner in tuple(self._sink_owners.items()):
            if route_owner == owner:
                self._remove_sink(stream)

    def clear(self) -> None:
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
        self._remove_stream(StreamKey(session_id, stream_id))

    def output_ssrc(self, stream: StreamKey, cancellation_epoch: int = 0) -> int:
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
        return self._correlations.get(stream)

    def _register_source(
        self,
        stream: StreamKey,
        ssrc: int,
        peer_ip: str,
        owner: ConnectionId | None,
    ) -> None:
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
        if stream in self._sinks:
            raise DuplicateRouteError(stream)

        self._sinks[stream] = endpoint

        if owner is not None:
            self._sink_owners[stream] = owner

    def _remove_stream(self, stream: StreamKey) -> None:
        self._remove_source(stream)

        self._remove_sink(stream)

    def _remove_source(self, stream: StreamKey) -> None:
        if self._onsite_bridge is not None:
            self._onsite_bridge.disconnect_stream(stream)

        _ = self._pending_sources.pop(stream, None)

        _ = self._correlations.pop(stream, None)

        _ = self._source_owners.pop(stream, None)

        for route, route_stream in tuple(self._pinned_sources.items()):
            if route_stream == stream:
                del self._pinned_sources[route]

    def _remove_sink(self, stream: StreamKey) -> None:
        self._invalidate_stream(stream)

        _ = self._sinks.pop(stream, None)

        _ = self._sink_owners.pop(stream, None)

    def _invalidate_stream(self, stream: StreamKey) -> None:
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
    return (
        len(data) == RTP_HEADER_BYTES + L16_FRAME_BYTES
        and data[0] == RTP_V2_HEADER
        and data[1] & 0x7F == RTP_PAYLOAD_TYPE
    )


def _rtp_ssrc(data: bytes) -> int:
    return int.from_bytes(data[8:12])
