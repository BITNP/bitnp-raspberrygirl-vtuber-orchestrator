"""Pinned RTP routing and WSS connection-owned route lifecycle."""

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
    """Sends one UDP datagram without owning route-selection policy."""

    def sendto(self, data: bytes, addr: PeerAddress) -> None:
        """Send bytes to the supplied IP address and UDP port."""

    def close(self) -> None:
        """Release the underlying UDP transport."""


class OnsiteBridge(Protocol):
    """Accepts Mic RTP and asynchronously emits generated replacement RTP."""

    def set_output_callback(
        self,
        callback: Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]],
    ) -> None:
        """Install the hub-owned generated RTP callback."""

    def submit_mic_rtp(
        self, stream: StreamKey, packet: bytes, epoch: CancellationEpoch
    ) -> None:
        """Submit one authenticated Mic RTP packet without awaiting output."""

    def invalidate_stream(
        self, stream: StreamKey, next_epoch: CancellationEpoch
    ) -> None:
        """Synchronously discard a route's retired actor state."""

    async def wait_quiescent(self) -> None:
        """Wait until active and invalidated stream work has settled."""


@dataclass(frozen=True, slots=True)
class RouteKey:
    """Pinned identity for one RTP source endpoint."""

    session_id: str
    stream_id: str
    ssrc: int
    peer_ip: str
    udp_port: int


@dataclass(frozen=True, slots=True)
class PendingSource:
    """Authenticated source awaiting its first valid pinned RTP datagram."""

    stream: StreamKey
    ssrc: int
    peer_ip: str


@dataclass(frozen=True, slots=True)
class DuplicateRouteError(Exception):
    """Raised when a live source or sink route is registered twice."""

    stream: StreamKey

    @override
    def __str__(self) -> str:
        return f"duplicate RTP route: {self.stream.session_id}/{self.stream.stream_id}"


@final
class RtpHub:
    """Routes exact V2/PT96/L16 RTP bytes between authenticated control peers."""

    def __init__(
        self,
        transport: DatagramSender | None = None,
        *,
        onsite_bridge: OnsiteBridge | None = None,
    ) -> None:
        """Create an empty hub, optionally with an injected UDP sender."""
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
            onsite_bridge.set_output_callback(self._deliver_onsite_packet)

    def attach_transport(self, transport: DatagramSender) -> None:
        """Attach the UDP sender created by the runtime listener."""
        self._transport = transport

    def set_observability(self, observability: OnsiteObservability) -> None:
        """Attach the shared recorder without widening route construction inputs."""
        self._observability = observability

    def set_output_fence(self, output_fence: SchedulerOutputFence) -> None:
        """Bind generated RTP delivery to the session scheduler's reflex fence."""
        self._output_fence = output_fence

    @property
    def route_ready(self) -> bool:
        """Return whether at least one Mic source has a paired Sound route."""
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
        """Parse and apply one control envelope from an authenticated WSS peer."""
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
        """Forward a valid packet only after matching an authenticated source route."""
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

    async def _deliver_onsite_packet(
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

    async def deliver_generated_rtp(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        """Deliver generated RTP through every route and scheduler output fence."""
        await self._deliver_onsite_packet(stream, epoch, packet)

    def _record_rtp(self, stage: OnsiteStage, stream: StreamKey) -> None:
        observability = self._observability
        if observability is not None:
            observability.record_stream(stage, stream)

    async def wait_for_onsite_jobs(self) -> None:
        """Wait for all accepted onsite provider jobs to settle."""
        actors = self._onsite_actors
        if actors is not None:
            await actors.wait_quiescent()
        bridge = self._onsite_bridge
        if bridge is not None:
            await bridge.wait_quiescent()

    def remove_connection(self, owner: ConnectionId) -> None:
        """Remove exactly the source and sink components owned by one WSS session."""
        for stream, route_owner in tuple(self._source_owners.items()):
            if route_owner == owner:
                self._remove_source(stream)
        for stream, route_owner in tuple(self._sink_owners.items()):
            if route_owner == owner:
                self._remove_sink(stream)

    def clear(self) -> None:
        """Remove all routes when the runtime relinquishes its listeners."""
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
        """Remove one stream route before its cancellation reaches the sink."""
        self._remove_stream(StreamKey(session_id, stream_id))

    def output_ssrc(self, stream: StreamKey, cancellation_epoch: int = 0) -> int:
        """Return the SSRC Sound must accept for this transport mode."""
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
        """Return the authenticated source-envelope correlation for one live route."""
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
        self._invalidate_stream(stream)
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
