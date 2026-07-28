"""Pinned RTP routing and WSS connection-owned route lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, final, override

from orchestrator.transport_control import (
    SinkRegistration,
    SourceRegistration,
    StreamReady,
    StreamState,
    parse_control_event,
)

if TYPE_CHECKING:
    from orchestrator.ids import ConnectionId

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


@dataclass(frozen=True, slots=True)
class RouteKey:
    """Pinned identity for one RTP source endpoint."""

    session_id: str
    stream_id: str
    ssrc: int
    peer_ip: str
    udp_port: int


@dataclass(frozen=True, slots=True)
class StreamKey:
    """Stable control-plane identity for one media stream."""

    session_id: str
    stream_id: str


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

    def __init__(self, transport: DatagramSender | None = None) -> None:
        """Create an empty hub, optionally with an injected UDP sender."""
        self._transport: DatagramSender | None = transport
        self._pending_sources: dict[StreamKey, PendingSource] = {}
        self._pinned_sources: dict[RouteKey, StreamKey] = {}
        self._sinks: dict[StreamKey, PeerAddress] = {}
        self._source_owners: dict[StreamKey, ConnectionId] = {}
        self._sink_owners: dict[StreamKey, ConnectionId] = {}

    def attach_transport(self, transport: DatagramSender) -> None:
        """Attach the UDP sender created by the runtime listener."""
        self._transport = transport

    def register_control(
        self,
        raw_message: str,
        peer_ip: str,
        owner: ConnectionId | None = None,
    ) -> None:
        """Parse and apply one control envelope from an authenticated WSS peer."""
        event = parse_control_event(raw_message)
        match event:
            case SourceRegistration(
                session_id=session_id, stream_id=stream_id, ssrc=ssrc
            ):
                self._register_source(
                    StreamKey(session_id, stream_id), ssrc, peer_ip, owner
                )
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
            case StreamReady() | StreamState():
                return

    def route_datagram(self, data: bytes, peer: PeerAddress) -> bool:
        """Forward a valid packet only after matching an authenticated source route."""
        if not _is_canonical_rtp(data):
            return False
        stream = self._find_route(_rtp_ssrc(data), peer)
        if stream is None:
            return False
        sink = self._sinks.get(stream)
        if sink is None or self._transport is None:
            return False
        self._transport.sendto(data, sink)
        return True

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
        self._pending_sources.clear()
        self._pinned_sources.clear()
        self._sinks.clear()
        self._source_owners.clear()
        self._sink_owners.clear()

    def remove_stream(self, session_id: str, stream_id: str) -> None:
        """Remove one stream route before its cancellation reaches the sink."""
        self._remove_stream(StreamKey(session_id, stream_id))

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
        _ = self._pending_sources.pop(stream, None)
        _ = self._source_owners.pop(stream, None)
        for route, route_stream in tuple(self._pinned_sources.items()):
            if route_stream == stream:
                del self._pinned_sources[route]

    def _remove_sink(self, stream: StreamKey) -> None:
        _ = self._sinks.pop(stream, None)
        _ = self._sink_owners.pop(stream, None)

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
