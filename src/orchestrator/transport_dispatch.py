"""Outbound control dispatch for the Orchestrator-owned RTP transport."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from orchestrator.ids import ConnectionId
from orchestrator.transport_control import (
    SinkRegistration,
    SourceRegistration,
    StreamState,
    parse_control_event,
)

_CODEC = {
    "format": "L16",
    "clock_rate_hz": 16_000,
    "channels": 1,
    "payload_type": 96,
    "samples_per_frame": 320,
}


class ControlPeer(Protocol):
    """A live control session that can receive canonical text envelopes."""

    async def send(self, message: str) -> None:
        """Send one canonical control envelope."""


class RouteRegistry(Protocol):
    """Applies inbound route registrations and removes cancelled stream routes."""

    def register_control(
        self, raw_message: str, peer_ip: str, owner: ConnectionId | None = None
    ) -> None:
        """Apply one authenticated inbound control envelope."""

    def remove_connection(self, owner: ConnectionId) -> None:
        """Remove route components owned by one closed WSS connection."""

    def remove_stream(self, session_id: str, stream_id: str) -> None:
        """Remove one stream route before cancellation reaches Sound."""


@dataclass(frozen=True, slots=True)
class StreamKey:
    """Identity of one Mic-to-Sound control route."""

    session_id: str
    stream_id: str


@dataclass(frozen=True, slots=True)
class _SourcePeer:
    connection: ControlPeer
    ssrc: int


@dataclass(frozen=True, slots=True)
class _SinkPeer:
    connection: ControlPeer
    host: str
    udp_port: int


class TransportControlDispatch:
    """Coordinates Mic and Sound sessions through the hub control boundary."""

    def __init__(self, hub: RouteRegistry) -> None:
        """Create a dispatcher bound to the authoritative hub route registry."""
        self._hub: RouteRegistry = hub
        self._sources: dict[StreamKey, _SourcePeer] = {}
        self._sinks: dict[StreamKey, _SinkPeer] = {}
        self._dispatched: set[StreamKey] = set()

    async def register(
        self, raw_message: str, peer_ip: str, connection: ControlPeer
    ) -> None:
        """Dispatch startup messages after both peers have registered."""
        event = parse_control_event(raw_message)
        self._hub.register_control(raw_message, peer_ip, _connection_id(connection))
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
            case StreamState(
                session_id=session_id,
                stream_id=stream_id,
                state="cancelled" | "finished" | "error",
            ):
                self._discard(StreamKey(session_id, stream_id))
            case StreamState():
                return
            case _:
                return
        await self._dispatch_start(StreamKey(event.session_id, event.stream_id))

    async def cancel_stream(self, session_id: str, stream_id: str) -> None:
        """Stop forwarding first, then signal Sound to suppress late RTP."""
        stream = StreamKey(session_id, stream_id)
        sink = self._sinks.get(stream)
        self._hub.remove_stream(session_id, stream_id)
        self._discard(stream)
        if sink is not None:
            await sink.connection.send(_cancel_envelope(session_id, stream_id))

    def clear(self) -> None:
        """Release retained live-session references during runtime shutdown."""
        self._sources.clear()
        self._sinks.clear()
        self._dispatched.clear()

    def remove_connection(self, connection: ControlPeer) -> None:
        """Discard only control and RTP state owned by one closed WSS peer."""
        self._hub.remove_connection(_connection_id(connection))
        for stream, source in tuple(self._sources.items()):
            if source.connection is connection:
                del self._sources[stream]
                self._dispatched.discard(stream)
        for stream, sink in tuple(self._sinks.items()):
            if sink.connection is connection:
                del self._sinks[stream]
                self._dispatched.discard(stream)

    async def _dispatch_start(self, stream: StreamKey) -> None:
        source = self._sources.get(stream)
        sink = self._sinks.get(stream)
        if source is None or sink is None or stream in self._dispatched:
            return
        self._dispatched.add(stream)
        await source.connection.send(_source_ready_envelope(stream, source.ssrc))
        await sink.connection.send(_stream_command_envelope(stream, source.ssrc, sink))

    def _discard(self, stream: StreamKey) -> None:
        _ = self._sources.pop(stream, None)
        _ = self._sinks.pop(stream, None)
        self._dispatched.discard(stream)


def _source_ready_envelope(stream: StreamKey, ssrc: int) -> str:
    return _envelope(
        event_type="media.rtp.source.ready",
        session_id=stream.session_id,
        data={"stream_id": stream.stream_id, "ssrc": ssrc},
    )


def _stream_command_envelope(stream: StreamKey, ssrc: int, sink: _SinkPeer) -> str:
    return _envelope(
        event_type="media.stream.command",
        session_id=stream.session_id,
        segment_id=stream.stream_id,
        data={
            "command_id": f"rtp-{stream.stream_id}",
            "stream_id": stream.stream_id,
            "start_rtp_timestamp": 96_000,
            "ssrc": ssrc,
            "codec": _CODEC,
            "rtp_endpoint": {"host": sink.host, "port": sink.udp_port},
        },
    )


def _cancel_envelope(session_id: str, stream_id: str) -> str:
    return _envelope(
        event_type="cancel",
        session_id=session_id,
        segment_id=stream_id,
        data={"reason": "transport_cancelled"},
    )


def _connection_id(connection: ControlPeer) -> ConnectionId:
    return ConnectionId(str(id(connection)))


def _envelope(
    *,
    event_type: str,
    session_id: str,
    data: dict[str, object],
    segment_id: str | None = None,
) -> str:
    envelope: dict[str, object] = {
        "schema_version": "1.0.0",
        "event_type": event_type,
        "event_id": f"transport-{event_type}-{session_id}",
        "source": "orchestrator",
        "time": "2026-07-27T00:00:00Z",
        "trace_id": "transport-runtime",
        "session_id": session_id,
        "seq": 0,
        "data": data,
    }
    if segment_id is not None:
        envelope["segment_id"] = segment_id
    return json.dumps(envelope, separators=(",", ":"))
