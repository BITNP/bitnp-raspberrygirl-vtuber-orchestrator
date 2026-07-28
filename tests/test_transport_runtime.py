from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import pytest

from orchestrator.config import TrustedLanToken
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_control import AuthenticatedControl
from orchestrator.transport_hub import DuplicateRouteError, RtpHub
from orchestrator.transport_runtime import (
    ControlHandler,
    ControlListener,
    DatagramListener,
    TransportRuntime,
)

if TYPE_CHECKING:
    from orchestrator.json_boundary import JsonValue

SESSION_ID: Final = "session-transport-001"
STREAM_ID: Final = "mic-stream-001"
SSRC: Final = 0x12345678
SOURCE_PEER: Final = ("192.0.2.10", 41_000)
SINK_PEER: Final = ("192.0.2.11", 41_001)


@dataclass
class FakeDatagramTransport:
    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)
    closed: bool = False

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        self.sent.append((data, addr))

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeControlServer:
    closed: bool = False
    waited: bool = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.waited = True


def test_hub_forwards_only_valid_pinned_v2_pt96_l16_packets() -> None:
    # Given: authenticated Mic and Sound registrations for one canonical stream.
    transport = FakeDatagramTransport()
    hub = RtpHub(transport)
    hub.register_control(_source_registration(), SOURCE_PEER[0])
    hub.register_control(_sink_registration(), SINK_PEER[0])
    packet = _rtp_packet(payload_type=96)

    # When: the first valid Mic packet arrives from its authenticated peer address.
    forwarded = hub.route_datagram(packet, SOURCE_PEER)

    # Then: the unchanged V2 PT96 L16 packet reaches Sound's authenticated IP and port.
    assert forwarded is True
    assert transport.sent == [(packet, (SINK_PEER[0], 5006))]


def test_hub_accepts_canonical_source_and_sink_ready_events() -> None:
    # Given: registered Mic and Sound routes for a canonical media stream.
    transport = FakeDatagramTransport()
    hub = RtpHub(transport)
    hub.register_control(_source_registration(), SOURCE_PEER[0])
    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: both peers acknowledge their canonical RTP readiness events.
    hub.register_control(_source_ready(), SOURCE_PEER[0])
    hub.register_control(_sink_ready(), SINK_PEER[0])

    # Then: readiness is accepted without changing the registered forward route.
    assert hub.route_datagram(_rtp_packet(payload_type=96), SOURCE_PEER) is True
    assert len(transport.sent) == 1


def test_hub_rejects_invalid_rtp() -> None:
    # Given: a registered stream with a malformed RTP payload type.
    transport = FakeDatagramTransport()
    hub = RtpHub(transport)
    hub.register_control(_source_registration(), SOURCE_PEER[0])
    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: the packet arrives from the registered Mic endpoint.
    forwarded = hub.route_datagram(_rtp_packet(payload_type=97), SOURCE_PEER)

    # Then: no UDP bytes are sent to the sink.
    assert forwarded is False
    assert transport.sent == []


def test_hub_rejects_rtp_from_unregistered_peer() -> None:
    # Given: a registered stream and a valid packet from the Sound peer instead of Mic.
    transport = FakeDatagramTransport()
    hub = RtpHub(transport)
    hub.register_control(_source_registration(), SOURCE_PEER[0])
    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: the valid packet arrives from an IP not registered as its source.
    forwarded = hub.route_datagram(_rtp_packet(payload_type=96), SINK_PEER)

    # Then: no UDP bytes are sent to the sink.
    assert forwarded is False
    assert transport.sent == []


def test_hub_registers_authenticated_control_envelopes_and_rejects_duplicates() -> None:
    # Given: a hub receiving the canonical source and sink envelopes from WSS peers.
    hub = RtpHub(FakeDatagramTransport())
    hub.register_control(_source_registration(), SOURCE_PEER[0])
    hub.register_control(_sink_registration(), SINK_PEER[0])

    # When: a second route claims either existing session-stream route.
    with pytest.raises(DuplicateRouteError):
        hub.register_control(_source_registration(), SOURCE_PEER[0])

    # Then: the duplicate is refused rather than replacing the authenticated route.
    with pytest.raises(DuplicateRouteError):
        hub.register_control(_sink_registration(), SINK_PEER[0])


def test_authenticated_control_registers_only_matching_bearer_token() -> None:
    # Given: a production bearer token protecting a new Mic source route.
    transport = FakeDatagramTransport()
    hub = RtpHub(transport)
    control = AuthenticatedControl(hub, TrustedLanToken("transport-test-token"))

    # When: Mic supplies its canonical envelope with the matching bearer value.
    accepted = control.register(
        _source_registration(),
        SOURCE_PEER[0],
        "Bearer transport-test-token",
    )
    rejected = control.register(
        _sink_registration(),
        SINK_PEER[0],
        "Bearer wrong-token",
    )

    # Then: only the authenticated route is retained and no unauthenticated sink exists.
    assert accepted is True
    assert rejected is False
    assert hub.route_datagram(_rtp_packet(payload_type=96), SOURCE_PEER) is False


def test_hub_removes_stream_routes_when_sound_cancels_stream() -> None:
    # Given: a pinned source route and registered sink route.
    transport = FakeDatagramTransport()
    hub = RtpHub(transport)
    hub.register_control(_source_registration(), SOURCE_PEER[0])
    hub.register_control(_sink_registration(), SINK_PEER[0])
    assert hub.route_datagram(_rtp_packet(payload_type=96), SOURCE_PEER) is True

    # When: Sound reports the canonical cancelled stream state.
    hub.register_control(_stream_state("cancelled"), SINK_PEER[0])

    # Then: the removed route cannot forward further RTP packets.
    assert hub.route_datagram(_rtp_packet(payload_type=96), SOURCE_PEER) is False
    assert len(transport.sent) == 1


def test_runtime_reports_ready_after_listeners_start_and_closes_them() -> None:
    # Given: injected control and datagram listeners for an explicit loopback runtime.
    datagram_transport = FakeDatagramTransport()
    control_server = FakeControlServer()
    runtime = TransportRuntime(
        _loopback_config(),
        datagram_listener=_fake_datagram_listener(datagram_transport),
        control_listener=_fake_control_listener(control_server),
    )

    # When: the runtime starts then receives its shutdown signal.
    asyncio.run(runtime.start())
    ready = runtime.readiness()
    asyncio.run(runtime.close())

    # Then: readiness requires both listeners and shutdown closes each resource.
    assert ready.ready is True
    assert control_server.closed is True
    assert control_server.waited is True
    assert datagram_transport.closed is True


def _source_registration() -> str:
    return _envelope(
        "media.rtp.source.register",
        "mic",
        {
            "stream_id": STREAM_ID,
            "ssrc": SSRC,
            "codec": _codec(),
            "rtp_endpoint": _endpoint(5004),
        },
    )


def _sink_registration() -> str:
    return _envelope(
        "media.rtp.sink.register",
        "sound",
        {"stream_id": STREAM_ID, "codec": _codec(), "rtp_endpoint": _endpoint(5006)},
    )


def _source_ready() -> str:
    return _envelope(
        "media.rtp.source.ready",
        "mic",
        {"stream_id": STREAM_ID, "ssrc": SSRC},
    )


def _sink_ready() -> str:
    return _envelope(
        "media.rtp.sink.ready",
        "sound",
        {"stream_id": STREAM_ID},
    )


def _stream_state(state: str) -> str:
    return _envelope(
        "media.stream.state",
        "sound",
        {"stream_id": STREAM_ID, "state": state},
    )


def _envelope(event_type: str, source: str, data: dict[str, JsonValue]) -> str:
    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": event_type,
            "event_id": f"evt-{event_type}",
            "source": source,
            "time": "2026-07-08T00:00:00Z",
            "trace_id": "trace-001",
            "session_id": SESSION_ID,
            "seq": 1,
            "data": data,
        }
    )


def _codec() -> dict[str, JsonValue]:
    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _endpoint(port: int) -> dict[str, JsonValue]:
    return {"host": "declared.example.test", "port": port}


def _rtp_packet(payload_type: int) -> bytes:
    header = bytes(
        (0x80, payload_type, 0, 1, 0, 0, 0, 1, 0x12, 0x34, 0x56, 0x78)
    )
    return header + (b"\x00\x01" * 320)


def _loopback_config() -> TransportConfig:
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


def _fake_datagram_listener(transport: FakeDatagramTransport) -> DatagramListener:
    async def listen(_host: str, _port: int, _hub: RtpHub) -> FakeDatagramTransport:
        return transport

    return listen


def _fake_control_listener(server: FakeControlServer) -> ControlListener:
    async def listen(
        _config: TransportConfig, _handler: ControlHandler
    ) -> FakeControlServer:
        return server

    return listen
