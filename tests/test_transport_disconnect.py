
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.transport_config import TransportConfig
from orchestrator.transport_runtime import ControlHandler, TransportRuntime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from http import HTTPStatus

    from websockets.http11 import Response

    from orchestrator.json_boundary import JsonValue
    from orchestrator.transport_hub import RtpHub
    from orchestrator.transport_runtime import DatagramListener


@dataclass
class _FakeDatagramTransport:

    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:

        self.sent.append((data, addr))

    def close(self) -> None:

        return


@dataclass
class _FakeControlServer:

    def close(self) -> None:

        return

    async def wait_closed(self) -> None:

        return


@dataclass
class _LiveControlConnection:

    message: str

    peer_ip: str

    opened: asyncio.Event = field(default_factory=asyncio.Event)

    closed: asyncio.Event = field(default_factory=asyncio.Event)

    sent: list[str] = field(default_factory=list)

    delivered: bool = False

    @property
    def remote_address(self) -> tuple[str, int]:

        return (self.peer_ip, 443)

    def __aiter__(self) -> AsyncIterator[str]:

        return self

    async def __anext__(self) -> str:

        if not self.delivered:
            self.delivered = True

            self.opened.set()

            return self.message

        _ = await self.closed.wait()

        raise StopAsyncIteration

    async def send(self, message: str) -> None:

        self.sent.append(message)

    def respond(self, status: HTTPStatus, text: str) -> Response:

        _ = status

        _ = text

        raise AssertionError


def test_runtime_drops_pinned_rtp_after_owning_wss_connection_closes() -> None:

    asyncio.run(_disconnect_route_proof())


async def _disconnect_route_proof() -> None:
    # Given: live Mic and Sound WSS sessions sharing one IP but owning distinct routes.


    transport = _FakeDatagramTransport()

    runtime = TransportRuntime(
        _loopback_config(),
        datagram_listener=_datagram_listener(transport),
        control_listener=_control_listener,
    )

    await runtime.start()

    source = _LiveControlConnection(
        _source_registration("stream-001", 0x12345678), "127.0.0.1"
    )

    sink = _LiveControlConnection(_sink_registration("stream-001", 5006), "127.0.0.1")

    retained_source = _LiveControlConnection(
        _source_registration("stream-002", 0x12345679), "127.0.0.1"
    )

    retained_sink = _LiveControlConnection(
        _sink_registration("stream-002", 5007), "127.0.0.1"
    )

    source_task = asyncio.create_task(runtime.handle_control(source))

    sink_task = asyncio.create_task(runtime.handle_control(sink))

    retained_source_task = asyncio.create_task(runtime.handle_control(retained_source))

    retained_sink_task = asyncio.create_task(runtime.handle_control(retained_sink))

    _ = await source.opened.wait()

    _ = await sink.opened.wait()

    _ = await retained_source.opened.wait()

    _ = await retained_sink.opened.wait()

    packet = _rtp_packet()

    retained_packet = _rtp_packet(0x12345679)

    endpoint = ("127.0.0.1", 41_000)

    # When: the source route receives RTP, then only its WSS connection closes.

    assert runtime.route_datagram(packet, endpoint) is False

    source.closed.set()

    await source_task

    # Then: the identical valid packet from the exact pinned endpoint is dropped.

    assert runtime.route_datagram(packet, endpoint) is False

    assert runtime.route_datagram(retained_packet, endpoint) is False

    assert transport.sent == []

    sink.closed.set()

    retained_source.closed.set()

    retained_sink.closed.set()

    await sink_task

    await retained_source_task

    await retained_sink_task

    await runtime.close()


def _source_registration(stream_id: str, ssrc: int) -> str:

    return _envelope(
        "media.rtp.source.register",
        "mic",
        {
            "stream_id": stream_id,
            "ssrc": ssrc,
            "codec": _codec(),
            "rtp_endpoint": {"host": "127.0.0.1", "port": 5004},
        },
    )


def _sink_registration(stream_id: str, udp_port: int) -> str:

    return _envelope(
        "media.rtp.sink.register",
        "sound",
        {
            "stream_id": stream_id,
            "codec": _codec(),
            "rtp_endpoint": {"host": "127.0.0.1", "port": udp_port},
        },
    )


def _envelope(event_type: str, source: str, data: dict[str, JsonValue]) -> str:

    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": event_type,
            "event_id": event_type,
            "source": source,
            "time": "2026-07-27T00:00:00Z",
            "trace_id": "trace-001",
            "session_id": "session-001",
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


def _rtp_packet(ssrc: int = 0x12345678) -> bytes:

    header = b"\x80\x60\x00\x01\x00\x00\x00\x01" + ssrc.to_bytes(4, "big")

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


def _datagram_listener(transport: _FakeDatagramTransport) -> DatagramListener:

    async def listen(_host: str, _port: int, _hub: RtpHub) -> _FakeDatagramTransport:

        return transport

    return listen


async def _control_listener(
    _config: TransportConfig, _handler: ControlHandler
) -> _FakeControlServer:

    return _FakeControlServer()
