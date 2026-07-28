"""Authenticated WSS control listener and pinned UDP RTP forwarding runtime."""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Protocol, final, override

from websockets.asyncio.server import serve

from orchestrator.transport_config import TransportConfig
from orchestrator.transport_control import ControlEnvelopeError, bearer_token_matches
from orchestrator.transport_dispatch import TransportControlDispatch
from orchestrator.transport_hub import (
    DatagramSender,
    OnsiteBridge,
    RtpHub,
)

if TYPE_CHECKING:
    from websockets.http11 import Request, Response

type DatagramListener = Callable[[str, int, RtpHub], Awaitable[DatagramSender]]


class ControlServer(Protocol):
    """Minimal close lifecycle supplied by a websockets listener."""

    def close(self) -> None:
        """Stop accepting new WSS control sessions."""

    async def wait_closed(self) -> None:
        """Wait until the control listener has released its resources."""


class ControlConnection(Protocol):
    """Typed subset of a live WSS connection required by the control listener."""

    @property
    def remote_address(self) -> tuple[str, int] | None:
        """Return the TCP peer that authenticated this control connection."""

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        """Yield WSS control messages until the peer disconnects."""
        ...

    def respond(self, status: HTTPStatus, text: str) -> Response:
        """Abort a handshake with one typed HTTP response."""
        ...

    async def send(self, message: str) -> None:
        """Send one canonical text control envelope."""


type ControlHandler = Callable[[ControlConnection], Awaitable[None]]
type ControlListener = Callable[
    [TransportConfig, ControlHandler], Awaitable[ControlServer]
]


@dataclass(frozen=True, slots=True)
class TransportReadiness:
    """Readiness status for the two listener resources owned by the runtime."""

    ready: bool


@final
class TransportRuntime:
    """Owns the WSS and UDP listener lifecycles for the Orchestrator transport."""

    def __init__(
        self,
        config: TransportConfig,
        datagram_listener: DatagramListener | None = None,
        control_listener: ControlListener | None = None,
        onsite_bridge: OnsiteBridge | None = None,
    ) -> None:
        """Create one transport runtime with optional fake listener factories."""
        self._config: TransportConfig = config
        self._datagram_listener: DatagramListener = (
            _listen_udp if datagram_listener is None else datagram_listener
        )
        self._control_listener: ControlListener = (
            _listen_control if control_listener is None else control_listener
        )
        self._hub: RtpHub = RtpHub(onsite_bridge=onsite_bridge)
        self._control_dispatch: TransportControlDispatch = TransportControlDispatch(
            self._hub
        )
        self._datagram_transport: DatagramSender | None = None
        self._control_server: ControlServer | None = None

    async def start(self) -> None:
        """Start UDP then authenticated WSS listeners and publish ready state."""
        self._datagram_transport = await self._datagram_listener(
            self._config.udp_bind_host,
            self._config.udp_bind_port,
            self._hub,
        )
        self._hub.attach_transport(self._datagram_transport)
        self._control_server = await self._control_listener(
            self._config, self.handle_control
        )

    async def cancel_stream(self, session_id: str, stream_id: str) -> None:
        """Cancel a live stream through the same WSS/UDP transport boundary."""
        await self._control_dispatch.cancel_stream(session_id, stream_id)

    def readiness(self) -> TransportReadiness:
        """Return ready only while both listener resources are active."""
        ready = (
            self._datagram_transport is not None
            and self._control_server is not None
        )
        return TransportReadiness(ready)

    async def close(self) -> None:
        """Close WSS and UDP resources in deterministic reverse startup order."""
        if self._control_server is not None:
            self._control_server.close()
            await self._control_server.wait_closed()
            self._control_server = None
        if self._datagram_transport is not None:
            self._datagram_transport.close()
            self._datagram_transport = None
        self._hub.clear()
        await self._hub.wait_for_onsite_jobs()
        self._control_dispatch.clear()

    async def handle_control(self, connection: ControlConnection) -> None:
        """Apply WSS control messages until this exact connection disconnects."""
        peer_ip = _peer_ip(connection)
        try:
            async for message in connection:
                if isinstance(message, str):
                    await self._control_dispatch.register(message, peer_ip, connection)
        finally:
            self._control_dispatch.remove_connection(connection)

    def route_datagram(self, data: bytes, peer: tuple[str, int]) -> bool:
        """Route one UDP datagram through the authoritative connection-owned hub."""
        return self._hub.route_datagram(data, peer)

    async def wait_for_onsite_jobs(self) -> None:
        """Wait for accepted onsite provider work to finish or be cancelled."""
        await self._hub.wait_for_onsite_jobs()


@final
class _RtpDatagramProtocol(asyncio.DatagramProtocol):
    """Bridges asyncio datagrams into the synchronous RTP route matcher."""

    def __init__(self, hub: RtpHub) -> None:
        self._hub: RtpHub = hub

    @override
    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Route a received datagram without copying or decoding its payload."""
        _ = self._hub.route_datagram(data, addr)


async def _listen_udp(host: str, port: int, hub: RtpHub) -> DatagramSender:
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _RtpDatagramProtocol(hub),
        local_addr=(host, port),
    )
    return transport


async def _listen_control(
    config: TransportConfig, handler: ControlHandler
) -> ControlServer:
    ssl_context = _ssl_context(config)

    def authorize(connection: ControlConnection, request: Request) -> Response | None:
        authorization = request.headers.get("Authorization")
        if bearer_token_matches(config.control_token, authorization):
            return None
        return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")

    return await serve(
        handler,
        config.control_bind_host,
        config.control_bind_port,
        process_request=authorize,
        ssl=ssl_context,
    )


def _ssl_context(config: TransportConfig) -> ssl.SSLContext | None:
    if config.control_scheme == "ws":
        return None
    if config.tls_cert_path is None or config.tls_key_path is None:
        raise ControlEnvelopeError(field_name="tls")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(config.tls_cert_path, config.tls_key_path)
    return context


def _peer_ip(connection: ControlConnection) -> str:
    remote_address = connection.remote_address
    if remote_address is None:
        raise ControlEnvelopeError(field_name="peer")
    return remote_address[0]
