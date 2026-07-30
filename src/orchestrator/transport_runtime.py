"""Authenticated WSS control listener and pinned UDP RTP forwarding runtime."""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from time import monotonic_ns
from typing import TYPE_CHECKING, Protocol, final, override

from websockets.asyncio.server import serve

from orchestrator.comment_ingress import (
    AuthenticatedCommentIngress,
    CommentAccessToken,
    CommentIngressConfig,
    CommentTokenValue,
)
from orchestrator.control_ingress import parse_session_control
from orchestrator.interaction_ingress import parse_comment_proposal
from orchestrator.json_boundary import JsonBoundaryError
from orchestrator.onsite_bridge import OnsiteExplainerBridge
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

    from orchestrator.observability import OnsiteObservability
    from orchestrator.scheduler_reflex import SchedulerOutputFence
    from orchestrator.scheduler_runtime import SessionRuntime
    from orchestrator.streaming_contracts import FlushClock, FlushFailure, StreamFlush

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

    listener_ready: bool
    route_ready: bool

    @property
    def ready(self) -> bool:
        """Retain the legacy aggregate listener readiness result."""
        return self.listener_ready


@final
class TransportRuntime:
    """Owns the WSS and UDP listener lifecycles for the Orchestrator transport."""

    def __init__(
        self,
        config: TransportConfig,
        datagram_listener: DatagramListener | None = None,
        control_listener: ControlListener | None = None,
        onsite_bridge: OnsiteBridge | None = None,
        clock: FlushClock | None = None,
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
        self._onsite_bridge: OnsiteBridge | None = onsite_bridge
        self._control_dispatch: TransportControlDispatch = TransportControlDispatch(
            self._hub, clock=clock
        )
        self._datagram_transport: DatagramSender | None = None
        self._control_server: ControlServer | None = None
        self._flush_driver: asyncio.Task[None] | None = None
        self._session_runtime: SessionRuntime | None = None
        self._comment_ingresses: dict[int, AuthenticatedCommentIngress] = {}

    def set_session_runtime(self, session_runtime: SessionRuntime) -> None:
        """Install the sole scheduler-owned authority for one control session."""
        self._session_runtime = session_runtime
        self.set_output_fence(session_runtime.output_fence)

    def set_observability(self, observability: OnsiteObservability) -> None:
        """Wire one shared recorder to the runtime's hub and control dispatch."""
        self._hub.set_observability(observability)
        self._control_dispatch.set_observability(observability)
        bridge = self._onsite_bridge
        if isinstance(bridge, OnsiteExplainerBridge):
            bridge.set_observability(observability)

    def set_output_fence(self, output_fence: SchedulerOutputFence) -> None:
        """Compose one scheduler fence into generated RTP and Sound acknowledgements."""
        self._hub.set_output_fence(output_fence)
        self._control_dispatch.set_output_fence(output_fence)

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
        self._flush_driver = asyncio.create_task(self._drive_flush_admission())

    async def cancel_stream(self, session_id: str, stream_id: str) -> None:
        """Cancel a live stream through the same WSS/UDP transport boundary."""
        await self._control_dispatch.cancel_stream(session_id, stream_id)

    async def request_stream_flush(self, flush: StreamFlush) -> None:
        """Initiate a Sound flush through the registered WSS control peer."""
        await self._control_dispatch.request_stream_flush(flush)

    async def advance_flush_admission(self) -> None:
        """Advance retry and timeout checks from the runtime-owned clock."""
        await self._control_dispatch.advance_flush_admission()

    async def admit_replacement(self, flush: StreamFlush) -> bool:
        """Create replacement control media only after matching Sound admission."""
        return await self._control_dispatch.admit_replacement(flush)

    @property
    def flush_failures(self) -> tuple[FlushFailure, ...]:
        """Return typed replacement-admission failures observed by this runtime."""
        return self._control_dispatch.flush_failures

    def readiness(self) -> TransportReadiness:
        """Return separate listener and Mic-to-Sound route readiness facts."""
        listener_ready = (
            self._datagram_transport is not None
            and self._control_server is not None
        )
        return TransportReadiness(listener_ready, self._hub.route_ready)

    async def close(self) -> None:
        """Close WSS and UDP resources in deterministic reverse startup order."""
        flush_driver = self._flush_driver
        if flush_driver is not None:
            _ = flush_driver.cancel()
            with suppress(asyncio.CancelledError):
                await flush_driver
            self._flush_driver = None
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
                    if not bearer_token_matches(
                        self._config.control_token,
                        _connection_authorization(connection),
                    ):
                        continue
                    session_runtime = self._session_runtime
                    if session_runtime is not None and parse_comment_proposal(message):
                        self._receive_comment(connection, session_runtime, message)
                        continue
                    if (
                        session_runtime is not None
                        and session_runtime.receive_control(message)
                    ):
                        continue
                    if session_runtime is not None:
                        control = parse_session_control(message)
                        if control is not None:
                            _ = await session_runtime.receive_session_control_async(
                                control
                            )
                            continue
                    try:
                        await self._control_dispatch.register(
                            message,
                            peer_ip,
                            connection,
                        )
                    except (ControlEnvelopeError, JsonBoundaryError):
                        continue
        finally:
            comment_ingress = self._comment_ingresses.pop(id(connection), None)
            if comment_ingress is not None:
                comment_ingress.cancel_pending()
            self._control_dispatch.remove_connection(connection)

    def _receive_comment(
        self,
        connection: ControlConnection,
        session_runtime: SessionRuntime,
        message: str,
    ) -> None:
        ingress = self._comment_ingresses.setdefault(
            id(connection),
            AuthenticatedCommentIngress(
                session_runtime.interaction_ingress,
                _comment_ingress_config(self._config),
            ),
        )
        receipt = ingress.receive(
            message,
            _connection_authorization(connection),
            now_ms=monotonic_ns() // 1_000_000,
        )
        if not receipt.accepted:
            return
        while (proposal := ingress.take_next()) is not None:
            _ = session_runtime.receive_comment(proposal)

    def route_datagram(self, data: bytes, peer: tuple[str, int]) -> bool:
        """Route one UDP datagram through the authoritative connection-owned hub."""
        return self._hub.route_datagram(data, peer)

    async def wait_for_onsite_jobs(self) -> None:
        """Wait for accepted onsite provider work to finish or be cancelled."""
        await self._hub.wait_for_onsite_jobs()

    async def _drive_flush_admission(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            await self._control_dispatch.advance_flush_admission()


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


def _comment_ingress_config(config: TransportConfig) -> CommentIngressConfig:
    token = config.control_token
    credential = (
        None
        if token is None
        else CommentAccessToken(CommentTokenValue(token), (1 << 63) - 1)
    )
    return CommentIngressConfig(
        token=credential,
        replay_window=128,
        max_payload_bytes=16_384,
        max_pending=16,
    )


def _connection_authorization(connection: ControlConnection) -> str | None:
    authorization = getattr(connection, "authorization", None)
    if isinstance(authorization, str):
        return authorization
    request = getattr(connection, "request", None)
    headers = getattr(request, "headers", None)
    candidate = getattr(headers, "get", None)
    value = candidate("Authorization") if callable(candidate) else None
    return value if isinstance(value, str) else None


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
