"""模块契约说明.

职责: 提供 orchestrator.transport_runtime
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 声明 ControlServer
    协议接口,约束实现方必须提供的行为。
    契约: 方法: close、wait_closed。
    """

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

    async def wait_closed(self) -> None:
        """函数契约说明.

        功能: 执行 wait_closed
        的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """


class ControlConnection(Protocol):
    """类契约说明.

    职责: 声明 ControlConnection
    协议接口,约束实现方必须提供的行为。
    契约: 方法: remote_address、__aiter__、res
    pond、send。
    """

    @property
    def remote_address(self) -> tuple[str, int] | None:
        """函数契约说明.

        功能: 执行 remote_address
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[str, int] |
        None`。
        """

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        """函数契约说明.

        功能: 执行 __aiter__ 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `AsyncIterator[str
        | bytes]`。
        """
        ...

    def respond(self, status: HTTPStatus, text: str) -> Response:
        """函数契约说明.

        功能: 执行 respond 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 status:
        HTTPStatus。 必填。 text: str。 必填。
        契约: 同步调用。 返回 `Response`。
        """
        ...

    async def send(self, message: str) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 message: str。
        必填。
        契约: 异步调用。 返回 `None`。
        """


type ControlHandler = Callable[[ControlConnection], Awaitable[None]]

type ControlListener = Callable[
    [TransportConfig, ControlHandler], Awaitable[ControlServer]
]


@dataclass(frozen=True, slots=True)
class TransportReadiness:
    """类契约说明.

    职责: 保存 TransportReadiness
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: listener_ready、route_ready。
    方法: ready。
    """

    listener_ready: bool

    route_ready: bool

    @property
    def ready(self) -> bool:
        """函数契约说明.

        功能: 执行 ready 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `bool`。
        """
        return self.listener_ready


@final
class TransportRuntime:
    """类契约说明.

    职责: 定义 TransportRuntime
    的状态、行为和对外协作边界。
    契约: 方法: __init__、set_session_runtime
    、set_observability、set_output_fence、
    start、cancel_stream。
    """

    def __init__(
        self,
        config: TransportConfig,
        datagram_listener: DatagramListener | None = None,
        control_listener: ControlListener | None = None,
        onsite_bridge: OnsiteBridge | None = None,
        clock: FlushClock | None = None,
    ) -> None:
        """函数契约说明.

        功能: 初始化 TransportRuntime
        的字段并建立实例不变式。
        参数: self 表示当前实例。 config:
        TransportConfig。 必填。
        datagram_listener:
        DatagramListener | None。 可省略。
        control_listener:
        ControlListener | None。 可省略。
        onsite_bridge: OnsiteBridge |
        None。 可省略。 clock: FlushClock |
        None。 可省略。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 set_session_runtime
        的同步逻辑,并协调 set_output_fence。
        参数: self 表示当前实例。
        session_runtime: SessionRuntime。
        必填。
        契约: 同步调用。 返回 `None`。
        """
        self._session_runtime = session_runtime

        self.set_output_fence(session_runtime.output_fence)

    def set_observability(self, observability: OnsiteObservability) -> None:
        """函数契约说明.

        功能: 执行 set_observability
        的同步逻辑,并协调 set_observability,
        isinstance。
        参数: self 表示当前实例。 observability:
        OnsiteObservability。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._hub.set_observability(observability)

        self._control_dispatch.set_observability(observability)

        bridge = self._onsite_bridge

        if isinstance(bridge, OnsiteExplainerBridge):
            bridge.set_observability(observability)

    def set_output_fence(self, output_fence: SchedulerOutputFence) -> None:
        """函数契约说明.

        功能: 执行 set_output_fence
        的同步逻辑,并协调 set_output_fence。
        参数: self 表示当前实例。 output_fence:
        SchedulerOutputFence。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._hub.set_output_fence(output_fence)

        self._control_dispatch.set_output_fence(output_fence)

    async def start(self) -> None:
        """函数契约说明.

        功能: 执行 start 的异步逻辑,并协调
        attach_transport, create_task,
        _datagram_listener,
        _control_listener。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
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
        """函数契约说明.

        功能: 执行 cancel_stream 的异步逻辑,并协调
        cancel_stream。
        参数: self 表示当前实例。 session_id:
        str。 必填。 stream_id: str。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        await self._control_dispatch.cancel_stream(session_id, stream_id)

    async def request_stream_flush(self, flush: StreamFlush) -> None:
        """函数契约说明.

        功能: 执行 request_stream_flush
        的异步逻辑,并协调 request_stream_flush。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        await self._control_dispatch.request_stream_flush(flush)

    async def advance_flush_admission(self) -> None:
        """函数契约说明.

        功能: 执行 advance_flush_admission
        的异步逻辑,并协调
        advance_flush_admission。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        await self._control_dispatch.advance_flush_admission()

    async def admit_replacement(self, flush: StreamFlush) -> bool:
        """函数契约说明.

        功能: 执行 admit_replacement
        的异步逻辑,并协调 admit_replacement。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `bool`。
        """
        return await self._control_dispatch.admit_replacement(flush)

    @property
    def flush_failures(self) -> tuple[FlushFailure, ...]:
        """函数契约说明.

        功能: 执行 flush_failures
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `tuple[FlushFailure, ...]`。
        """
        return self._control_dispatch.flush_failures

    def readiness(self) -> TransportReadiness:
        """函数契约说明.

        功能: 执行 readiness 的同步逻辑,并协调
        TransportReadiness。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `TransportReadiness`。
        """
        listener_ready = (
            self._datagram_transport is not None and self._control_server is not None
        )

        return TransportReadiness(listener_ready, self._hub.route_ready)

    async def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的异步逻辑,并协调 clear,
        cancel, close,
        wait_for_onsite_jobs。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
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
        """函数契约说明.

        功能: 处理输入事件、请求或状态转换。
        参数: self 表示当前实例。 connection:
        ControlConnection。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
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

                    if session_runtime is not None and session_runtime.receive_control(
                        message
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
        """函数契约说明.

        功能: 执行 _receive_comment
        的同步逻辑,并协调 setdefault, receive,
        id, AuthenticatedCommentIngress。
        参数: self 表示当前实例。 connection:
        ControlConnection。 必填。
        session_runtime: SessionRuntime。
        必填。 message: str。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 route_datagram 的同步逻辑,并协调
        route_datagram。
        参数: self 表示当前实例。 data: bytes。
        必填。 peer: tuple[str, int]。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        return self._hub.route_datagram(data, peer)

    async def wait_for_onsite_jobs(self) -> None:
        """函数契约说明.

        功能: 执行 wait_for_onsite_jobs
        的异步逻辑,并协调 wait_for_onsite_jobs。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        await self._hub.wait_for_onsite_jobs()

    async def _drive_flush_admission(self) -> None:
        """函数契约说明.

        功能: 执行 _drive_flush_admission
        的异步逻辑,并协调 sleep,
        advance_flush_admission。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `None`。
        """
        while True:
            await asyncio.sleep(0.25)

            await self._control_dispatch.advance_flush_admission()


@final
class _RtpDatagramProtocol(asyncio.DatagramProtocol):
    """类契约说明.

    职责: 声明 _RtpDatagramProtocol
    协议接口,约束实现方必须提供的行为。
    契约: 方法: __init__、datagram_received。
    """

    def __init__(self, hub: RtpHub) -> None:
        """函数契约说明.

        功能: 初始化 _RtpDatagramProtocol
        的字段并建立实例不变式。
        参数: self 表示当前实例。 hub: RtpHub。
        必填。
        契约: 同步调用。 返回 `None`。
        """
        self._hub: RtpHub = hub

    @override
    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """函数契约说明.

        功能: 执行 datagram_received
        的同步逻辑,并协调 route_datagram。
        参数: self 表示当前实例。 data: bytes。
        必填。 addr: tuple[str, int]。 必填。
        契约: 同步调用。 返回 `None`。
        """
        _ = self._hub.route_datagram(data, addr)


async def _listen_udp(host: str, port: int, hub: RtpHub) -> DatagramSender:
    """函数契约说明.

    功能: 执行 _listen_udp 的异步逻辑,并协调
    get_running_loop,
    create_datagram_endpoint,
    _RtpDatagramProtocol。
    参数: host: str。 必填。 port: int。 必填。
    hub: RtpHub。 必填。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回
    `DatagramSender`。
    """
    loop = asyncio.get_running_loop()

    transport, _ = await loop.create_datagram_endpoint(
        lambda: _RtpDatagramProtocol(hub),
        local_addr=(host, port),
    )

    return transport


async def _listen_control(
    config: TransportConfig, handler: ControlHandler
) -> ControlServer:
    """函数契约说明.

    功能: 执行 _listen_control 的异步逻辑,并协调
    _ssl_context, get,
    bearer_token_matches, respond。
    参数: config: TransportConfig。 必填。
    handler: ControlHandler。 必填。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回
    `ControlServer`。
    """
    ssl_context = _ssl_context(config)

    def authorize(connection: ControlConnection, request: Request) -> Response | None:
        """函数契约说明.

        功能: 执行 authorize 的同步逻辑,并协调 get,
        bearer_token_matches, respond。
        参数: connection:
        ControlConnection。 必填。 request:
        Request。 必填。
        契约: 同步调用。 返回 `Response | None`。
        """
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
    """函数契约说明.

    功能: 执行 _comment_ingress_config
    的同步逻辑,并协调 CommentIngressConfig,
    CommentAccessToken,
    CommentTokenValue。
    参数: config: TransportConfig。 必填。
    契约: 同步调用。 返回 `CommentIngressConfig`。
    """
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
    """函数契约说明.

    功能: 执行 _connection_authorization
    的同步逻辑,并协调 getattr, isinstance,
    callable, candidate。
    参数: connection: ControlConnection。
    必填。
    契约: 同步调用。 返回 `str | None`。
    """
    authorization = getattr(connection, "authorization", None)

    if isinstance(authorization, str):
        return authorization

    request = getattr(connection, "request", None)

    headers = getattr(request, "headers", None)

    candidate = getattr(headers, "get", None)

    value = candidate("Authorization") if callable(candidate) else None

    return value if isinstance(value, str) else None


def _ssl_context(config: TransportConfig) -> ssl.SSLContext | None:
    """函数契约说明.

    功能: 执行 _ssl_context 的同步逻辑,并协调
    SSLContext, load_cert_chain,
    ControlEnvelopeError。
    参数: config: TransportConfig。 必填。
    契约: 同步调用。 返回 `ssl.SSLContext |
    None`。 可能抛出 ControlEnvelopeError。
    """
    if config.control_scheme == "ws":
        return None

    if config.tls_cert_path is None or config.tls_key_path is None:
        raise ControlEnvelopeError(field_name="tls")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    context.load_cert_chain(config.tls_cert_path, config.tls_key_path)

    return context


def _peer_ip(connection: ControlConnection) -> str:
    """函数契约说明.

    功能: 执行 _peer_ip 的同步逻辑,并协调
    ControlEnvelopeError。
    参数: connection: ControlConnection。
    必填。
    契约: 同步调用。 返回 `str`。 可能抛出
    ControlEnvelopeError。
    """
    remote_address = connection.remote_address

    if remote_address is None:
        raise ControlEnvelopeError(field_name="peer")

    return remote_address[0]
