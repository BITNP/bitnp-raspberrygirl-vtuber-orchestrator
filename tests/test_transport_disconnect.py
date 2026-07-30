"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 _FakeDatagramTransport
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: sent。 方法: sendto、close。
    """

    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 data: bytes。
        必填。 addr: tuple[str, int]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.sent.append((data, addr))

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return


@dataclass
class _FakeControlServer:
    """类契约说明.

    职责: 保存 _FakeControlServer
    不可变数据结构,用类型标注表达字段契约。
    契约: 方法: close、wait_closed。
    """

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return

    async def wait_closed(self) -> None:
        """函数契约说明.

        功能: 执行 wait_closed
        的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """

        return


@dataclass
class _LiveControlConnection:
    """类契约说明.

    职责: 保存 _LiveControlConnection
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: message、peer_ip、opened、close
    d、sent、delivered。 方法: remote_address
    、__aiter__、__anext__、send、respond。
    """

    message: str

    peer_ip: str

    opened: asyncio.Event = field(default_factory=asyncio.Event)

    closed: asyncio.Event = field(default_factory=asyncio.Event)

    sent: list[str] = field(default_factory=list)

    delivered: bool = False

    @property
    def remote_address(self) -> tuple[str, int]:
        """函数契约说明.

        功能: 执行 remote_address
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[str, int]`。
        """

        return (self.peer_ip, 443)

    def __aiter__(self) -> AsyncIterator[str]:
        """函数契约说明.

        功能: 执行 __aiter__ 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `AsyncIterator[str]`。
        """

        return self

    async def __anext__(self) -> str:
        """函数契约说明.

        功能: 执行 __anext__ 的异步逻辑,并协调 set,
        wait。
        参数: self 表示当前实例。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `str`。
        """

        if not self.delivered:
            self.delivered = True

            self.opened.set()

            return self.message

        _ = await self.closed.wait()

        raise StopAsyncIteration

    async def send(self, message: str) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 message: str。
        必填。
        契约: 异步调用。 返回 `None`。
        """

        self.sent.append(message)

    def respond(self, status: HTTPStatus, text: str) -> Response:
        """函数契约说明.

        功能: 执行 respond 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 status:
        HTTPStatus。 必填。 text: str。 必填。
        契约: 同步调用。 返回 `Response`。
        """

        _ = status

        _ = text

        raise AssertionError


def test_runtime_drops_pinned_rtp_after_owning_wss_connection_closes() -> None:
    """函数契约说明.

    功能: 验证 runtime drops pinned rtp
    after owning wss connection closes
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_disconnect_route_proof())


async def _disconnect_route_proof() -> None:
    # Given: live Mic and Sound WSS sessions sharing one IP but owning distinct routes.

    """函数契约说明.

    功能: 执行 _disconnect_route_proof
    的异步逻辑,并协调 _FakeDatagramTransport,
    TransportRuntime,
    _LiveControlConnection, create_task。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

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

    # When: the source route forwards once, then only its WSS connection closes.

    assert runtime.route_datagram(packet, endpoint) is True

    source.closed.set()

    await source_task

    # Then: the identical valid packet from the exact pinned endpoint is dropped.

    assert runtime.route_datagram(packet, endpoint) is False

    assert runtime.route_datagram(retained_packet, endpoint) is True

    assert transport.sent == [
        (packet, ("127.0.0.1", 5006)),
        (retained_packet, ("127.0.0.1", 5007)),
    ]

    sink.closed.set()

    retained_source.closed.set()

    retained_sink.closed.set()

    await sink_task

    await retained_source_task

    await retained_sink_task

    await runtime.close()


def _source_registration(stream_id: str, ssrc: int) -> str:
    """函数契约说明.

    功能: 执行 _source_registration
    的同步逻辑,并协调 _envelope, _codec。
    参数: stream_id: str。 必填。 ssrc: int。
    必填。
    契约: 同步调用。 返回 `str`。
    """

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
    """函数契约说明.

    功能: 执行 _sink_registration 的同步逻辑,并协调
    _envelope, _codec。
    参数: stream_id: str。 必填。 udp_port:
    int。 必填。
    契约: 同步调用。 返回 `str`。
    """

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
    """函数契约说明.

    功能: 执行 _envelope 的同步逻辑,并协调 dumps。
    参数: event_type: str。 必填。 source:
    str。 必填。 data: dict[str, JsonValue]。
    必填。
    契约: 同步调用。 返回 `str`。
    """

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
    """函数契约说明.

    功能: 执行 _codec 的同步逻辑,并维持签名契约。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """

    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _rtp_packet(ssrc: int = 0x12345678) -> bytes:
    """函数契约说明.

    功能: 执行 _rtp_packet 的同步逻辑,并协调
    to_bytes。
    参数: ssrc: int。 可省略。
    契约: 同步调用。 返回 `bytes`。
    """

    header = b"\x80\x60\x00\x01\x00\x00\x00\x01" + ssrc.to_bytes(4, "big")

    return header + (b"\x00\x01" * 320)


def _loopback_config() -> TransportConfig:
    """函数契约说明.

    功能: 执行 _loopback_config 的同步逻辑,并协调
    TransportConfig。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `TransportConfig`。
    """

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
    """函数契约说明.

    功能: 执行 _datagram_listener
    的同步逻辑,并维持签名契约。
    参数: transport:
    _FakeDatagramTransport。 必填。
    契约: 同步调用。 返回 `DatagramListener`。
    """

    async def listen(_host: str, _port: int, _hub: RtpHub) -> _FakeDatagramTransport:
        """函数契约说明.

        功能: 执行 listen 的异步逻辑,并维持签名契约。
        参数: _host: str。 必填。 _port: int。
        必填。 _hub: RtpHub。 必填。
        契约: 异步调用。 返回
        `_FakeDatagramTransport`。
        """

        return transport

    return listen


async def _control_listener(
    _config: TransportConfig, _handler: ControlHandler
) -> _FakeControlServer:
    """函数契约说明.

    功能: 执行 _control_listener 的异步逻辑,并协调
    _FakeControlServer。
    参数: _config: TransportConfig。 必填。
    _handler: ControlHandler。 必填。
    契约: 异步调用。 返回 `_FakeControlServer`。
    """

    return _FakeControlServer()
