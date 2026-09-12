import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass, field
from typing import cast

import pytest

from orchestrator.json_boundary import parse_json_value
from orchestrator.mcp_allowlist import McpToolAllowance
from orchestrator.mcp_config import McpServer
from orchestrator.mcp_http import StreamableHttpMcpRequester


@dataclass
class _LoopbackServer:
    cancel: bool
    methods: list[str] = field(default_factory=list)
    call_started: asyncio.Event = field(default_factory=asyncio.Event)
    call_closed: asyncio.Event = field(default_factory=asyncio.Event)
    connections: set[asyncio.Task[None]] = field(default_factory=set)

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            header = (await reader.readuntil(b"\r\n\r\n")).decode()
            first, *lines = header.split("\r\n")
            length = next(
                (
                    int(line.split(":", 1)[1])
                    for line in lines
                    if line.lower().startswith("content-length:")
                ),
                0,
            )
            payload = await reader.readexactly(length)
            if first.startswith("DELETE"):
                self.methods.append("DELETE")
                writer.write(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return
            message = parse_json_value(payload.decode())
            assert isinstance(message, dict)
            method = message["method"]
            assert isinstance(method, str)
            self.methods.append(method)
            if method.startswith("notifications/"):
                writer.write(
                    b"\r\n".join(
                        (
                            b"HTTP/1.1 202 Accepted",
                            b"Content-Length: 0",
                            b"Connection: close",
                            b"",
                            b"",
                        )
                    )
                )
                await writer.drain()
                return
            if method == "tools/call" and self.cancel:
                writer.write(
                    b"\r\n".join(
                        (
                            b"HTTP/1.1 200 OK",
                            b"Content-Type: text/event-stream",
                            b"Connection: close",
                            b"",
                            b": waiting\n\n",
                        )
                    )
                )
                await writer.drain()
                self.call_started.set()
                assert await reader.read() == b""
                self.call_closed.set()
                return
            result: dict[str, object] = (
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "loopback", "version": "1"},
                }
                if method == "initialize"
                else {"content": [{"type": "text", "text": "真实 HTTP 查询成功"}]}
            )
            response = json.dumps(
                {"jsonrpc": "2.0", "id": message["id"], "result": result}
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(response)}\r\n".encode()
                + b"Mcp-Session-Id: loopback-session\r\nConnection: close\r\n\r\n"
                + response
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    def accepted(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections.add(asyncio.create_task(self.handle(reader, writer)))


@pytest.mark.parametrize("cancel", [False, True])
def test_real_http_mcp_roundtrip_and_cancellation(*, cancel: bool) -> None:
    async def scenario() -> None:
        fixture = _LoopbackServer(cancel)
        server = await asyncio.start_server(fixture.accepted, "127.0.0.1", 0)
        port = cast("tuple[str, int]", server.sockets[0].getsockname())[1]
        allowance = McpToolAllowance("local", "lookup", "catalog.read", 1000, 1024)
        requester = StreamableHttpMcpRequester(
            (McpServer("local", f"http://127.0.0.1:{port}/mcp"),)
        )
        task = asyncio.create_task(
            requester.request(allowance, {"query": "产品"}, timeout_ms=1000)
        )
        try:
            if cancel:
                _ = await asyncio.wait_for(fixture.call_started.wait(), timeout=2)
                _ = task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    _ = await task
                _ = await asyncio.wait_for(fixture.call_closed.wait(), timeout=2)
                assert "notifications/cancelled" in fixture.methods
            else:
                result = await task
                assert result is not None
                assert "真实 HTTP 查询成功" in json.dumps(result, ensure_ascii=False)
            assert fixture.methods[-1] == "DELETE"
            assert fixture.methods[:3] == [
                "initialize",
                "notifications/initialized",
                "tools/call",
            ]
        finally:
            _ = task.cancel()
            with suppress(asyncio.CancelledError):
                _ = await task
            server.close()
            await server.wait_closed()
            _ = await asyncio.gather(*fixture.connections)

    asyncio.run(scenario())
