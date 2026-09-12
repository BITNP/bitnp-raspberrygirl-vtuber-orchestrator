import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import override

import httpx
import pytest

from orchestrator.json_boundary import parse_json_value
from orchestrator.mcp_allowlist import McpToolAllowance
from orchestrator.mcp_config import McpServer
from orchestrator.mcp_http import McpProtocolError, StreamableHttpMcpRequester

_ALLOWANCE = McpToolAllowance("catalog", "lookup", "knowledge.external", 1000, 1024)


@dataclass
class _McpServer:
    calls: list[httpx.Request] = field(default_factory=list)
    sse: bool = False
    result_id: int = 2
    result: dict[str, object] = field(
        default_factory=lambda: {"content": [{"type": "text", "text": "展会周六举行"}]}
    )
    stream: httpx.AsyncByteStream | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        payload = parse_json_value(request.content.decode())
        assert isinstance(payload, dict)
        method = payload["method"]
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "isolated-session"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test", "version": "1"},
                    },
                },
            )
        assert request.headers["Mcp-Session-Id"] == "isolated-session"
        assert request.headers["MCP-Protocol-Version"] == "2025-06-18"
        if method != "tools/call":
            return httpx.Response(202)
        if self.stream is not None:
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, stream=self.stream
            )
        body = {"jsonrpc": "2.0", "id": self.result_id, "result": self.result}
        if self.sse:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=f": ping\r\n\r\ndata: {json.dumps(body)}\r\n\r\n".encode(),
            )
        return httpx.Response(200, json=body)


@pytest.mark.parametrize("sse", [False, True])
def test_mcp_initializes_calls_and_closes_isolated_session(*, sse: bool) -> None:
    server = _McpServer(sse=sse)
    requester = StreamableHttpMcpRequester(
        (McpServer("catalog", "https://mcp.example.test/mcp", "test-secret"),),
        transport=httpx.MockTransport(server.handle),
    )
    result = asyncio.run(
        requester.request(_ALLOWANCE, {"query": "展会时间"}, timeout_ms=1000)
    )
    assert result == server.result
    assert len(server.calls) == 4
    assert server.calls[-1].method == "DELETE"
    assert server.calls[0].headers["Authorization"] == "Bearer test-secret"
    assert b"tools/list" not in b"".join(request.content for request in server.calls)


def test_mcp_rejects_wrong_response_id() -> None:
    server = _McpServer(result_id=99)
    requester = StreamableHttpMcpRequester(
        (McpServer("catalog", "https://mcp.example.test/mcp"),),
        transport=httpx.MockTransport(server.handle),
    )
    with pytest.raises(McpProtocolError, match="correlation"):
        _ = asyncio.run(requester.request(_ALLOWANCE, {}, timeout_ms=1000))
    assert server.calls[-1].method == "DELETE"


def test_mcp_bounds_http_body_before_parsing() -> None:
    server = _McpServer(result={"content": [{"type": "text", "text": "x" * 20_000}]})
    requester = StreamableHttpMcpRequester(
        (McpServer("catalog", "https://mcp.example.test/mcp"),),
        transport=httpx.MockTransport(server.handle),
    )
    with pytest.raises(McpProtocolError, match="size"):
        _ = asyncio.run(requester.request(_ALLOWANCE, {}, timeout_ms=1000))


class _BlockedStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started: asyncio.Event = asyncio.Event()
        self.closed: bool = False

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await asyncio.Future[None]()
        yield b""

    @override
    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_and_timeout_close_stream_and_notify_server(*, cancel: bool) -> None:
    async def scenario() -> None:
        stream = _BlockedStream()
        server = _McpServer(stream=stream)
        requester = StreamableHttpMcpRequester(
            (McpServer("catalog", "https://mcp.example.test/mcp"),),
            transport=httpx.MockTransport(server.handle),
        )
        task = asyncio.create_task(
            requester.request(_ALLOWANCE, {}, timeout_ms=100 if not cancel else 1000)
        )
        _ = await asyncio.wait_for(stream.started.wait(), timeout=1)
        if cancel:
            _ = task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            _ = await task
        assert stream.closed
        assert any(
            b"notifications/cancelled" in request.content for request in server.calls
        )
        assert server.calls[-1].method == "DELETE"

    asyncio.run(scenario())
