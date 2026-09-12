"""Cancellable, bounded MCP Streamable HTTP with an isolated session per call."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from http import HTTPStatus
from typing import TYPE_CHECKING, cast, final

import httpx

from orchestrator.json_boundary import parse_json_value
from orchestrator.tls import build_tls_context

if TYPE_CHECKING:
    from orchestrator.mcp_allowlist import McpToolAllowance
    from orchestrator.mcp_config import McpServer

_PROTOCOL_VERSION = "2025-06-18"
_SUPPORTED_VERSIONS = {"2025-03-26", _PROTOCOL_VERSION, "2025-11-25"}
_CONTROL_BYTES = 16_384
_CLEANUP_SECONDS = 0.25
_SESSION_ID = re.compile(r"[!-~]{1,256}")


class McpProtocolError(ValueError):
    """Malformed, oversized, or unsolicited protocol material."""


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        message = "MCP response must be an object"
        raise McpProtocolError(message)
    return cast("dict[str, object]", value)


def _result(value: object, request_id: int) -> dict[str, object] | None:
    message = _object(value)
    if message.get("jsonrpc") != "2.0":
        message = "invalid MCP JSON-RPC version"
        raise McpProtocolError(message)
    if "id" not in message and isinstance(message.get("method"), str):
        # Server notifications have no authority to initiate local effects.
        return None
    if type(message.get("id")) is not int or message["id"] != request_id:
        message = "MCP response correlation mismatch"
        raise McpProtocolError(message)
    if "error" in message or "method" in message:
        message = "MCP operation failed or requires unsupported client capabilities"
        raise McpProtocolError(message)
    return _object(message.get("result"))


def _sse_result(
    payload: bytes, request_id: int
) -> tuple[dict[str, object] | None, bytes]:
    while b"\n\n" in payload:
        event, payload = payload.split(b"\n\n", 1)
        data = b"\n".join(
            line[5:].removeprefix(b" ")
            for line in event.split(b"\n")
            if line.startswith(b"data:")
        )
        if data:
            result = _result(parse_json_value(data.decode("utf-8")), request_id)
            if result is not None:
                return result, payload
    return None, payload


async def _read_result(
    response: httpx.Response,
    request_id: int,
    limit: int,
) -> dict[str, object]:
    if response.headers.get("content-encoding", "identity") != "identity":
        message = "compressed MCP responses are not supported"
        raise McpProtocolError(message)
    content_type = (
        cast("str", response.headers.get("content-type", "")).split(";", 1)[0].strip()
    )
    if content_type not in {"application/json", "text/event-stream"}:
        message = "unsupported MCP response type"
        raise McpProtocolError(message)
    payload = bytearray()
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            message = "MCP response exceeds size limit"
            raise McpProtocolError(message)
        payload.extend(chunk)
        if content_type == "text/event-stream":
            # Normalize line endings after buffering; a CRLF may span chunks.
            normalized = bytes(payload).replace(b"\r\n", b"\n")
            result, normalized = _sse_result(normalized, request_id)
            if result is not None:
                return result
            payload = bytearray(normalized)
    if content_type == "application/json":
        result = _result(parse_json_value(payload.decode("utf-8")), request_id)
        if result is not None:
            return result
    message = "MCP stream ended without a correlated result"
    raise McpProtocolError(message)


@final
class _Exchange:
    def __init__(self, client: httpx.AsyncClient, server: McpServer) -> None:
        self.client = client
        self.server = server
        self.headers = {
            "Accept": "application/json, text/event-stream",
            "Accept-Encoding": "identity",
        }
        if server.token is not None:
            self.headers["Authorization"] = f"Bearer {server.token}"
        self.session: str | None = None
        self.call_started = False
        self.call_finished = False

    async def rpc(
        self, method: str, params: dict[str, object], request_id: int, limit: int
    ) -> dict[str, object]:
        async with self.client.stream(
            "POST",
            self.server.url,
            headers=self.headers,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        ) as response:
            _ = response.raise_for_status()
            if method == "initialize":
                session = cast("str | None", response.headers.get("mcp-session-id"))
                if session is not None:
                    if _SESSION_ID.fullmatch(session) is None:
                        message = "invalid MCP session identifier"
                        raise McpProtocolError(message)
                    self.session = session
                    self.headers["Mcp-Session-Id"] = session
            return await _read_result(response, request_id, limit)

    async def notify(
        self, method: str, params: dict[str, object] | None = None
    ) -> None:
        message: dict[str, object] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        async with self.client.stream(
            "POST", self.server.url, headers=self.headers, json=message
        ) as response:
            _ = response.raise_for_status()
            if response.status_code != HTTPStatus.ACCEPTED:
                reason = "MCP notification was not accepted"
                raise McpProtocolError(reason)

    async def close(self) -> None:
        with suppress(httpx.HTTPError, TimeoutError, McpProtocolError):
            async with asyncio.timeout(_CLEANUP_SECONDS):
                if self.call_started and not self.call_finished:
                    with suppress(httpx.HTTPError, McpProtocolError):
                        await self.notify(
                            "notifications/cancelled",
                            {"requestId": 2, "reason": "本轮请求已取消或超时"},
                        )
                if self.session is not None:
                    async with self.client.stream(
                        "DELETE", self.server.url, headers=self.headers
                    ):
                        pass


@final
class StreamableHttpMcpRequester:
    """No discovery, retries, shared server sessions, or background SSE readers."""

    def __init__(
        self,
        servers: tuple[McpServer, ...],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._servers = {server.name: server for server in servers}
        self._transport = transport
        self._tls = {
            server.name: build_tls_context(server.ca_path) or True for server in servers
        }

    async def request(
        self,
        allowance: McpToolAllowance,
        arguments: dict[str, object],
        *,
        timeout_ms: int,
    ) -> dict[str, object] | None:
        server = self._servers.get(allowance.server)
        if server is None:
            return None
        encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > allowance.max_request_bytes:
            return None
        try:
            async with httpx.AsyncClient(
                verify=self._tls[server.name],
                transport=self._transport,
                trust_env=False,
                follow_redirects=False,
                timeout=timeout_ms / 1000,
            ) as client:
                exchange = _Exchange(client, server)
                try:
                    async with asyncio.timeout(timeout_ms / 1000):
                        initialized = await exchange.rpc(
                            "initialize",
                            {
                                "protocolVersion": _PROTOCOL_VERSION,
                                "capabilities": {},
                                "clientInfo": {
                                    "name": "raspberrygirl-orchestrator",
                                    "version": "0.1.0",
                                },
                            },
                            1,
                            _CONTROL_BYTES,
                        )
                        version = initialized.get("protocolVersion")
                        if (
                            not isinstance(version, str)
                            or version not in _SUPPORTED_VERSIONS
                        ):
                            message = "unsupported MCP protocol version"
                            raise McpProtocolError(message)
                        capabilities = _object(initialized.get("capabilities"))
                        if not isinstance(capabilities.get("tools"), dict):
                            message = "MCP server does not support tools"
                            raise McpProtocolError(message)
                        exchange.headers["MCP-Protocol-Version"] = version
                        await exchange.notify("notifications/initialized")
                        exchange.call_started = True
                        result = await exchange.rpc(
                            "tools/call",
                            {"name": allowance.tool, "arguments": arguments},
                            2,
                            allowance.max_response_bytes,
                        )
                        exchange.call_finished = True
                        return result
                finally:
                    await exchange.close()
        except httpx.HTTPError:
            message = "MCP HTTP request failed"
            raise McpProtocolError(message) from None
