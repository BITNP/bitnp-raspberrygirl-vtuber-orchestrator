"""Static, local MCP server and trusted intent configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from orchestrator.json_boundary import parse_json_value
from orchestrator.mcp_allowlist import McpToolAllowance, StaticMcpAllowlist

if TYPE_CHECKING:
    from collections.abc import Mapping

_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_MAX_TEXT = 2048
_MAX_SERVERS = 16
_MAX_TOOLS = 32
_MAX_CONFIG_BYTES = 65_536


class McpConfigError(ValueError):
    """Configuration is invalid; never echo values that could contain secrets."""


@dataclass(frozen=True, slots=True)
class McpServer:
    name: str
    url: str = field(repr=False)
    token: str | None = field(default=None, repr=False)
    ca_path: Path | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class McpTool:
    allowance: McpToolAllowance
    intent: str
    description: str
    arguments_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class McpConfiguration:
    servers: tuple[McpServer, ...] = ()
    tools: tuple[McpTool, ...] = ()

    @property
    def allowlist(self) -> StaticMcpAllowlist:
        return StaticMcpAllowlist(tuple(tool.allowance for tool in self.tools))

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(
            name
            for tool in self.tools
            for name in (f"mcp:{tool.allowance.name}", tool.allowance.capability)
        )


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        message = "MCP configuration must contain objects"
        raise McpConfigError(message)
    return cast("dict[str, object]", value)


def _text(value: object, *, name: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT:
        message = "invalid MCP string"
        raise McpConfigError(message)
    if name and _NAME.fullmatch(value) is None:
        message = "invalid MCP identifier"
        raise McpConfigError(message)
    return value


def _integer(value: object, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        message = "invalid MCP budget"
        raise McpConfigError(message)
    return value


def _no_references(value: object) -> None:
    if isinstance(value, dict):
        for key, child in cast("dict[str, object]", value).items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                message = "MCP schemas must be self-contained"
                raise McpConfigError(message)
            _no_references(child)
    elif isinstance(value, list):
        for item in cast("list[object]", value):
            _no_references(item)


def _server(raw: object, env: Mapping[str, str]) -> McpServer:
    value = _object(raw)
    if set(value) - {"name", "url", "token_env", "ca_path"}:
        message = "unknown MCP server field"
        raise McpConfigError(message)
    name = _text(value.get("name"), name=True)
    url = _text(value.get("url"))
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (
            parsed.scheme == "http"
            and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        )
    ):
        message = "MCP requires HTTPS or loopback HTTP without URL credentials"
        raise McpConfigError(message)
    token = None
    if "token_env" in value:
        token = env.get(_text(value["token_env"], name=True))
        if not token or any(char in token for char in "\r\n"):
            message = "MCP bearer environment variable is missing or invalid"
            raise McpConfigError(message)
    ca_path = Path(_text(value["ca_path"])) if "ca_path" in value else None
    return McpServer(name, url, token, ca_path)


def _tool(raw: object) -> McpTool:
    value = _object(raw)
    expected = {
        "server",
        "tool",
        "capability",
        "intent",
        "description",
        "arguments_schema",
        "timeout_ms",
        "max_request_bytes",
        "max_response_bytes",
    }
    if set(value) != expected:
        message = "MCP tool configuration fields must be explicit"
        raise McpConfigError(message)
    schema = _object(value["arguments_schema"])
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        message = "MCP arguments must be a closed object schema"
        raise McpConfigError(message)
    _no_references(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        message = "invalid MCP argument schema"
        raise McpConfigError(message) from None
    intent = _text(value["intent"], name=True)
    if not intent.startswith("mcp."):
        message = "MCP intent must use the mcp namespace"
        raise McpConfigError(message)
    return McpTool(
        McpToolAllowance(
            _text(value["server"], name=True),
            _text(value["tool"], name=True),
            _text(value["capability"], name=True),
            _integer(value["timeout_ms"], 30_000),
            _integer(value["max_request_bytes"], 65_536),
            _integer(value["max_response_bytes"], 1_048_576),
        ),
        intent,
        _text(value["description"]),
        schema,
    )


def load_mcp_configuration(env: Mapping[str, str]) -> McpConfiguration:
    path = env.get("ORCHESTRATOR_MCP_CONFIG", "").strip()
    if not path:
        return McpConfiguration()
    with Path(path).open("rb") as stream:
        payload = stream.read(_MAX_CONFIG_BYTES + 1)
    if len(payload) > _MAX_CONFIG_BYTES:
        message = "MCP configuration exceeds size limit"
        raise McpConfigError(message)
    value = _object(parse_json_value(payload.decode("utf-8")))
    if (
        set(value) != {"version", "servers", "tools"}
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        message = "unsupported MCP configuration version or fields"
        raise McpConfigError(message)
    servers = value["servers"]
    tools = value["tools"]
    if (
        not isinstance(servers, list)
        or not isinstance(tools, list)
        or len(cast("list[object]", servers)) > _MAX_SERVERS
        or len(cast("list[object]", tools)) > _MAX_TOOLS
    ):
        message = "MCP server or tool limits exceeded"
        raise McpConfigError(message)
    result = McpConfiguration(
        tuple(_server(item, env) for item in cast("list[object]", servers)),
        tuple(_tool(item) for item in cast("list[object]", tools)),
    )
    names = {server.name for server in result.servers}
    if (
        len(names) != len(result.servers)
        or len({tool.intent for tool in result.tools}) != len(result.tools)
        or any(tool.allowance.server not in names for tool in result.tools)
    ):
        message = "duplicate MCP mapping or unknown server"
        raise McpConfigError(message)
    _ = result.allowlist
    return result
