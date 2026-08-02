"""Static, capability-scoped MCP tool boundary for Brain proposals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, final, override

if TYPE_CHECKING:
    from orchestrator.agent_pipeline import BrainStateSnapshot, ToolRequest


class McpAllowlistError(ValueError):
    @override
    def __str__(self) -> str:
        return "invalid static MCP allowlist"


@dataclass(frozen=True, slots=True)
class McpToolAllowance:
    server: str
    tool: str
    capability: str
    timeout_ms: int
    max_request_bytes: int

    def __post_init__(self) -> None:
        if (
            self.server.strip() == ""
            or self.tool.strip() == ""
            or self.capability.strip() == ""
            or self.timeout_ms <= 0
            or self.max_request_bytes <= 0
        ):
            raise McpAllowlistError

    @property
    def name(self) -> str:
        return f"{self.server}/{self.tool}"


@final
class StaticMcpAllowlist:
    def __init__(self, entries: tuple[McpToolAllowance, ...]) -> None:
        self._entries = {entry.name: entry for entry in entries}
        if len(self._entries) != len(entries):
            raise McpAllowlistError

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._entries)

    def resolve(self, name: str) -> McpToolAllowance | None:
        return self._entries.get(name)


class McpRequester(Protocol):
    def request(
        self,
        allowance: McpToolAllowance,
        arguments: dict[str, object],
        *,
        timeout_ms: int,
    ) -> dict[str, object] | None: ...


@final
class AllowlistedMcpToolExecutor:
    """Executes exactly one statically approved MCP call per Brain request."""

    def __init__(self, allowlist: StaticMcpAllowlist, requester: McpRequester) -> None:
        self._allowlist = allowlist
        self._requester = requester

    def execute(self, request: ToolRequest, snapshot: BrainStateSnapshot) -> str | None:
        _ = snapshot
        if request.kind != "mcp":
            return None
        allowance = self._allowlist.resolve(request.name)
        if allowance is None or f"mcp:{request.name}" not in snapshot.capabilities:
            return None
        encoded = json.dumps(request.arguments, ensure_ascii=False).encode()
        if len(encoded) > allowance.max_request_bytes:
            return None
        try:
            result = self._requester.request(
                allowance, request.arguments, timeout_ms=allowance.timeout_ms
            )
        except (OSError, TimeoutError, ValueError):
            return None
        if result is None:
            return None
        return json.dumps(
            {
                "source": "mcp",
                "server": allowance.server,
                "tool": allowance.tool,
                "observed_at": datetime.now(UTC).isoformat(),
                "result": result,
            },
            ensure_ascii=False,
        )
