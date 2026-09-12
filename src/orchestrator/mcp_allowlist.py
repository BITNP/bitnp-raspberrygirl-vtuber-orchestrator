"""Static, capability-scoped MCP tool boundary for Brain proposals."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, cast, final, override

if TYPE_CHECKING:
    from orchestrator.brain_contracts import BrainStateSnapshot, ToolRequest

_LOGGER = logging.getLogger(__name__)


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

    max_response_bytes: int = 16_384

    def __post_init__(self) -> None:
        if (
            self.server.strip() == ""
            or self.tool.strip() == ""
            or self.capability.strip() == ""
            or self.timeout_ms <= 0
            or self.max_request_bytes <= 0
            or self.max_response_bytes <= 0
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


class AsyncMcpRequester(Protocol):
    async def request(
        self,
        allowance: McpToolAllowance,
        arguments: dict[str, object],
        *,
        timeout_ms: int,
    ) -> dict[str, object] | None: ...


@final
class AsyncMcpToolExecutor:
    def __init__(
        self, allowlist: StaticMcpAllowlist, requester: AsyncMcpRequester
    ) -> None:
        self._allowlist = allowlist
        self._requester = requester

    async def execute(
        self, request: ToolRequest, snapshot: BrainStateSnapshot
    ) -> str | None:
        allowance = self._allowlist.resolve(request.name)
        if (
            request.kind != "mcp"
            or allowance is None
            or not {
                f"mcp:{request.name}",
                allowance.capability,
            }.issubset(snapshot.capabilities)
        ):
            return None
        outcome = "failed"
        payload: str | None = None
        observation: str | None = None
        try:
            encoded = json.dumps(
                request.arguments, ensure_ascii=False, allow_nan=False
            ).encode()
            if len(encoded) > allowance.max_request_bytes:
                return None
            async with asyncio.timeout(allowance.timeout_ms / 1000):
                result = await self._requester.request(
                    allowance,
                    request.arguments,
                    timeout_ms=allowance.timeout_ms,
                )
            observation = _mcp_observation(allowance, result)
            payload = None if result is None else _mcp_text(result)
            outcome = "success" if observation is not None else "failed"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except (OSError, TimeoutError, ValueError):
            observation = None
        finally:
            self._log_result(request, snapshot, observation, payload, outcome)
        return observation

    @staticmethod
    def _log_result(
        request: ToolRequest,
        snapshot: BrainStateSnapshot,
        observation: str | None,
        payload: str | None,
        outcome: str,
    ) -> None:
        _LOGGER.debug(
            "mcp_result trace=%s session=%s seq=%s turn=%s tool=%s payload=%r observation=%r outcome=%s",  # noqa: E501
            snapshot.input.trace_id,
            snapshot.session_id,
            snapshot.input.sequence,
            snapshot.turn_id,
            request.name,
            payload,
            observation,
            outcome,
        )


def _mcp_observation(
    allowance: McpToolAllowance, result: dict[str, object] | None
) -> str | None:
    """Only successful text enters context; binary and resource data stay out."""
    if result is None or result.get("isError", False) is not False:
        return None
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
    if len(encoded) > allowance.max_response_bytes:
        return None
    plain_text = _mcp_text(result)
    if plain_text is None:
        return None
    text = " ".join(plain_text.split())[:512]
    return (
        f"server_tool={allowance.name} status=success "
        f"digest=sha256:{sha256(encoded).hexdigest()} text={text}"
    )


def _mcp_text(result: dict[str, object]) -> str | None:
    content = result.get("content")
    if not isinstance(content, list):
        return None
    texts: list[str] = []
    for raw_block in cast("list[object]", content):
        block = (
            cast("dict[str, object]", raw_block)
            if isinstance(raw_block, dict)
            else None
        )
        if not isinstance(block, dict):
            return None
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(str(block["text"]))
        else:
            binary = json.dumps(block, ensure_ascii=False).encode()
            digest = sha256(binary).hexdigest()
            kind = block.get("type", "unknown")
            content_type = block.get("mimeType", "unknown")
            texts.append(
                " ".join(
                    (
                        f"非文本结果 kind={kind} content_type={content_type}",
                        f"bytes={len(binary)} digest=sha256:{digest}",
                    )
                )
            )
    if "structuredContent" in result:
        texts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(texts)
