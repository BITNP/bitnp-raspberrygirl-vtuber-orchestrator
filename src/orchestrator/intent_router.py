"""Trusted mapping from small model intents to bounded tool arguments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import final

from orchestrator.agent_pipeline import BrainStateSnapshot, ToolRequest

type ArgumentBuilder = Callable[[BrainStateSnapshot], dict[str, object] | None]


class IntentSpecError(ValueError):
    """A trusted intent registration is incomplete or unsafe."""


class McpIntentMappingError(ValueError):
    """Configured MCP tools and trusted intent mappings differ."""


@dataclass(frozen=True, slots=True)
class IntentSpec:
    intent_id: str
    tool_kind: str
    tool_name: str
    required_capability: str
    build_arguments: ArgumentBuilder
    model_label: str = ""
    lane: str = "deliberative"
    timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        if (
            not self.intent_id.strip()
            or not self.tool_kind.strip()
            or not self.tool_name.strip()
            or not self.required_capability.strip()
            or self.timeout_ms <= 0
            or self.lane not in {"interactive", "deliberative"}
        ):
            raise IntentSpecError

    def available(self, snapshot: BrainStateSnapshot) -> bool:
        return self.required_capability in snapshot.capabilities


@final
class IntentRouter:
    def __init__(self, specs: tuple[IntentSpec, ...]) -> None:
        self._specs = {spec.intent_id: spec for spec in specs}
        if len(self._specs) != len(specs) or "answer" in self._specs:
            raise ValueError

    def allowed_intents(self, snapshot: BrainStateSnapshot) -> frozenset[str]:
        enabled = {
            spec.intent_id for spec in self._specs.values() if spec.available(snapshot)
        }
        return frozenset({"answer", *enabled})

    def request(
        self, intent_id: str, snapshot: BrainStateSnapshot
    ) -> ToolRequest | None:
        spec = self._specs.get(intent_id)
        if spec is None or not spec.available(snapshot):
            return None
        arguments = spec.build_arguments(snapshot)
        if arguments is None:
            return None
        return ToolRequest(spec.tool_kind, spec.tool_name, arguments)

    @property
    def specs(self) -> tuple[IntentSpec, ...]:
        """Expose immutable startup registrations for configuration validation."""
        return tuple(self._specs.values())

    def validate_mcp_allowlist(self, configured_names: frozenset[str]) -> None:
        """Fail closed when a configured MCP tool lacks trusted arguments.

        A configured server/tool is never implicitly model-accessible.  Its
        corresponding intent must explicitly target ``mcp`` and use the exact
        allowlist name; an extra mapping is also rejected so stale code cannot
        expose a removed allowance.
        """
        mapped = frozenset(
            spec.tool_name
            for spec in self._specs.values()
            if spec.tool_kind == "mcp"
        )
        if mapped != configured_names:
            missing = configured_names - mapped
            extra = mapped - configured_names
            raise McpIntentMappingError(missing, extra)
