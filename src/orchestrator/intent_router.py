"""Trusted operation registry and bounded argument validation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, final

from jsonschema import Draft202012Validator

from orchestrator.brain_contracts import BrainStateSnapshot, ToolRequest

if TYPE_CHECKING:
    from orchestrator.response_contracts import OperationProposal

type RuntimeArgumentBuilder = Callable[
    [Mapping[str, object], BrainStateSnapshot], dict[str, object] | None
]


class IntentSpecError(ValueError):
    """A trusted intent registration is incomplete or unsafe."""


class McpIntentMappingError(ValueError):
    """Configured MCP tools and trusted intent mappings differ."""


def identity_arguments(
    arguments: Mapping[str, object], snapshot: BrainStateSnapshot
) -> dict[str, object]:
    _ = snapshot
    return dict(arguments)


@dataclass(frozen=True, slots=True)
class IntentSpec:
    intent_id: str
    tool_kind: str
    tool_name: str
    required_capability: str
    argument_schema: Mapping[str, object]
    build_runtime_arguments: RuntimeArgumentBuilder = identity_arguments
    model_label: str = ""
    lane: str = "deliberative"
    timeout_ms: int = 30_000
    additional_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if (
            not self.intent_id.strip()
            or not self.tool_kind.strip()
            or not self.tool_name.strip()
            or not self.required_capability.strip()
            or self.timeout_ms <= 0
            or self.lane not in {"interactive", "deliberative"}
            or self.argument_schema.get("type") != "object"
            or self.argument_schema.get("additionalProperties") is not False
        ):
            raise IntentSpecError

    def available(self, snapshot: BrainStateSnapshot) -> bool:
        return (
            self.required_capability in snapshot.capabilities
            and self.additional_capabilities.issubset(snapshot.capabilities)
        )


@final
class IntentRouter:
    def __init__(self, specs: tuple[IntentSpec, ...]) -> None:
        self._specs = {spec.intent_id: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError

    def available_operations(
        self, snapshot: BrainStateSnapshot
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "intent": spec.intent_id,
                "description": spec.model_label,
                "arguments_schema": spec.argument_schema,
            }
            for spec in sorted(self._specs.values(), key=lambda item: item.intent_id)
            if spec.available(snapshot)
        )

    def request(
        self, proposal: OperationProposal, snapshot: BrainStateSnapshot
    ) -> ToolRequest | None:
        spec = self._specs.get(proposal.intent)
        if spec is None or not spec.available(snapshot):
            return None
        if not _validate_json_schema(proposal.arguments, spec.argument_schema):
            return None
        arguments = spec.build_runtime_arguments(proposal.arguments, snapshot)
        if arguments is None:
            return None
        try:
            _ = json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
        return ToolRequest(spec.tool_kind, spec.tool_name, arguments)

    def timeout_for(self, intent_id: str) -> int | None:
        spec = self._specs.get(intent_id)
        return None if spec is None else spec.timeout_ms

    def permits_request(
        self, request: ToolRequest, capabilities: frozenset[str]
    ) -> bool:
        """Revalidate a materialized request against current capabilities."""
        return any(
            spec.tool_kind == request.kind
            and spec.tool_name == request.name
            and spec.required_capability in capabilities
            and spec.additional_capabilities.issubset(capabilities)
            for spec in self._specs.values()
        )

    @property
    def specs(self) -> tuple[IntentSpec, ...]:
        return tuple(self._specs.values())

    def validate_mcp_allowlist(self, configured_names: frozenset[str]) -> None:
        mapped = frozenset(
            spec.tool_name for spec in self._specs.values() if spec.tool_kind == "mcp"
        )
        if mapped != configured_names:
            raise McpIntentMappingError(
                configured_names - mapped, mapped - configured_names
            )


class _SchemaValidator(Protocol):
    def is_valid(self, instance: object) -> bool: ...


def _validate_json_schema(value: object, schema: Mapping[str, object]) -> bool:
    validator = cast(
        "_SchemaValidator", cast("object", Draft202012Validator(dict(schema)))
    )
    return isinstance(value, dict) and validator.is_valid(
        cast("dict[str, object]", value)
    )
