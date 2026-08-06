"""Trusted operation registry and bounded argument validation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, final

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
        return self.required_capability in snapshot.capabilities


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


def _validate_json_schema(value: object, schema: Mapping[str, object]) -> bool:
    if not isinstance(value, dict):
        return False
    parsed_value = cast("dict[str, object]", cast("object", value))
    raw_properties = schema.get("properties", {})
    raw_required = schema.get("required", [])
    if not isinstance(raw_properties, dict) or not isinstance(raw_required, list):
        return False
    properties = cast("dict[str, object]", cast("object", raw_properties))
    required_values = cast("list[object]", cast("object", raw_required))
    if not all(isinstance(item, str) for item in required_values):
        return False
    required = cast("list[str]", cast("object", required_values))
    if not set(required).issubset(parsed_value) or not set(parsed_value).issubset(
        properties
    ):
        return False
    for name, item in parsed_value.items():
        field_schema = properties.get(name)
        if not isinstance(field_schema, dict) or not _validate_field(
            item, cast("dict[str, object]", field_schema)
        ):
            return False
    return True


def _validate_field(value: object, schema: Mapping[str, object]) -> bool:
    match schema.get("type"):
        case "string":
            return _validate_string(value, schema)
        case "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            return (not isinstance(minimum, int) or value >= minimum) and (
                not isinstance(maximum, int) or value <= maximum
            )
        case "boolean":
            return isinstance(value, bool)
        case _:
            return False


def _validate_string(value: object, schema: Mapping[str, object]) -> bool:
    if not isinstance(value, str):
        return False
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        return False
    minimum = schema.get("minLength", 0)
    maximum = schema.get("maxLength")
    return (
        isinstance(minimum, int)
        and len(value) >= minimum
        and (not isinstance(maximum, int) or len(value) <= maximum)
    )
