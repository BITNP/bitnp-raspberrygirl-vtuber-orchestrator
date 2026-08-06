import pytest

from orchestrator.intent_router import (
    IntentRouter,
    IntentSpec,
    IntentSpecError,
    McpIntentMappingError,
)


def _mcp_spec(name: str) -> IntentSpec:
    return IntentSpec(
        intent_id=f"intent_{name.replace('/', '_')}",
        tool_kind="mcp",
        tool_name=name,
        required_capability=f"mcp:{name}",
        argument_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {"query": {"type": "string", "maxLength": 64}},
        },
        model_label="受控工具",
    )


def test_mcp_allowlist_requires_an_exact_trusted_intent_mapping() -> None:
    router = IntentRouter((_mcp_spec("server/search"),))

    router.validate_mcp_allowlist(frozenset({"server/search"}))
    with pytest.raises(McpIntentMappingError):
        router.validate_mcp_allowlist(frozenset({"server/search", "server/get"}))
    with pytest.raises(McpIntentMappingError):
        router.validate_mcp_allowlist(frozenset())


def test_intent_spec_rejects_invalid_lifecycle_configuration() -> None:
    with pytest.raises(IntentSpecError):
        _ = IntentSpec(
            "bad",
            "mcp",
            "server/tool",
            "mcp:server/tool",
            {"type": "object", "additionalProperties": False},
            lane="bad",
        )
