from dataclasses import dataclass

from orchestrator.agent_pipeline import (
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
    ToolRequest,
)
from orchestrator.json_boundary import parse_json_value
from orchestrator.mcp_allowlist import (
    AllowlistedMcpToolExecutor,
    McpToolAllowance,
    StaticMcpAllowlist,
)


@dataclass
class _Requester:
    calls: list[tuple[str, int]]

    def request(
        self,
        allowance: McpToolAllowance,
        arguments: dict[str, object],
        *,
        timeout_ms: int,
    ) -> dict[str, object] | None:
        self.calls.append((allowance.name, timeout_ms))
        return {"url": "https://example.test", "echo": arguments}


def _snapshot(capabilities: frozenset[str]) -> BrainStateSnapshot:
    return BrainStateSnapshot(
        session_id="session-1",
        turn_id="turn-1",
        revision=1,
        cancellation_epoch=0,
        input=AudienceInput(
            "session-1", "trace-1", 1, AudienceSource.COMMENT, 1, "查询"
        ),
        context_summary="",
        recent_context=(),
        memory_markdown="# 会话记忆\n",
        capabilities=capabilities,
    )


def test_static_allowlist_executes_only_matching_capability_and_bounds_request() -> (
    None
):
    allowance = McpToolAllowance("web", "search", "network.search", 500, 64)
    requester = _Requester([])
    executor = AllowlistedMcpToolExecutor(StaticMcpAllowlist((allowance,)), requester)
    request = ToolRequest("mcp", "web/search", {"query": "树莓女孩"})

    observation = executor.execute(request, _snapshot(frozenset({"mcp:web/search"})))

    assert observation is not None
    parsed = parse_json_value(observation)
    assert isinstance(parsed, dict)
    assert parsed["source"] == "mcp"
    assert parsed["server"] == "web"
    assert requester.calls == [("web/search", 500)]
    assert executor.execute(request, _snapshot(frozenset())) is None
    assert requester.calls == [("web/search", 500)]


def test_allowlist_rejects_duplicate_entries_and_oversized_requests() -> None:
    allowance = McpToolAllowance("deck", "load", "ppt", 500, 8)
    requester = _Requester([])
    executor = AllowlistedMcpToolExecutor(StaticMcpAllowlist((allowance,)), requester)

    assert (
        executor.execute(
            ToolRequest("mcp", "deck/load", {"deck": "too-long"}),
            _snapshot(frozenset({"mcp:deck/load"})),
        )
        is None
    )
    assert requester.calls == []
