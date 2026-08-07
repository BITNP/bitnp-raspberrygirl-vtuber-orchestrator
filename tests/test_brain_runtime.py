# ruff: noqa: RUF001
import asyncio
import json
from dataclasses import dataclass, field, replace

import pytest

from orchestrator.brain_contracts import (
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
    ToolRequest,
)
from orchestrator.brain_runtime import (
    AsyncJsonContextCompactor,
    AsyncJsonMemoryCandidateExtractor,
    JsonResponseBrain,
    McpIntentRegistration,
    build_async_response_coordinator,
    is_asr_clarification_speech,
    is_deterministic_asr_echo,
    is_explicit_asr_interruption,
    is_low_information_asr,
)
from orchestrator.ids import SessionId
from orchestrator.llm import (
    BRAIN_MAX_COMPLETION_TOKENS,
    MAINTENANCE_MAX_COMPLETION_TOKENS,
    LLMRequest,
    LLMWorkload,
    ReasoningMode,
)
from orchestrator.mcp_allowlist import McpToolAllowance, StaticMcpAllowlist
from orchestrator.response_contracts import (
    BrainDecision,
    OperationProposal,
    ResponseProposal,
)
from orchestrator.transient_context import (
    ContextComposition,
    TokenBudget,
    TransientContextSnapshot,
)


@dataclass
class _Completion:
    responses: list[str]
    requests: list[LLMRequest] = field(default_factory=list)

    def complete_json(
        self, request: LLMRequest, *, schema_name: str, schema: dict[str, object]
    ) -> str:
        _ = schema_name, schema
        self.requests.append(request)
        return self.responses.pop(0)


@dataclass
class _AsyncCompletion:
    responses: list[str]
    requests: list[LLMRequest] = field(default_factory=list)

    async def complete_json(
        self, request: LLMRequest, *, schema_name: str, schema: dict[str, object]
    ) -> str:
        _ = schema_name, schema
        self.requests.append(request)
        return self.responses.pop(0)


def _snapshot() -> BrainStateSnapshot:
    return BrainStateSnapshot(
        "session-1",
        "candidate-1",
        5,
        2,
        AudienceInput("session-1", "trace-1", 1, AudienceSource.ASR, 1, "介绍产品"),
        "正在介绍产品",
        ("观众：请介绍产品",),
        "# memory\n",
        frozenset({"mcp:web/search"}),
        was_playing_1000ms_ago=True,
    )


def test_single_brain_prompt_contains_brain_contract_and_playback_policy() -> None:
    completion = _Completion(
        [json.dumps({"decision": "accept", "speech": "您好", "operation": None})]
    )
    proposal = JsonResponseBrain(completion).respond(
        _snapshot(), available_operations=()
    )
    assert proposal.decision is BrainDecision.ACCEPT
    request = completion.requests[0]
    assert "唯一的业务决策 Brain" in request.prompt.system
    assert "was_playing_1000ms_ago" in request.prompt.user
    assert "该规则不适用于 comment" in request.prompt.system
    assert request.workload is LLMWorkload.BRAIN
    assert request.reasoning is ReasoningMode.DISABLED
    assert request.max_completion_tokens == BRAIN_MAX_COMPLETION_TOKENS


def test_brain_system_prompt_defines_input_output_syntax_and_semantics() -> None:
    completion = _Completion(
        [json.dumps({"decision": "accept", "speech": "您好", "operation": None})]
    )
    _ = JsonResponseBrain(completion).respond(
        _snapshot(), available_operations=()
    )
    system = completion.requests[0].prompt.system

    assert "【输入语法】" in system
    assert '"stage":"输入判定与回复"|"操作结果回复"' in system
    assert '"available_operations"' in system
    assert '"arguments_schema"' in system
    assert "【state 输入语法与语义】" in system
    assert '"input":{"source":"asr"|"comment"' in system
    assert "capabilities 是当前能力快照，但不能替代 available_operations 授权" in system
    assert "【输出语法】" in system
    assert '"decision":"accept"|"discard"' in system
    assert '"operation":null|{"intent":string,"arguments":object}' in system
    assert "不得输出 Markdown、代码围栏、解释文字或第二个对象" in system
    assert "【操作结果阶段】" in system


def test_brain_system_prompt_defines_allowed_inline_actions() -> None:
    completion = _Completion(
        [json.dumps({"decision": "accept", "speech": "您好", "operation": None})]
    )
    _ = JsonResponseBrain(completion).respond(
        _snapshot(), available_operations=()
    )
    system = completion.requests[0].prompt.system

    assert "【动作标记】" in system
    assert '<action name="hello"/>' in system
    assert '<action name="act_cute"/>' in system
    assert '<action name="emphasis"/>' in system
    assert "当前没有允许的 expression 标记" in system
    assert "动作标记只是 speech 时间线提示，不是 operation" in system


def test_malformed_brain_output_has_no_plain_text_fallback() -> None:
    with pytest.raises(ValueError, match="invalid Brain proposal"):
        _ = JsonResponseBrain(_Completion(["not-json"])).respond(
            _snapshot(), available_operations=()
        )


def test_deterministic_echo_applies_only_to_asr() -> None:
    assert is_deterministic_asr_echo(_snapshot().input, "这里介绍产品功能", ())
    comment = replace(_snapshot().input, source=AudienceSource.COMMENT)
    assert not is_deterministic_asr_echo(comment, "这里介绍产品功能", ())


@pytest.mark.parametrize(
    ("candidate", "reply"),
    [
        ("很高兴", "智能体 - 你好，很高兴为您服务"),
        ("而请您再重复一遍吗", "智能体 - 能请您再重复一遍吗"),
        (
            "请问您是想还是需要我继续为您服务呢",
            "智能体 - 请问您是想确认什么，还是需要我继续为您服务呢",
        ),
    ],
)
def test_deterministic_echo_tolerates_bounded_asr_fragment_errors(
    candidate: str, reply: str
) -> None:
    assert is_deterministic_asr_echo(
        replace(_snapshot().input, text=candidate), "", (reply,)
    )


def test_low_information_asr_rejects_single_character_noise_only() -> None:
    assert is_low_information_asr(replace(_snapshot().input, text="y"))
    assert is_low_information_asr(replace(_snapshot().input, text="あ"))
    assert not is_low_information_asr(replace(_snapshot().input, text="你好"))


@pytest.mark.parametrize(
    "speech",
    [
        "您好，我听到您说了一个 y。",
        "抱歉，我刚才没有听清楚，能请您再重复一遍吗？",
        "我这边听到的有些模糊，请再说一次。",
    ],
)
def test_asr_clarification_speech_is_fail_closed(speech: str) -> None:
    assert is_asr_clarification_speech(_snapshot().input, speech)
    assert not is_asr_clarification_speech(
        replace(_snapshot().input, source=AudienceSource.COMMENT), speech
    )


def test_explicit_repeat_request_may_receive_a_repeat_response() -> None:
    assert not is_asr_clarification_speech(
        replace(_snapshot().input, text="请再说一遍"),
        "好的，我重新说一遍。",
    )


@pytest.mark.parametrize(
    "text", ["停一下", "等等，让我说", "不对，你说错了", "换个话题"]
)
def test_explicit_asr_interruption_is_narrowly_recognized(text: str) -> None:
    assert is_explicit_asr_interruption(replace(_snapshot().input, text=text))


def test_ordinary_asr_and_comments_are_not_explicit_interruptions() -> None:
    ordinary = replace(_snapshot().input, text="我在听，请继续讲")
    assert not is_explicit_asr_interruption(ordinary)
    assert not is_explicit_asr_interruption(
        replace(ordinary, source=AudienceSource.COMMENT, text="停一下")
    )


def test_maintenance_calls_remain_independent() -> None:
    memory_completion = _AsyncCompletion(
        ['{"key":"drink","value":"绿茶","confidence":95}']
    )
    _ = asyncio.run(
        AsyncJsonMemoryCandidateExtractor(memory_completion).extract(
            user_text="喜欢绿茶", reply_text="记住了"
        )
    )
    assert memory_completion.requests[0].workload is LLMWorkload.MAINTENANCE
    assert (
        memory_completion.requests[0].max_completion_tokens
        == MAINTENANCE_MAX_COMPLETION_TOKENS
    )

    compact_completion = _AsyncCompletion(['{"summary":"摘要"}'])
    composition = ContextComposition(
        TransientContextSnapshot(SessionId("session-1"), 1, (), ""),
        (),
        (),
        TokenBudget(0),
    )
    assert (
        asyncio.run(AsyncJsonContextCompactor(compact_completion).compact(composition))
        == "摘要"
    )
    assert compact_completion.requests[0].reasoning is ReasoningMode.DISABLED


@dataclass
class _Requester:
    arguments: list[dict[str, object]] = field(default_factory=list)

    def request(
        self,
        allowance: McpToolAllowance,
        arguments: dict[str, object],
        *,
        timeout_ms: int,
    ) -> dict[str, object]:
        _ = allowance, timeout_ms
        self.arguments.append(arguments)
        return {"result": "晴"}


def test_mcp_operation_uses_its_own_schema_and_arguments() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 64}},
    }
    requester = _Requester()
    coordinator = build_async_response_coordinator(
        _AsyncCompletion([]),
        mcp_allowlist=StaticMcpAllowlist(
            (McpToolAllowance("web", "search", "network.search", 500, 128),)
        ),
        mcp_requester=requester,
        mcp_intents=(
            McpIntentRegistration("mcp.web_search", "web/search", "联网检索", schema),
        ),
    )
    snapshot = _snapshot()
    proposal = replace(
        JsonResponseBrain(
            _Completion(
                [
                    json.dumps(
                        {"decision": "accept", "speech": "我来查询", "operation": None}
                    )
                ]
            )
        ).respond(snapshot, available_operations=()),
        operation=OperationProposal("mcp.web_search", {"query": "上海天气"}),
    )
    request = coordinator.tool_request(proposal, snapshot)
    assert request is not None
    _ = asyncio.run(coordinator.execute_tool(request, snapshot))
    assert requester.arguments == [{"query": "上海天气"}]


@dataclass
class _PresentationExecutor:
    requests: list[dict[str, object]] = field(default_factory=list)

    async def execute(
        self, request: ToolRequest, snapshot: BrainStateSnapshot
    ) -> str | None:
        _ = snapshot
        self.requests.append(dict(request.arguments))
        return "已执行"


def test_presentation_operations_use_fixed_schemas_and_trusted_runtime_fields() -> None:
    executor = _PresentationExecutor()
    coordinator = build_async_response_coordinator(
        _AsyncCompletion([]),
        presentation_executor=executor,
        presentation_decks=frozenset({"launch-deck"}),
    )
    snapshot = replace(
        _snapshot(),
        turn_id="turn-7",
        capabilities=frozenset({"presentation.deck"}),
        ppt_deck_id="launch-deck",
        ppt_deck_version="v3",
        ppt_page=2,
    )

    operations = {
        item["intent"]: item["arguments_schema"]
        for item in coordinator.router.available_operations(snapshot)
    }
    assert set(operations) == {
        "presentation.load",
        "presentation.navigate",
        "presentation.play",
    }
    assert operations["presentation.load"] == {
        "type": "object",
        "additionalProperties": False,
        "required": ["deck_id"],
        "properties": {
            "deck_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 64,
                "enum": ["launch-deck"],
            }
        },
    }

    load = coordinator.tool_request(
        ResponseProposal(
            BrainDecision.ACCEPT,
            "正在加载演示文稿",
            OperationProposal("presentation.load", {"deck_id": "launch-deck"}),
        ),
        snapshot,
    )
    assert load is not None
    assert load.arguments == {
        "deck_id": "launch-deck",
        "deck_version": "v1",
        "page": 1,
        "command_id": "brain-turn-7-presentation",
        "session_id": "session-1",
        "turn_id": "turn-7",
    }

    navigate = coordinator.tool_request(
        ResponseProposal(
            BrainDecision.ACCEPT,
            "我来翻页",
            OperationProposal("presentation.navigate", {"page": 9}),
        ),
        snapshot,
    )
    assert navigate is not None
    assert navigate.arguments["page"] == 9
    assert navigate.arguments["deck_id"] == "launch-deck"
    assert navigate.arguments["deck_version"] == "v3"


@pytest.mark.parametrize(
    ("intent", "arguments"),
    [
        ("presentation.load", {"deck_id": "../../secret"}),
        ("presentation.load", {"deck_id": "launch-deck", "path": "other-deck"}),
        ("presentation.navigate", {"page": 0}),
        ("presentation.navigate", {"page": 10_001}),
        ("presentation.navigate", {"page": True}),
        ("presentation.play", {"page": 1}),
    ],
)
def test_presentation_operation_rejects_invalid_arguments(
    intent: str, arguments: dict[str, object]
) -> None:
    coordinator = build_async_response_coordinator(
        _AsyncCompletion([]),
        presentation_executor=_PresentationExecutor(),
        presentation_decks=frozenset({"launch-deck"}),
    )
    snapshot = replace(
        _snapshot(),
        capabilities=frozenset({"presentation.deck"}),
        ppt_deck_id="launch-deck",
        ppt_deck_version="v1",
        ppt_page=1,
    )
    proposal = ResponseProposal(
        BrainDecision.ACCEPT, "执行操作", OperationProposal(intent, arguments)
    )
    assert coordinator.tool_request(proposal, snapshot) is None
