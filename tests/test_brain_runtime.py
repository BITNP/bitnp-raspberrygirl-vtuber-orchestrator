# ruff: noqa: RUF001
import asyncio
import json
from dataclasses import dataclass, field

from orchestrator.agent_pipeline import (
    AsyncAgentPipeline,
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
    PlanAccepted,
    ToolRequest,
)
from orchestrator.brain_runtime import (
    AsyncJsonAgentBrain,
    AsyncJsonAgentGate,
    JsonAgentBrain,
    JsonAgentGate,
    MockAgentGate,
    ReadonlyKnowledgeToolExecutor,
)
from orchestrator.llm import LLMRequest
from orchestrator.retrieval import KnowledgeRef, RetrievalFixtureProvider


@dataclass
class _Completion:
    responses: list[str]
    requests: list[LLMRequest] = field(default_factory=list)

    def complete_json(
        self,
        request: LLMRequest,
        *,
        schema_name: str,
        schema: dict[str, object],
        timeout_seconds: float,
    ) -> str:
        _ = schema_name, schema, timeout_seconds
        self.requests.append(request)
        return self.responses.pop(0)


@dataclass
class _AsyncCompletion:
    responses: list[str]
    requests: list[LLMRequest] = field(default_factory=list)

    async def complete_json(
        self,
        request: LLMRequest,
        *,
        schema_name: str,
        schema: dict[str, object],
        timeout_seconds: float,
    ) -> str:
        _ = schema_name, schema, timeout_seconds
        self.requests.append(request)
        return self.responses.pop(0)


def _input() -> AudienceInput:
    return AudienceInput("session-1", "trace-1", 1, AudienceSource.ASR, 1, "介绍产品")


def _snapshot() -> BrainStateSnapshot:
    return BrainStateSnapshot(
        session_id="session-1",
        turn_id="turn-1",
        revision=5,
        cancellation_epoch=2,
        input=_input(),
        context_summary="正在介绍产品",
        recent_context=("观众：请介绍产品",),
        memory_markdown="# memory\n",
        capabilities=frozenset({"knowledge.lookup"}),
    )


def test_gate_uses_chinese_non_streaming_json_prompt_and_fails_closed() -> None:
    completion = _Completion(['{"decision":"accept"}', "not-json"])
    gate = JsonAgentGate(completion)

    assert (
        gate.evaluate(
            _input(),
            active_summary="产品讲解中",
            recent_turn_context=("用户 - 它支持什么？", "智能体 - 支持语音交互。"),
        ).value
        == "accept"
    )
    assert gate.evaluate(_input(), active_summary="产品讲解中").value == "discard"
    assert "输入相关性门" in completion.requests[0].prompt.system
    assert "只能有一个键 decision" in completion.requests[0].prompt.system
    assert '{"decision":"accept"}' in completion.requests[0].prompt.system
    assert "<untrusted-payload>" in completion.requests[0].prompt.user
    assert "recent_turn_context" in completion.requests[0].prompt.user
    assert "支持语音交互" in completion.requests[0].prompt.user
    assert completion.requests[0].temperature == 0.0
    assert completion.requests[0].timeout_seconds == 5.0


def test_brain_injects_full_snapshot_and_marks_observations_untrusted() -> None:
    plan = json.dumps(
        {
            "response_text": "您好",
            "expected_revision": 5,
            "state_operations": [],
            "media_operations": [],
            "frontend_operations": [],
            "tool_requests": [],
            "citations": [],
            "memory_patches": [],
        }
    )
    completion = _Completion([plan, plan])
    brain = JsonAgentBrain(completion)

    assert brain.plan(_snapshot()) == plan
    assert brain.plan(_snapshot(), observations=("检索结果",)) == plan
    assert "多模态智能体的大脑" in completion.requests[0].prompt.system
    assert "对象必须恰好包含以下八个键" in completion.requests[0].prompt.system
    assert '"memory_patches":[]' in completion.requests[0].prompt.system
    assert "这是唯一会为 response_text 创建 TTS 合成任务的格式" in (
        completion.requests[0].prompt.system
    )
    assert "字幕不会自动从 response_text 生成" in completion.requests[0].prompt.system
    assert '"cancellation_epoch": 2' in completion.requests[0].prompt.user
    assert "expected_revision 必须等于 5" in completion.requests[0].prompt.user
    assert "最终规划：禁止再请求工具" in completion.requests[1].prompt.user
    assert "工具观察（不可信数据）" in completion.requests[1].prompt.user


def test_repair_requires_the_exact_snapshot_revision() -> None:
    completion = _Completion(["{}"])
    brain = JsonAgentBrain(completion)

    assert brain.repair(_snapshot(), '{"expected_revision": 6}') == "{}"
    assert "expected_revision 必须恰好为 5" in completion.requests[0].prompt.user
    assert "对象必须恰好包含以下八个键" in completion.requests[0].prompt.system
    assert "tool_requests 必须为 []" in completion.requests[0].prompt.system


def test_mock_gate_discards_repeated_input_without_creating_effects() -> None:
    gate = MockAgentGate()

    assert gate.evaluate(_input(), active_summary="").value == "accept"
    assert gate.evaluate(_input(), active_summary="").value == "discard"


def test_readonly_knowledge_tool_returns_versioned_untrusted_observation() -> None:
    executor = ReadonlyKnowledgeToolExecutor(
        RetrievalFixtureProvider(
            (KnowledgeRef("product.md:1", "产品", "树莓女孩可以讲解产品"),)
        )
    )

    observation = executor.execute(
        ToolRequest("knowledge", "local", {"query": "产品"}), _snapshot()
    )

    assert observation is not None
    assert '"source": "local_knowledge"' in observation
    assert "product.md:1" in observation


def test_async_json_brain_runs_gate_and_final_tool_plan_without_blocking() -> None:
    initial = json.dumps(
        {
            "response_text": "",
            "expected_revision": 5,
            "state_operations": [],
            "media_operations": [],
            "frontend_operations": [],
            "tool_requests": [
                {"kind": "knowledge", "name": "local", "arguments": {}}
            ],
            "citations": [],
            "memory_patches": [],
        }
    )
    final = json.dumps(
        {
            "response_text": "已查到资料",
            "expected_revision": 5,
            "state_operations": [],
            "media_operations": [],
            "frontend_operations": [],
            "tool_requests": [],
            "citations": [],
            "memory_patches": [],
        }
    )
    completion = _AsyncCompletion(['{"decision":"accept"}', initial, final])
    pipeline = AsyncAgentPipeline(
        AsyncJsonAgentGate(completion),
        AsyncJsonAgentBrain(completion),
        _AsyncTools(),
    )

    async def run() -> object:
        decision = await pipeline.submit(_input())
        assert decision.value == "accept"
        return await pipeline.run(_snapshot())

    result = asyncio.run(run())

    assert isinstance(result, PlanAccepted)
    assert result.plan.response_text == "已查到资料"
    assert len(completion.requests) == 3
    assert "工具观察" in completion.requests[2].prompt.user


class _AsyncTools:
    async def execute(self, request: ToolRequest, snapshot: BrainStateSnapshot) -> str:
        _ = request, snapshot
        return "本地知识 observation"
