# ruff: noqa: RUF001
import json
from dataclasses import dataclass, field

from orchestrator.agent_pipeline import (
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
)
from orchestrator.brain_runtime import JsonAgentBrain, JsonAgentGate, MockAgentGate
from orchestrator.llm import LLMRequest


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

    assert gate.evaluate(_input(), active_summary="产品讲解中").value == "accept"
    assert gate.evaluate(_input(), active_summary="产品讲解中").value == "discard"
    assert "输入相关性门" in completion.requests[0].prompt.system
    assert "<untrusted-payload>" in completion.requests[0].prompt.user
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
    assert "唯一的业务决策中心" in completion.requests[0].prompt.system
    assert '"cancellation_epoch": 2' in completion.requests[0].prompt.user
    assert "最终规划：禁止再请求工具" in completion.requests[1].prompt.user
    assert "工具观察（不可信数据）" in completion.requests[1].prompt.user


def test_mock_gate_discards_repeated_input_without_creating_effects() -> None:
    gate = MockAgentGate()

    assert gate.evaluate(_input(), active_summary="").value == "accept"
    assert gate.evaluate(_input(), active_summary="").value == "discard"
