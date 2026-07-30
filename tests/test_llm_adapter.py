"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import pytest

from orchestrator.llm import (
    LLMChunk,
    LLMFinal,
    LLMRequest,
    MockLLMAdapter,
    OpenAICompatibleAdapter,
    build_llm_request,
)
from orchestrator.modes import (
    AdaptiveAgentPolicy,
    AudienceInput,
    AudienceSource,
)
from orchestrator.retrieval import KnowledgeRef


def test_mock_llm_streams_deterministic_chunks_and_final_answer() -> None:
    # Given: a lecturer prompt built from a selected answer candidate.

    """函数契约说明.

    功能: 验证 mock llm streams
    deterministic chunks and final
    answer 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    policy = AdaptiveAgentPolicy()

    candidate = policy.select_answer_candidate(
        (
            AudienceInput(
                source=AudienceSource.ASR,
                text="What is this theorem used for?",
                received_at_ms=2_000,
            ),
        ),
    )

    assert candidate is not None

    request = build_llm_request(
        candidate,
        context_refs=(
            KnowledgeRef(
                ref_id="slide-7",
                title="Slide 7 note",
                text="Use this theorem for bounded latency queues.",
            ),
        ),
    )

    adapter = MockLLMAdapter(answer_chunks=("Bounded ", "queues"))

    # When: the deterministic local adapter streams the answer.

    events = tuple(adapter.stream(request))

    # Then: chunks arrive in order and the final event carries the joined text.

    assert events == (
        LLMChunk(index=0, text="Bounded "),
        LLMChunk(index=1, text="queues"),
        LLMFinal(text="Bounded queues", used_fallback=False),
    )


def test_openai_compatible_adapter_builds_request_payload_without_api_key() -> None:
    # Given: a virtual streamer prompt and an OpenAI-compatible adapter boundary.

    """函数契约说明.

    功能: 验证 openai compatible adapter
    builds request payload without api
    key 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    policy = AdaptiveAgentPolicy()

    candidate = policy.select_answer_candidate(
        (
            AudienceInput(
                source=AudienceSource.COMMENT,
                text="What console had the best sound chip?",
                received_at_ms=10,
            ),
        ),
    )

    assert candidate is not None

    request = build_llm_request(candidate, context_refs=())

    adapter = OpenAICompatibleAdapter(model="local-chat", timeout_seconds=12.0)

    # When: unit code asks for the provider payload only.

    payload = adapter.build_payload(request)

    # Then: the payload uses the OpenAI chat-completions shape and no secret.

    assert payload["model"] == "local-chat"

    assert payload["stream"] is True

    assert payload["timeout_seconds"] == 12.0

    assert payload["messages"] == [
        {"role": "system", "content": request.prompt.system},
        {"role": "user", "content": request.prompt.user},
    ]

    assert "api_key" not in payload


def test_openai_adapter_rejects_malformed_model_boundary_input() -> None:
    # Given: malformed provider configuration crossing the adapter boundary.

    """函数契约说明.

    功能: 验证 openai adapter rejects
    malformed model boundary input
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    candidate = AdaptiveAgentPolicy().select_answer_candidate(
        (
            AudienceInput(
                source=AudienceSource.ASR,
                text="Where is registration?",
                received_at_ms=1,
            ),
        ),
    )

    assert candidate is not None

    request = LLMRequest(
        prompt=build_llm_request(
            candidate,
            context_refs=(),
        ).prompt,
    )

    # When / Then: the adapter rejects blank model names before payload creation.

    with pytest.raises(ValueError, match="model"):
        _ = OpenAICompatibleAdapter(model=" ").build_payload(request)
