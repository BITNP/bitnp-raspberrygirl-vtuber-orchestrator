import asyncio
import json
import logging
from typing import cast

import httpx
import pytest

from orchestrator.llm import (
    BRAIN_MAX_COMPLETION_TOKENS,
    MAINTENANCE_MAX_COMPLETION_TOKENS,
    LLMPrompt,
    LLMRequest,
    LLMWorkload,
    ReasoningMode,
)
from orchestrator.openai_llm_runtime import AsyncOpenAICompatibleLLMRuntime
from orchestrator.provider_streaming import ProviderResponseError


def _request(
    workload: LLMWorkload, reasoning: ReasoningMode, tokens: int
) -> LLMRequest:
    return LLMRequest(LLMPrompt("系统", "用户"), workload, reasoning, tokens)


def test_async_runtime_routes_brain_model_and_logs_complete_http_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured: list[dict[str, object]] = []
    response_payload = {
        "id": "completion-1",
        "choices": [
            {
                "message": {
                    "reasoning_content": "完整隐藏思考内容",
                    "content": "{}",
                }
            }
        ],
        "usage": {"prompt_tokens": 321, "completion_tokens": 45},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(cast("dict[str, object]", json.loads(request.content)))
        return httpx.Response(200, json=response_payload)

    async def run() -> str:
        runtime = AsyncOpenAICompatibleLLMRuntime(
            "https://example.test/v1",
            "default",
            "key",
            "deepseek",
            brain_model="brain",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            return await runtime.complete_json(
                _request(
                    LLMWorkload.BRAIN,
                    ReasoningMode.ENABLED,
                    BRAIN_MAX_COMPLETION_TOKENS,
                ),
                schema_name="brain_proposal",
                schema={"type": "object"},
            )
        finally:
            await runtime.aclose()

    caplog.set_level(logging.DEBUG, logger="orchestrator.openai_llm_runtime")
    assert asyncio.run(run()) == "{}"
    assert captured[0]["model"] == "brain"
    assert captured[0]["stream"] is False
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert "llm_json_http_response" in caplog.text
    assert "完整隐藏思考内容" in caplog.text
    assert '"prompt_tokens":321' in caplog.text
    assert '"completion_tokens":45' in caplog.text


def test_async_runtime_routes_maintenance_without_reasoning() -> None:
    captured: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(cast("dict[str, object]", json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def run() -> None:
        runtime = AsyncOpenAICompatibleLLMRuntime(
            "https://example.test/v1",
            "default",
            "key",
            "openai",
            maintenance_model="maintenance",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            _ = await runtime.complete_json(
                _request(
                    LLMWorkload.MAINTENANCE,
                    ReasoningMode.DISABLED,
                    MAINTENANCE_MAX_COMPLETION_TOKENS,
                ),
                schema_name="memory_candidate",
                schema={"type": "object"},
            )
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert captured[0]["model"] == "maintenance"
    assert captured[0]["reasoning_effort"] == "none"


def test_provider_error_is_raised_and_shared_client_closes() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(400, json={"error": {"message": "bad"}})

    async def run() -> bool:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = AsyncOpenAICompatibleLLMRuntime(
            "https://example.test/v1", "default", "key", "deepseek", http_client=client
        )
        with pytest.raises(ProviderResponseError):
            _ = await runtime.complete_json(
                _request(LLMWorkload.BRAIN, ReasoningMode.ENABLED, 100),
                schema_name="brain",
                schema={"type": "object"},
            )
        await runtime.aclose()
        return client.is_closed

    assert asyncio.run(run())
