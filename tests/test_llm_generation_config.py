import asyncio
import json
from typing import cast

import httpx
import pytest

from orchestrator.config import ConfigParseError, load_config_from_env
from orchestrator.llm import LLMPrompt, LLMRequest, LLMWorkload, ReasoningMode
from orchestrator.openai_llm_runtime import AsyncOpenAICompatibleLLMRuntime


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("workload", "expected"),
    [
        (
            LLMWorkload.BRAIN,
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.3,
                "presence_penalty": -0.2,
                "thinking": {"type": "enabled"},
                "max_tokens": 2048,
            },
        ),
        (
            LLMWorkload.MAINTENANCE,
            {"reasoning_effort": "low", "max_tokens": 512},
        ),
    ],
)
def test_env_overrides_reach_both_wire_paths(
    streaming: bool, workload: LLMWorkload, expected: dict[str, object]
) -> None:
    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_REASONING_DIALECT": "deepseek",
            "ORCHESTRATOR_LLM_TEMPERATURE": "0.7",
            "ORCHESTRATOR_LLM_TOP_P": "0.9",
            "ORCHESTRATOR_LLM_FREQUENCY_PENALTY": "0.3",
            "ORCHESTRATOR_LLM_PRESENCE_PENALTY": "-0.2",
            "ORCHESTRATOR_LLM_REASONING": "enabled",
            "ORCHESTRATOR_LLM_MAX_COMPLETION_TOKENS": "2048",
            "ORCHESTRATOR_LLM_BRAIN_TEMPERATURE": " ",
            "ORCHESTRATOR_LLM_MAINTENANCE_TEMPERATURE": "omit",
            "ORCHESTRATOR_LLM_MAINTENANCE_TOP_P": "omit",
            "ORCHESTRATOR_LLM_MAINTENANCE_FREQUENCY_PENALTY": "omit",
            "ORCHESTRATOR_LLM_MAINTENANCE_PRESENCE_PENALTY": "omit",
            "ORCHESTRATOR_LLM_MAINTENANCE_REASONING_DIALECT": "openai",
            "ORCHESTRATOR_LLM_MAINTENANCE_REASONING_EFFORT": "low",
            "ORCHESTRATOR_LLM_MAINTENANCE_TOKEN_PARAMETER": "max_tokens",
            "ORCHESTRATOR_LLM_MAINTENANCE_MAX_COMPLETION_TOKENS": "512",
        }
    )
    captured: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(cast("dict[str, object]", json.loads(request.content)))
        if streaming:
            chunk = {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "delta": {"content": "{}"}}],
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def run() -> None:
        runtime = AsyncOpenAICompatibleLLMRuntime(
            "https://example.test/v1",
            "test",
            "test",
            "deepseek",
            generation=config.llm_generation,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        request = LLMRequest(
            LLMPrompt("系统", "用户"), workload, ReasoningMode.DISABLED, 100
        )
        try:
            if streaming:
                events = [event async for event in runtime.stream(request)]
                assert events
            else:
                assert (
                    await runtime.complete_json(request, schema_name="test", schema={})
                    == "{}"
                )
        finally:
            await runtime.aclose()

    asyncio.run(run())
    body = captured[0]
    for key in ("model", "messages", "stream", "response_format"):
        _ = body.pop(key, None)
    assert body == expected


@pytest.mark.parametrize(
    ("suffix", "value"),
    [
        ("TEMPERATURE", "nan"),
        ("TEMPERATURE", "inf"),
        ("TEMPERATURE", "-0.1"),
        ("TEMPERATURE", "2.1"),
        ("TEMPERATURE", "invalid"),
        ("TOP_P", "1.1"),
        ("FREQUENCY_PENALTY", "-2.1"),
        ("PRESENCE_PENALTY", "2.1"),
        ("MAX_COMPLETION_TOKENS", "0"),
        ("MAX_COMPLETION_TOKENS", "-1"),
        ("MAX_COMPLETION_TOKENS", "1.5"),
        ("MAX_COMPLETION_TOKENS", "omit"),
        ("REASONING", "true"),
        ("REASONING_EFFORT", "extreme"),
        ("TOKEN_PARAMETER", "tokens"),
        ("REASONING_DIALECT", "unknown"),
    ],
)
@pytest.mark.parametrize("prefix", ["", "BRAIN_", "MAINTENANCE_"])
def test_invalid_generation_config_fails_at_startup(
    suffix: str, value: str, prefix: str
) -> None:
    key = f"ORCHESTRATOR_LLM_{prefix}{suffix}"
    with pytest.raises(ConfigParseError, match=key):
        _ = load_config_from_env({key: value})


def test_overrides_do_not_hide_invalid_global_config() -> None:
    with pytest.raises(ConfigParseError, match="ORCHESTRATOR_LLM_TEMPERATURE"):
        _ = load_config_from_env(
            {
                "ORCHESTRATOR_LLM_TEMPERATURE": "nan",
                "ORCHESTRATOR_LLM_BRAIN_TEMPERATURE": "0.5",
                "ORCHESTRATOR_LLM_MAINTENANCE_TEMPERATURE": "0.0",
            }
        )


@pytest.mark.parametrize("env", [{}, {"ORCHESTRATOR_LLM_TEMPERATURE": " "}])
def test_defaults_preserve_workload_request_values(env: dict[str, str]) -> None:
    config = load_config_from_env(env).llm_generation
    for settings, temperature, tokens in (
        (config.brain, 0.2, 8192),
        (config.maintenance, 0.0, 4096),
    ):
        assert settings.parameters(
            temperature=temperature,
            reasoning="disabled",
            max_completion_tokens=tokens,
            dialect="deepseek",
        ) == {
            "temperature": temperature,
            "thinking": {"type": "disabled"},
            "max_tokens": tokens,
        }


@pytest.mark.parametrize(
    "env",
    [
        {"ORCHESTRATOR_LLM_REASONING_DIALECT": "none"},
        {"ORCHESTRATOR_LLM_REASONING": "omit"},
    ],
)
def test_compatibility_omission_keeps_output_budget(env: dict[str, str]) -> None:
    settings = load_config_from_env(
        env
        | {
            "ORCHESTRATOR_LLM_TEMPERATURE": "omit",
            "ORCHESTRATOR_LLM_TOKEN_PARAMETER": "max_completion_tokens",
        }
    ).llm_generation.brain
    assert settings.parameters(
        temperature=0.2,
        reasoning="disabled",
        max_completion_tokens=100,
        dialect="deepseek",
    ) == {"max_completion_tokens": 100}
