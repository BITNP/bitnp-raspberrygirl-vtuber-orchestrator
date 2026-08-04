from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar, cast, override

import httpx
import pytest

from orchestrator.asr_semantic_gate import AsrGateDecision, AsyncAsrSemanticGate
from orchestrator.llm import (
    BRAIN_MAX_COMPLETION_TOKENS,
    GATE_MAX_COMPLETION_TOKENS,
    MAINTENANCE_MAX_COMPLETION_TOKENS,
    LLMFinal,
    LLMPrompt,
    LLMRequest,
    LLMWorkload,
    ReasoningMode,
)
from orchestrator.openai_llm_runtime import AsyncOpenAICompatibleLLMRuntime
from orchestrator.provider_streaming import ProviderResponseError
from tests.openai_llm_test_helper import OpenAICompatibleLLMRuntimeAdapter


@dataclass(slots=True)
class _CapturedRequest:
    path: str

    authorization: str

    body: bytes


@dataclass(slots=True)
class _FakeOpenAICompatibleServer:
    requests: list[_CapturedRequest] = field(default_factory=list)

    _server: ThreadingHTTPServer = field(init=False)

    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:

        _OpenAICompatibleHandler.requests = self.requests

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> str:

        self._thread.start()

        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:

        self._server.shutdown()

        self._thread.join(timeout=1.0)

        self._server.server_close()


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[_CapturedRequest]]

    def do_POST(self) -> None:

        body = self.rfile.read(int(self.headers["content-length"]))

        self.requests.append(
            _CapturedRequest(
                path=self.path,
                authorization=self.headers["authorization"],
                body=body,
            )
        )

        response = json.dumps(
            {"choices": [{"message": {"content": " onsite answer "}}]}
        ).encode()

        self.send_response(200)

        self.send_header("content-type", "application/json")

        self.send_header("content-length", str(len(response)))

        self.end_headers()

        _ = self.wfile.write(response)

    @override
    def log_message(self, format: str, *args: object) -> None:

        _ = (format, args)


def test_runtime_adapter_posts_openai_chat_completion_and_yields_final() -> None:
    # Given: a fake-local OpenAI-compatible completion server and typed turn request.

    server = _FakeOpenAICompatibleServer()

    request = LLMRequest(
        prompt=LLMPrompt(system="system context", user="audience question"),
        workload=LLMWorkload.BRAIN,
        reasoning=ReasoningMode.ENABLED,
        max_completion_tokens=BRAIN_MAX_COMPLETION_TOKENS,
        temperature=0.25,
    )

    # When: the production adapter completes the request through real HTTP.

    with server as endpoint:
        events = tuple(
            OpenAICompatibleLLMRuntimeAdapter(
                endpoint=endpoint,
                model="onsite-model",
                api_key="local-secret",
            ).stream(request)
        )

    # Then: it sends the documented request and returns the provider final answer.

    assert len(server.requests) == 1
    captured = server.requests[0]
    assert captured.path == "/v1/chat/completions"
    assert captured.authorization == "Bearer local-secret"
    assert json.loads(captured.body) == {
        "model": "onsite-model",
        "messages": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "audience question"},
        ],
        "temperature": 0.25,
    }

    assert events == (LLMFinal(text="onsite answer", used_fallback=False),)


def test_async_runtime_routes_deepseek_gate_and_streaming_brain_parameters() -> None:
    captured: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = cast("dict[str, object]", json.loads(request.content))
        captured.append(body)
        if body["stream"] is False:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"decision":"accept"}'}}]},
            )
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async def run() -> tuple[str, tuple[object, ...]]:
        runtime = AsyncOpenAICompatibleLLMRuntime(
            endpoint="https://example.test/v1",
            model="test-model",
            api_key="test-key",
            reasoning_dialect="deepseek",
            gate_model="gate-model",
            brain_model="brain-model",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            gate = await runtime.complete_json(
                LLMRequest(
                    LLMPrompt("gate", "input"),
                    workload=LLMWorkload.GATE,
                    reasoning=ReasoningMode.DISABLED,
                    max_completion_tokens=GATE_MAX_COMPLETION_TOKENS,
                    temperature=0.0,
                    timeout_seconds=5.0,
                ),
                schema_name="audience_gate",
                schema={"type": "object"},
            )
            events = tuple(
                [
                    event
                    async for event in runtime.stream(
                        LLMRequest(
                            LLMPrompt("system", "user"),
                            workload=LLMWorkload.BRAIN,
                            reasoning=ReasoningMode.ENABLED,
                            max_completion_tokens=BRAIN_MAX_COMPLETION_TOKENS,
                        )
                    )
                ]
            )
            return gate, events
        finally:
            await runtime.aclose()

    gate, events = asyncio.run(run())

    assert gate == '{"decision":"accept"}'
    assert captured[0]["stream"] is False
    assert captured[0]["model"] == "gate-model"
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["temperature"] == 0.0
    assert captured[0]["max_tokens"] == GATE_MAX_COMPLETION_TOKENS
    assert captured[1]["stream"] is True
    assert captured[1]["model"] == "brain-model"
    assert captured[1]["thinking"] == {"type": "enabled"}
    assert captured[1]["temperature"] == 0.2
    assert captured[1]["max_tokens"] == BRAIN_MAX_COMPLETION_TOKENS
    assert events[-1] == LLMFinal(text="hello world", used_fallback=False)


def test_async_runtime_logs_complete_json_request_and_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(cast("dict[str, object]", json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def run() -> str:
        runtime = AsyncOpenAICompatibleLLMRuntime(
            endpoint="https://example.test/v1",
            model="test-model",
            api_key="test-key",
            reasoning_dialect="deepseek",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            return await runtime.complete_json(
                LLMRequest(
                    LLMPrompt("系统", "输入"),
                    workload=LLMWorkload.BRAIN,
                    reasoning=ReasoningMode.ENABLED,
                    max_completion_tokens=BRAIN_MAX_COMPLETION_TOKENS,
                ),
                schema_name="agent_plan",
                schema={"type": "object"},
            )
        finally:
            await runtime.aclose()

    with caplog.at_level(logging.DEBUG, logger="orchestrator.openai_llm_runtime"):
        assert asyncio.run(run()) == "{}"

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "llm_json_request workload=brain model=test-model schema=agent_plan" in message
        and "dialect=deepseek reasoning=enabled temperature=0.2" in message
        and "max_completion_tokens=8192 system='系统' user='输入'" in message
        for message in messages
    )
    assert "llm_json_response model=test-model schema=agent_plan text='{}'" in messages
    assert captured[0]["stream"] is False
    assert captured[0]["temperature"] == 0.2
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["max_tokens"] == BRAIN_MAX_COMPLETION_TOKENS


def test_async_gate_discards_provider_error_and_closes_shared_client() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(400, json={"error": {"message": "unsupported"}})

    async def run() -> tuple[object, bool]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = AsyncOpenAICompatibleLLMRuntime(
            endpoint="https://example.test/v1",
            model="test-model",
            api_key="test-key",
            reasoning_dialect="deepseek",
            http_client=client,
        )

        async def provider(request: object) -> str:
            _ = request
            return await runtime.complete_json(
                LLMRequest(
                    LLMPrompt("gate", "input"),
                    workload=LLMWorkload.GATE,
                    reasoning=ReasoningMode.DISABLED,
                    max_completion_tokens=GATE_MAX_COMPLETION_TOKENS,
                    temperature=0.0,
                    timeout_seconds=5.0,
                ),
                schema_name="audience_gate",
                schema={"type": "object"},
            )

        gate = AsyncAsrSemanticGate(provider)
        decision = await gate.evaluate("请继续")
        await runtime.aclose()
        return decision, client.is_closed

    decision, closed = asyncio.run(run())

    assert decision is AsrGateDecision.DISCARD
    assert closed


def test_async_runtime_maps_openai_reasoning_and_maintenance_model() -> None:
    captured: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(cast("dict[str, object]", json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def run() -> tuple[str, str]:
        runtime = AsyncOpenAICompatibleLLMRuntime(
            endpoint="https://example.test/v1",
            model="default-model",
            api_key="test-key",
            reasoning_dialect="openai",
            brain_model="brain-model",
            maintenance_model="maintenance-model",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            brain = await runtime.complete_json(
                LLMRequest(
                    LLMPrompt("brain", "input"),
                    workload=LLMWorkload.BRAIN,
                    reasoning=ReasoningMode.ENABLED,
                    max_completion_tokens=BRAIN_MAX_COMPLETION_TOKENS,
                ),
                schema_name="response_proposal",
                schema={"type": "object"},
            )
            maintenance = await runtime.complete_json(
                LLMRequest(
                    LLMPrompt("maintenance", "input"),
                    workload=LLMWorkload.MAINTENANCE,
                    reasoning=ReasoningMode.DISABLED,
                    max_completion_tokens=MAINTENANCE_MAX_COMPLETION_TOKENS,
                    temperature=0.0,
                    timeout_seconds=10.0,
                ),
                schema_name="context_compaction",
                schema={"type": "object"},
            )
            return brain, maintenance
        finally:
            await runtime.aclose()

    assert asyncio.run(run()) == ("{}", "{}")
    assert captured[0]["model"] == "brain-model"
    assert captured[0]["reasoning_effort"] == "medium"
    assert captured[0]["max_completion_tokens"] == BRAIN_MAX_COMPLETION_TOKENS
    assert "thinking" not in captured[0]
    assert "max_tokens" not in captured[0]
    assert captured[1]["model"] == "maintenance-model"
    assert captured[1]["reasoning_effort"] == "none"
    assert (
        captured[1]["max_completion_tokens"]
        == MAINTENANCE_MAX_COMPLETION_TOKENS
    )


def test_reasoning_only_response_fails_without_logging_hidden_reasoning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": None,
                            "reasoning_content": "private-reasoning",
                        },
                    }
                ]
            },
        )

    async def run() -> None:
        runtime = AsyncOpenAICompatibleLLMRuntime(
            endpoint="https://example.test/v1",
            model="brain-model",
            api_key="test-key",
            reasoning_dialect="deepseek",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            with pytest.raises(ProviderResponseError, match="missing_final"):
                _ = await runtime.complete_json(
                    LLMRequest(
                        LLMPrompt("system", "user"),
                        workload=LLMWorkload.BRAIN,
                        reasoning=ReasoningMode.ENABLED,
                        max_completion_tokens=BRAIN_MAX_COMPLETION_TOKENS,
                    ),
                    schema_name="response_proposal",
                    schema={"type": "object"},
                )
        finally:
            await runtime.aclose()

    with caplog.at_level(logging.DEBUG, logger="orchestrator.openai_llm_runtime"):
        asyncio.run(run())

    messages = [record.getMessage() for record in caplog.records]
    assert any("reasoning_chars=17" in message for message in messages)
    assert all("private-reasoning" not in message for message in messages)
