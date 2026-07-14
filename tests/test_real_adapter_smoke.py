from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import override
from urllib.parse import urlparse

import pytest

from orchestrator.llm import LLMPrompt, LLMRequest, OpenAICompatibleAdapter

LLM_ENDPOINT_ENV = "BITNP_REAL_LLM_ENDPOINT"
LLM_API_KEY_ENV = "BITNP_REAL_LLM_API_KEY"
FAKE_LOCAL_ENV = "BITNP_REAL_ADAPTER_FAKE_LOCAL"
MALFORMED_ENV = "BITNP_REAL_ADAPTER_MALFORMED_CHECK"


@dataclass(frozen=True, slots=True)
class LLMReadinessError(Exception):
    endpoint: str
    reason: str

    @override
    def __str__(self) -> str:
        return (
            "OpenAI-compatible LLM readiness failed for "
            f"{self.endpoint}: {self.reason}"
        )


@pytest.mark.real_adapter
def test_openai_compatible_endpoint_smoke_when_explicitly_enabled() -> None:
    # Given: either a fake local OpenAI-compatible endpoint or an explicit endpoint.
    with _llm_endpoint_or_skip() as endpoint:
        request = LLMRequest(
            prompt=LLMPrompt(system="smoke", user="ping"),
            timeout_seconds=1.0,
        )
        payload = OpenAICompatibleAdapter(model="smoke-model").build_payload(request)
        body = json.dumps(
            {"model": payload["model"], "messages": payload["messages"]},
        ).encode()

        # When: the smoke sends a minimal chat-completions request.
        response_text = _post_chat_completion(endpoint, body)

    # Then: a provider-shaped response is received without default credentials.
    assert "choices" in response_text


@pytest.mark.real_adapter
def test_openai_compatible_malformed_endpoint_reports_readiness_error() -> None:
    # Given: malformed endpoint checking is explicitly enabled.
    if os.environ.get(MALFORMED_ENV) != "1":
        pytest.skip(f"set {MALFORMED_ENV}=1 to run malformed endpoint smoke")

    # When / Then: connection failure is reported as an explicit readiness error.
    with pytest.raises(
        LLMReadinessError,
        match="OpenAI-compatible LLM readiness failed",
    ):
        _ = _post_chat_completion("http://127.0.0.1:1", b'{"model":"smoke"}')


def _llm_endpoint_or_skip() -> _FakeLLMServer | _StaticEndpoint:
    if os.environ.get(FAKE_LOCAL_ENV) == "1":
        return _FakeLLMServer()
    endpoint = os.environ.get(LLM_ENDPOINT_ENV, "").strip()
    if endpoint == "":
        pytest.skip(f"set {LLM_ENDPOINT_ENV} or {FAKE_LOCAL_ENV}=1 to run LLM smoke")
    return _StaticEndpoint(endpoint)


@dataclass(frozen=True, slots=True)
class _StaticEndpoint:
    endpoint: str

    def __enter__(self) -> str:
        return self.endpoint

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakeLLMServer:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            _LLMHandler,
        )
        self._thread: threading.Thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )

    def __enter__(self) -> str:
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_port}/v1/chat/completions"

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._server.shutdown()
        self._thread.join(timeout=1.0)
        self._server.server_close()


class _LLMHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        _ = self.rfile.read(int(self.headers.get("content-length", "0")))
        body = json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()
        _ = self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    @override
    def log_message(self, format: str, *args: object) -> None:
        return None


def _post_chat_completion(endpoint: str, body: bytes) -> str:
    parsed = urlparse(endpoint)
    if parsed.hostname is None:
        raise LLMReadinessError(endpoint=endpoint, reason="missing host")
    path = parsed.path or "/"
    headers = {"content-type": "application/json"}
    api_key = os.environ.get(LLM_API_KEY_ENV)
    if api_key is not None and api_key.strip() != "":
        headers["authorization"] = f"Bearer {api_key.strip()}"
    connection = HTTPConnection(parsed.hostname, parsed.port or 80, timeout=1.0)
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        raw_body: bytes = response.read()
    except OSError as error:
        raise LLMReadinessError(endpoint=endpoint, reason=str(error)) from error
    finally:
        connection.close()
    return raw_body.decode()
