
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar, override

from orchestrator.llm import LLMFinal, LLMPrompt, LLMRequest
from orchestrator.openai_llm_runtime import OpenAICompatibleLLMRuntimeAdapter


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

    assert server.requests == [
        _CapturedRequest(
            path="/v1/chat/completions",
            authorization="Bearer local-secret",
            body=json.dumps(
                {
                    "model": "onsite-model",
                    "messages": [
                        {"role": "system", "content": "system context"},
                        {"role": "user", "content": "audience question"},
                    ],
                    "stream": False,
                    "temperature": 0.25,
                }
            ).encode(),
        )
    ]

    assert events == (LLMFinal(text="onsite answer", used_fallback=False),)
