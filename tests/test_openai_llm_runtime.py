"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 _CapturedRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: path、authorization、body。
    """

    path: str

    authorization: str

    body: bytes


@dataclass(slots=True)
class _FakeOpenAICompatibleServer:
    """类契约说明.

    职责: 保存 _FakeOpenAICompatibleServer
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: requests、_server、_thread。
    方法:
    __post_init__、__enter__、__exit__。
    """

    requests: list[_CapturedRequest] = field(default_factory=list)

    _server: ThreadingHTTPServer = field(init=False)

    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化
        _FakeOpenAICompatibleServer
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        _OpenAICompatibleHandler.requests = self.requests

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        """函数契约说明.

        功能: 执行 __enter__ 的同步逻辑,并协调
        start。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """

        self._thread.start()

        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """函数契约说明.

        功能: 执行 __exit__ 的同步逻辑,并协调
        shutdown, join, server_close。
        参数: self 表示当前实例。 exc_type:
        object。 必填。 exc: object。 必填。
        traceback: object。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self._server.shutdown()

        self._thread.join(timeout=1.0)

        self._server.server_close()


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    """类契约说明.

    职责: 定义 _OpenAICompatibleHandler
    的状态、行为和对外协作边界。
    契约: 字段: requests。 方法:
    do_POST、log_message。
    """

    requests: ClassVar[list[_CapturedRequest]]

    def do_POST(self) -> None:
        """函数契约说明.

        功能: 执行 do_POST 的同步逻辑,并协调 read,
        append, encode, send_response。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

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
        """函数契约说明.

        功能: 执行 log_message 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 format: str。
        必填。 *args: object。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = (format, args)


def test_runtime_adapter_posts_openai_chat_completion_and_yields_final() -> None:
    # Given: a fake-local OpenAI-compatible completion server and typed turn request.

    """函数契约说明.

    功能: 验证 runtime adapter posts openai
    chat completion and yields final
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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
