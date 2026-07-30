"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 LLMReadinessError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: endpoint、reason。 方法:
    __str__。
    """

    endpoint: str

    reason: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """

        return (
            f"OpenAI-compatible LLM readiness failed for {self.endpoint}: {self.reason}"
        )


@pytest.mark.real_adapter
def test_openai_compatible_endpoint_smoke_when_explicitly_enabled() -> None:
    # Given: either a fake local OpenAI-compatible endpoint or an explicit endpoint.

    """函数契约说明.

    功能: 验证 openai compatible endpoint
    smoke when explicitly enabled
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 openai compatible malformed
    endpoint reports readiness error
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    if os.environ.get(MALFORMED_ENV) != "1":
        pytest.skip(f"set {MALFORMED_ENV}=1 to run malformed endpoint smoke")

    # When / Then: connection failure is reported as an explicit readiness error.

    with pytest.raises(
        LLMReadinessError,
        match="OpenAI-compatible LLM readiness failed",
    ):
        _ = _post_chat_completion("http://127.0.0.1:1", b'{"model":"smoke"}')


def test_provider_smoke_requires_explicit_opt_in_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the normal test environment has no provider endpoint or credential.

    """函数契约说明.

    功能: 验证 provider smoke requires
    explicit opt in without credentials
    的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。
    契约: 同步调用。 返回 `None`。
    """

    monkeypatch.delenv(FAKE_LOCAL_ENV, raising=False)

    monkeypatch.delenv(LLM_ENDPOINT_ENV, raising=False)

    monkeypatch.delenv(LLM_API_KEY_ENV, raising=False)

    # When / Then: the real-provider helper declines to construct any network target.

    with pytest.raises(pytest.skip.Exception, match="BITNP_REAL_LLM_ENDPOINT"):
        _ = _llm_endpoint_or_skip()


def _llm_endpoint_or_skip() -> _FakeLLMServer | _StaticEndpoint:
    """函数契约说明.

    功能: 执行 _llm_endpoint_or_skip
    的同步逻辑,并协调 strip, _StaticEndpoint,
    get, _FakeLLMServer。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `_FakeLLMServer |
    _StaticEndpoint`。
    """

    if os.environ.get(FAKE_LOCAL_ENV) == "1":
        return _FakeLLMServer()

    endpoint = os.environ.get(LLM_ENDPOINT_ENV, "").strip()

    if endpoint == "":
        pytest.skip(f"set {LLM_ENDPOINT_ENV} or {FAKE_LOCAL_ENV}=1 to run LLM smoke")

    return _StaticEndpoint(endpoint)


@dataclass(frozen=True, slots=True)
class _StaticEndpoint:
    """类契约说明.

    职责: 保存 _StaticEndpoint
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: endpoint。 方法:
    __enter__、__exit__。
    """

    endpoint: str

    def __enter__(self) -> str:
        """函数契约说明.

        功能: 执行 __enter__ 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """

        return self.endpoint

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """函数契约说明.

        功能: 执行 __exit__ 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 exc_type:
        object。 必填。 exc: object。 必填。 tb:
        object。 必填。
        契约: 同步调用。 返回 `None`。
        """

        return


class _FakeLLMServer:
    """类契约说明.

    职责: 定义 _FakeLLMServer 的状态、行为和对外协作边界。
    契约: 方法: __init__、__enter__、__exit__。
    """

    def __init__(self) -> None:
        """函数契约说明.

        功能: 初始化 _FakeLLMServer
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        self._server: ThreadingHTTPServer = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            _LLMHandler,
        )

        self._thread: threading.Thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )

    def __enter__(self) -> str:
        """函数契约说明.

        功能: 执行 __enter__ 的同步逻辑,并协调
        start。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """

        self._thread.start()

        return f"http://127.0.0.1:{self._server.server_port}/v1/chat/completions"

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """函数契约说明.

        功能: 执行 __exit__ 的同步逻辑,并协调
        shutdown, join, server_close。
        参数: self 表示当前实例。 exc_type:
        object。 必填。 exc: object。 必填。 tb:
        object。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self._server.shutdown()

        self._thread.join(timeout=1.0)

        self._server.server_close()


class _LLMHandler(BaseHTTPRequestHandler):
    """类契约说明.

    职责: 定义 _LLMHandler 的状态、行为和对外协作边界。
    契约: 方法: do_POST、log_message。
    """

    def do_POST(self) -> None:
        """函数契约说明.

        功能: 执行 do_POST 的同步逻辑,并协调 read,
        encode, send_response,
        send_header。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        _ = self.rfile.read(int(self.headers.get("content-length", "0")))

        body = json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()

        _ = self.send_response(200)

        self.send_header("content-type", "application/json")

        self.send_header("content-length", str(len(body)))

        self.end_headers()

        _ = self.wfile.write(body)

    @override
    def log_message(self, format: str, *args: object) -> None:
        """函数契约说明.

        功能: 执行 log_message
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 format: str。
        必填。 *args: object。 必填。
        契约: 同步调用。 返回 `None`。
        """

        return


def _post_chat_completion(endpoint: str, body: bytes) -> str:
    """函数契约说明.

    功能: 执行 _post_chat_completion
    的同步逻辑,并协调 urlparse, get,
    HTTPConnection, decode。
    参数: endpoint: str。 必填。 body: bytes。
    必填。
    契约: 同步调用。 返回 `str`。 可能抛出
    LLMReadinessError。
    """

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
