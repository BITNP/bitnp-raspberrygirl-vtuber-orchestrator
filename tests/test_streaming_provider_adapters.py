from __future__ import annotations

import json
import ssl
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar, Literal, final, override

import pytest

from orchestrator import provider_streaming
from orchestrator.llm import (
    BRAIN_MAX_COMPLETION_TOKENS,
    CancellationToken,
    LLMChunk,
    LLMFinal,
    LLMPrompt,
    LLMRequest,
    LLMStreamEvent,
    LLMWorkload,
    ReasoningMode,
)
from orchestrator.media_adapters import (
    ASRPartialEvent,
    ASRStreamRequest,
    OpenAICompatibleASRAdapter,
)
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import (
    ProviderDeadlines,
    ProviderRequest,
    ProviderResponseError,
    post_bytes,
)
from tests.openai_llm_test_helper import OpenAICompatibleLLMRuntimeAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _llm_request() -> LLMRequest:
    return LLMRequest(
        prompt=LLMPrompt(system="system", user="question"),
        workload=LLMWorkload.BRAIN,
        reasoning=ReasoningMode.ENABLED,
        max_completion_tokens=BRAIN_MAX_COMPLETION_TOKENS,
    )


_StreamMode = Literal["asr", "llm", "malformed", "error", "block", "final_block"]


def _sse_data(payload: dict[str, str]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


@dataclass(slots=True)
class _FakeStreamingServer:
    mode: _StreamMode

    entered_block: threading.Event = field(default_factory=threading.Event)

    release_block: threading.Event = field(default_factory=threading.Event)

    _server: ThreadingHTTPServer = field(init=False)

    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:

        _StreamingHandler.mode = self.mode

        _StreamingHandler.entered_block = self.entered_block

        _StreamingHandler.release_block = self.release_block

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingHandler)

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> str:

        self._thread.start()

        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:

        self.release_block.set()

        self._server.shutdown()

        self._thread.join(timeout=1.0)

        self._server.server_close()


class _StreamingHandler(BaseHTTPRequestHandler):
    mode: ClassVar[_StreamMode]

    entered_block: ClassVar[threading.Event]

    release_block: ClassVar[threading.Event]

    def do_POST(self) -> None:

        _ = self.rfile.read(int(self.headers["content-length"]))

        match self.mode:
            case "asr":
                self._sse(
                    (
                        _sse_data({"type": "transcript.text.delta", "delta": "hello"}),
                        _sse_data(
                            {"type": "transcript.text.done", "text": "hello world"}
                        ),
                        "data: [DONE]\n\n",
                    )
                )

            case "llm":
                self._sse(
                    (
                        'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n',
                        'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
                        "data: [DONE]\n\n",
                    )
                )

            case "malformed":
                self._sse(("data: not-json\n\n",))

            case "error":
                self.send_response(503)

                self.end_headers()

            case "block":
                self.send_response(200)

                self.send_header("content-type", "text/event-stream")

                self.end_headers()

                _ = self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
                )

                self.wfile.flush()

                self.entered_block.set()

                _ = self.release_block.wait()

            case "final_block":
                self.send_response(200)

                self.send_header("content-type", "application/json")

                self.end_headers()

                self.entered_block.set()

                _ = self.release_block.wait()

                _ = self.wfile.write(b'{"text":"stale final"}')

    def _sse(self, chunks: tuple[str, ...]) -> None:

        self.send_response(200)

        self.send_header("content-type", "text/event-stream")

        self.end_headers()

        for chunk in chunks:
            _ = self.wfile.write(chunk.encode())

            self.wfile.flush()

    @override
    def log_message(self, format: str, *args: object) -> None:

        _ = (format, args)


@final
class _ProviderSocket:
    def settimeout(self, _seconds: float) -> None:
        return


@final
class _ProviderResponse:
    status: int = 200

    def read(self) -> bytes:
        return b""

    def close(self) -> None:
        return


@final
class _ProviderConnection:
    def __init__(self) -> None:
        self.sock = _ProviderSocket()

    def request(
        self, _method: str, _path: str, *, body: bytes, headers: dict[str, str]
    ) -> None:
        _ = (body, headers)

    def getresponse(self) -> _ProviderResponse:
        return _ProviderResponse()

    def close(self) -> None:
        return


@pytest.fixture
def ca_path(tmp_path: Path) -> Path:
    certificate = ssl.create_default_context().get_ca_certs(binary_form=True)[0]
    path = tmp_path / "ca.pem"
    _ = path.write_text(ssl.DER_cert_to_PEM_cert(certificate), encoding="ascii")
    return path


def test_provider_https_connection_receives_verified_configured_ca_context(
    monkeypatch: pytest.MonkeyPatch, ca_path: Path
) -> None:
    # Given: a provider HTTPS endpoint and configured local CA bundle.

    arguments: list[dict[str, ssl.SSLContext | float]] = []

    def connect(*_args: str, **kwargs: ssl.SSLContext | float) -> _ProviderConnection:
        arguments.append(kwargs)
        return _ProviderConnection()

    monkeypatch.setattr(provider_streaming, "HTTPSConnection", connect)

    # When: the shared provider transport opens the secure endpoint.

    _ = post_bytes(
        ProviderRequest(
            "https://provider.example.test/v1",
            b"",
            {},
            "llm",
            ca_path,
        ),
        _deadlines(),
        None,
    )

    # Then: HTTPSConnection receives the existing verified CA-based context.

    context = arguments[0]["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_provider_http_connection_omits_tls_context_even_with_configured_ca_bundle(
    monkeypatch: pytest.MonkeyPatch, ca_path: Path
) -> None:
    # Given: a plaintext provider endpoint and configured local CA bundle.

    arguments: list[dict[str, float]] = []

    def connect(*_args: str, **kwargs: float) -> _ProviderConnection:
        arguments.append(kwargs)
        return _ProviderConnection()

    monkeypatch.setattr(provider_streaming, "HTTPConnection", connect)

    # When: the shared provider transport opens the plaintext endpoint.

    _ = post_bytes(
        ProviderRequest(
            "http://provider.example.test/v1",
            b"",
            {},
            "asr",
            ca_path,
        ),
        _deadlines(),
        None,
    )

    # Then: the existing HTTP constructor arguments remain unchanged.

    assert arguments == [{"timeout": 1.0}]


def test_provider_https_connection_omits_context_without_configured_ca_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a secure provider endpoint without custom CA configuration.

    arguments: list[dict[str, float]] = []

    def connect(*_args: str, **kwargs: float) -> _ProviderConnection:
        arguments.append(kwargs)
        return _ProviderConnection()

    monkeypatch.setattr(provider_streaming, "HTTPSConnection", connect)

    # When: the shared provider transport opens the endpoint.

    _ = post_bytes(
        ProviderRequest(
            "https://provider.example.test/v1",
            b"",
            {},
            "llm",
        ),
        _deadlines(),
        None,
    )

    # Then: default HTTPS construction is unchanged.

    assert arguments == [{"timeout": 1.0}]


def _deadlines(*, read_seconds: float = 1.0) -> ProviderDeadlines:

    return ProviderDeadlines(
        connect_seconds=1.0,
        read_seconds=read_seconds,
        total_seconds=2.0,
    )


def test_streaming_asr_normalizes_partial_then_final_events() -> None:
    # Given: a fake OpenAI-compatible ASR server producing provider SSE events.

    server = _FakeStreamingServer(mode="asr")

    # When: a streaming-capable adapter transcribes an endpointed utterance.

    with server as endpoint:
        events = tuple(
            OpenAICompatibleASRAdapter(
                endpoint=endpoint,
                model="local-asr",
                capability="streaming",
                deadlines=_deadlines(),
            ).stream(ASRStreamRequest(b"wav", "utterance.wav", 40, "segment-1", 2))
        )

    # Then: provider partials and final are normalized into typed Orchestrator events.

    assert events == (
        ASRPartialEvent(text="hello", received_at_ms=40, segment_id="segment-1", seq=2),
        ASRAudienceEvent("hello world", 40, "segment-1", 2),
    )


def test_streaming_llm_emits_sse_tokens_in_provider_order() -> None:
    # Given: a fake chat-completions stream with two ordered deltas.

    server = _FakeStreamingServer(mode="llm")

    # When: the streaming-capable answer adapter consumes it.

    with server as endpoint:
        events = tuple(
            OpenAICompatibleLLMRuntimeAdapter(
                endpoint=endpoint,
                model="local-chat",
                api_key="test-secret",
                capability="streaming",
                deadlines=_deadlines(),
            ).stream(_llm_request())
        )

    # Then: delta ordering is preserved and the final event joins the provider text.

    assert events == (
        LLMChunk(index=0, text="hello "),
        LLMChunk(index=1, text="world"),
        LLMFinal(text="hello world", used_fallback=False),
    )


def test_final_only_asr_does_not_claim_streaming() -> None:
    # Given: the default OpenAI-compatible ASR adapter capability.

    adapter = OpenAICompatibleASRAdapter(
        endpoint="http://127.0.0.1:8000/v1",
        model="local-asr",
    )

    # When: a caller inspects the provider declaration.

    capability = adapter.capability

    # Then: completed-response ASR remains an explicit final-only fallback.

    assert capability == "final_only"


def test_final_only_asr_cancellation_closes_in_flight_response_without_final() -> None:
    # Given: an ASR final response blocked before it can return stale transcript text.

    server = _FakeStreamingServer(mode="final_block")

    cancellation = CancellationToken()

    result: list[ASRAudienceEvent] = []

    failure: list[BaseException] = []

    # When: a caller cancels the adapter while its final-only request is in flight.

    with server as endpoint:
        stream = OpenAICompatibleASRAdapter(
            endpoint=endpoint,
            model="local-asr",
            deadlines=_deadlines(),
        ).stream(
            ASRStreamRequest(b"wav", "utterance.wav", 40, "segment-1", 2),
            cancellation=cancellation,
        )

        worker = threading.Thread(
            target=_consume_asr,
            args=(stream, result, failure),
        )

        worker.start()

        assert server.entered_block.wait(timeout=1.0)

        assert cancellation.cancel(reason="newer_turn") is True

        server.release_block.set()

        worker.join(timeout=1.0)

    # Then: cancellation owns the request resource and prevents a stale final event.

    assert worker.is_alive() is False

    assert result == []

    assert failure == []


@pytest.mark.parametrize("mode", ["error", "malformed"])
def test_streaming_providers_reject_non_success_or_malformed_events(
    mode: Literal["error", "malformed"],
) -> None:
    # Given: a fake provider that fails before producing a valid event.

    server = _FakeStreamingServer(mode=mode)

    # When / Then: both provider boundaries expose a typed provider failure.

    with server as endpoint, pytest.raises(ProviderResponseError):
        _ = tuple(
            OpenAICompatibleLLMRuntimeAdapter(
                endpoint=endpoint,
                model="local-chat",
                api_key="test-secret",
                capability="streaming",
                deadlines=_deadlines(),
            ).stream(_llm_request())
        )


def test_streaming_llm_honors_read_deadline_without_time_based_test_sleep() -> None:
    # Given: a server that establishes a stream but sends no readable event line.

    server = _FakeStreamingServer(mode="block")

    # When / Then: the adapter raises its typed read deadline outcome.

    with server as endpoint, pytest.raises(ProviderResponseError, match="read"):
        _ = tuple(
            OpenAICompatibleLLMRuntimeAdapter(
                endpoint=endpoint,
                model="local-chat",
                api_key="test-secret",
                capability="streaming",
                deadlines=_deadlines(read_seconds=0.01),
            ).stream(_llm_request())
        )


def test_streaming_llm_cancellation_closes_mid_read_without_stale_tokens() -> None:
    # Given: a stream that has yielded one token and is blocked before its next token.

    server = _FakeStreamingServer(mode="block")

    cancellation = CancellationToken()

    result: list[LLMStreamEvent] = []

    failure: list[BaseException] = []

    # When: cancellation is requested while the iterator waits for a provider read.

    with server as endpoint:
        stream = OpenAICompatibleLLMRuntimeAdapter(
            endpoint=endpoint,
            model="local-chat",
            api_key="test-secret",
            capability="streaming",
            deadlines=_deadlines(),
        ).stream(
            _llm_request(),
            cancellation=cancellation,
        )

        first = next(stream)

        worker = threading.Thread(
            target=_consume_remaining,
            args=(stream, result, failure),
        )

        worker.start()

        assert server.entered_block.wait(timeout=1.0)

        assert cancellation.cancel(reason="newer_turn") is True

        worker.join(timeout=1.0)

    # Then: the blocked read releases and neither stale token nor final is emitted.

    assert first == LLMChunk(index=0, text="first")

    assert worker.is_alive() is False

    assert result == []

    assert failure == []


def _consume_remaining(
    stream: Iterator[LLMStreamEvent],
    result: list[LLMStreamEvent],
    failure: list[BaseException],
) -> None:

    try:
        result.extend(stream)

    except BaseException as error:  # noqa: BLE001 - captures worker failure for test assertion.
        failure.append(error)


def _consume_asr(
    stream: Iterator[ASRAudienceEvent],
    result: list[ASRAudienceEvent],
    failure: list[BaseException],
) -> None:

    try:
        result.extend(stream)

    except BaseException as error:  # noqa: BLE001 - captures worker failure for test assertion.
        failure.append(error)
