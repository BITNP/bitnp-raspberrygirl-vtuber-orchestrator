
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from http.client import ResponseNotReady
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar, Literal, final, override

import pytest

from orchestrator import provider_streaming
from orchestrator.llm import (
    CancellationToken,
    LLMChunk,
    LLMFinal,
    LLMPrompt,
    LLMRequest,
    LLMStreamEvent,
)
from orchestrator.media_adapters import (
    ASRPartialEvent,
    ASRStreamRequest,
    OpenAICompatibleASRAdapter,
)
from orchestrator.openai_llm_runtime import OpenAICompatibleLLMRuntimeAdapter
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import ProviderDeadlines, ProviderResponseError

if TYPE_CHECKING:
    from collections.abc import Iterator


_StreamMode = Literal["asr", "llm", "malformed", "error", "block", "final_block"]


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
                        'data: {"text":"hello","is_final":false}\n\n',
                        'data: {"text":"hello world","is_final":true}\n\n',
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
class _RaceSocket:

    def settimeout(self, _seconds: float) -> None:

        return


@final
class _ResponseCloseRace:

    status: int = 200

    def __init__(self, read_error: Exception | None = None) -> None:

        self.entered: threading.Event = threading.Event()

        self.closed: bool = False

        self._closed: threading.Event = threading.Event()

        self._read_error = AttributeError() if read_error is None else read_error

    def read(self) -> bytes:

        self.entered.set()

        _ = self._closed.wait()

        raise self._read_error

    def close(self) -> None:

        self.closed = True

        self._closed.set()


@final
class _RaceConnection:

    def __init__(self, response: _ResponseCloseRace) -> None:

        self.sock: _RaceSocket = _RaceSocket()

        self.closed: bool = False

        self._response: _ResponseCloseRace = response

    def request(
        self, _method: str, _path: str, *, body: bytes, headers: dict[str, str]
    ) -> None:

        _ = (body, headers)

    def getresponse(self) -> _ResponseCloseRace:

        return self._response

    def close(self) -> None:

        self.closed = True


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
            ).stream(LLMRequest(prompt=LLMPrompt(system="system", user="question")))
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


def test_final_only_asr_cancellation_absorbs_response_close_attribute_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a final-only response whose concurrent close corrupts its read buffer.


    response = _ResponseCloseRace()

    connection = _RaceConnection(response)

    cancellation = CancellationToken()

    result: list[ASRAudienceEvent] = []

    failure: list[BaseException] = []

    def connection_factory(
        _url: str, _deadlines: ProviderDeadlines, _stage: str
    ) -> _RaceConnection:

        return connection

    monkeypatch.setattr(provider_streaming, "_connection", connection_factory)

    # When: cancellation closes the response while its read raises AttributeError.

    stream = OpenAICompatibleASRAdapter(
        endpoint="http://provider.example.test/v1",
        model="local-asr",
    ).stream(
        ASRStreamRequest(b"wav", "utterance.wav", 40, "segment-1", 2),
        cancellation=cancellation,
    )

    worker = threading.Thread(target=_consume_asr, args=(stream, result, failure))

    worker.start()

    assert response.entered.wait(timeout=1.0)

    assert cancellation.cancel(reason="newer_turn") is True

    worker.join(timeout=1.0)

    # Then: the race is a clean cancellation, with both owned resources closed.

    assert worker.is_alive() is False

    assert response.closed is True

    assert connection.closed is True

    assert result == []

    assert failure == []


def test_final_only_asr_cancellation_absorbs_response_not_ready_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: closing a final-only HTTP response makes its in-flight read idle.


    response = _ResponseCloseRace(ResponseNotReady("Idle"))

    connection = _RaceConnection(response)

    cancellation = CancellationToken()

    result: list[ASRAudienceEvent] = []

    failure: list[BaseException] = []

    def connection_factory(
        _url: str, _deadlines: ProviderDeadlines, _stage: str
    ) -> _RaceConnection:

        return connection

    monkeypatch.setattr(provider_streaming, "_connection", connection_factory)

    stream = OpenAICompatibleASRAdapter(
        endpoint="http://provider.example.test/v1",
        model="local-asr",
    ).stream(
        ASRStreamRequest(b"wav", "utterance.wav", 40, "segment-1", 2),
        cancellation=cancellation,
    )

    worker = threading.Thread(target=_consume_asr, args=(stream, result, failure))

    # When: cancellation closes the response while read raises ResponseNotReady.

    worker.start()

    assert response.entered.wait(timeout=1.0)

    assert cancellation.cancel(reason="newer_turn") is True

    worker.join(timeout=1.0)

    # Then: cancellation owns the race and no stale final or raw HTTP error escapes.

    assert worker.is_alive() is False

    assert response.closed is True

    assert connection.closed is True

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
            ).stream(LLMRequest(prompt=LLMPrompt(system="system", user="question")))
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
            ).stream(LLMRequest(prompt=LLMPrompt(system="system", user="question")))
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
            LLMRequest(prompt=LLMPrompt(system="system", user="question")),
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
