from orchestrator.llm import (
    CancellationToken,
    FallbackLLMAdapter,
    LLMChunk,
    LLMError,
    LLMFinal,
    LLMPrompt,
    LLMRequest,
    MockLLMAdapter,
    TimeoutLLMAdapter,
)


def test_cancellation_token_stops_pending_stream_and_is_idempotent() -> None:
    # Given: a deterministic stream with more chunks pending.
    adapter = MockLLMAdapter(answer_chunks=("first", "second", "third"))
    token = CancellationToken()
    request = LLMRequest(prompt=LLMPrompt(system="system", user="user"))
    stream = adapter.stream(request, cancellation=token)

    # When: the caller consumes one chunk and cancels the turn twice.
    first_event = next(stream)
    first_cancelled = token.cancel(reason="user_interrupt")
    second_cancelled = token.cancel(reason="duplicate_interrupt")
    remaining_events = tuple(stream)

    # Then: no later chunks or final answer are emitted after cancellation.
    assert first_event == LLMChunk(index=0, text="first")
    assert first_cancelled is True
    assert second_cancelled is False
    assert token.reason == "user_interrupt"
    assert remaining_events == ()


def test_timeout_adapter_returns_error_and_fallback_caption_deterministically() -> None:
    # Given: the primary adapter simulates a provider timeout before streaming.
    primary = TimeoutLLMAdapter(timeout_reason="provider deadline exceeded")
    adapter = FallbackLLMAdapter(
        primary=primary,
        fallback_text="I am having trouble answering right now.",
    )
    request = LLMRequest(prompt=LLMPrompt(system="system", user="user"))

    # When: the fallback adapter handles the request.
    events = tuple(adapter.stream(request))

    # Then: an error event requests downstream cancellation before fallback text.
    assert events == (
        LLMError(
            code="llm_timeout",
            message="provider deadline exceeded",
            cancel_pending_media=True,
        ),
        LLMFinal(
            text="I am having trouble answering right now.",
            used_fallback=True,
        ),
    )
