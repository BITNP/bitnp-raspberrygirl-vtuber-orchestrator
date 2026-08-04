import asyncio

from orchestrator.asr_semantic_gate import (
    AsrGateDecision,
    AsrSemanticGate,
    AsyncAsrSemanticGate,
)


def test_gate_accepts_only_closed_structured_accept_decision() -> None:
    # Given: a Chinese semantic gate with a valid closed response.

    gate = AsrSemanticGate(lambda request: '{"decision":"accept"}')

    # When: a finalized ASR utterance is evaluated.

    decision = gate.evaluate("请介绍 BitNet")

    # Then: only the typed accept decision crosses the interactive boundary.

    assert decision is AsrGateDecision.ACCEPT


def test_gate_discards_malformed_timeout_and_unknown_decisions() -> None:
    # Given: malformed, unavailable, and unrecognized model responses.

    responses = (
        "not json",
        '{"decision":"accept","extra":true}',
        '{"decision":"unknown"}',
        TimeoutError(),
    )

    # When: each response is evaluated as a final ASR decision.

    decisions = tuple(
        AsrSemanticGate(
            lambda request, response=response: _response_or_raise(response)
        ).evaluate("继续")
        for response in responses
    )

    # Then: a gate failure never opens a meaningful turn.

    assert decisions == (
        AsrGateDecision.DISCARD,
        AsrGateDecision.DISCARD,
        AsrGateDecision.DISCARD,
        AsrGateDecision.DISCARD,
    )


def test_gate_only_allows_interrupt_while_audio_is_playing() -> None:
    gate = AsrSemanticGate(lambda request: '{"decision":"interrupt"}')

    assert gate.evaluate("请停一下") is AsrGateDecision.DISCARD
    assert gate.evaluate("请停一下", is_playing=True) is AsrGateDecision.INTERRUPT


def test_gate_discards_a_possible_echo_before_calling_the_provider() -> None:
    calls = 0

    def provider(request: object) -> str:
        nonlocal calls
        _ = request
        calls += 1
        return '{"decision":"accept"}'

    gate = AsrSemanticGate(provider)

    assert (
        gate.evaluate(
            "BitNet 可以在低比特下高效推理",
            active_answer_excerpt="欢迎使用。BitNet可以在低比特下高效推理。",
        )
        is AsrGateDecision.DISCARD
    )
    assert calls == 0


def test_gate_discards_a_longer_utterance_with_a_previous_answer_fragment() -> None:
    gate = AsrSemanticGate(lambda request: '{"decision":"accept"}')

    assert (
        gate.evaluate(
            "我听到你说BitNet可以在低比特下高效推理、请继续介绍",
            active_answer_excerpt="BitNet可以在低比特下高效推理、同时减少内存占用。",
        )
        is AsrGateDecision.DISCARD
    )


def test_async_gate_fails_closed_for_non_json_timeout_and_parameter_rejection() -> None:
    async def run() -> tuple[AsrGateDecision, ...]:
        async def response(value: str | BaseException) -> str:
            if isinstance(value, BaseException):
                raise value
            return value

        values: tuple[str | BaseException, ...] = (
            "not json",
            TimeoutError(),
            OSError("unsupported reasoning_effort"),
        )
        return tuple(
            [
                await AsyncAsrSemanticGate(
                    lambda request, value=value: response(value)
                ).evaluate("继续")
                for value in values
            ]
        )

    assert asyncio.run(run()) == (AsrGateDecision.DISCARD,) * 3


def _response_or_raise(response: str | TimeoutError) -> str:

    if isinstance(response, TimeoutError):
        raise response

    return response
