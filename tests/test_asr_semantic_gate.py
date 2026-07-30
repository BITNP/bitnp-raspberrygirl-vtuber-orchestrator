from orchestrator.asr_semantic_gate import AsrGateDecision, AsrSemanticGate


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


def _response_or_raise(response: str | TimeoutError) -> str:
    if isinstance(response, TimeoutError):
        raise response
    return response
