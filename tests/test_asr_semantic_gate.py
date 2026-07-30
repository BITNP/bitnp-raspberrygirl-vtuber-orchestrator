"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from orchestrator.asr_semantic_gate import AsrGateDecision, AsrSemanticGate


def test_gate_accepts_only_closed_structured_accept_decision() -> None:
    # Given: a Chinese semantic gate with a valid closed response.

    """函数契约说明.

    功能: 验证 gate accepts only closed
    structured accept decision
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    gate = AsrSemanticGate(lambda request: '{"decision":"accept"}')

    # When: a finalized ASR utterance is evaluated.

    decision = gate.evaluate("请介绍 BitNet")

    # Then: only the typed accept decision crosses the interactive boundary.

    assert decision is AsrGateDecision.ACCEPT


def test_gate_discards_malformed_timeout_and_unknown_decisions() -> None:
    # Given: malformed, unavailable, and unrecognized model responses.

    """函数契约说明.

    功能: 验证 gate discards malformed
    timeout and unknown decisions
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。 可能抛出
    TimeoutError。
    """

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
    """函数契约说明.

    功能: 执行 _response_or_raise 的同步逻辑,并协调
    isinstance。
    参数: response: str | TimeoutError。
    必填。
    契约: 同步调用。 返回 `str`。
    """

    if isinstance(response, TimeoutError):
        raise response

    return response
