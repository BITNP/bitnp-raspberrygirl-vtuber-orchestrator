"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass, field
from typing import Literal

import pytest

from orchestrator.provider_streaming import ProviderResponseError
from orchestrator.semantic_barge_in import (
    ActiveAnswer,
    BargeInClassifierFailure,
    BargeInClassifierRequest,
    EndpointedTranscript,
    SemanticBargeInGate,
)
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    GeneratedSsrc,
    SegmentId,
    StreamFlush,
    StreamKey,
    TurnId,
)


@dataclass
class _Classifier:
    """类契约说明.

    职责: 保存 _Classifier
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: responses、requests。 方法:
    classify。
    """

    responses: list[str | Exception]

    requests: list[BargeInClassifierRequest] = field(default_factory=list)

    def classify(self, request: BargeInClassifierRequest) -> str:
        """函数契约说明.

        功能: 执行 classify 的同步逻辑,并协调
        append, pop, isinstance。
        参数: self 表示当前实例。 request:
        BargeInClassifierRequest。 必填。
        契约: 同步调用。 返回 `str`。
        """

        self.requests.append(request)

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


@dataclass
class _FlushSender:
    """类契约说明.

    职责: 保存 _FlushSender
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: flushes。 方法: send_flush。
    """

    flushes: list[StreamFlush] = field(default_factory=list)

    def send_flush(self, flush: StreamFlush) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.flushes.append(flush)


@dataclass
class _Clock:
    """类契约说明.

    职责: 保存 _Clock 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: now_ms。
    """

    now_ms: int = 0


def test_interrupt_cancels_active_turn_and_waits_for_matching_flush_ack() -> None:
    # Given: a stable endpoint transcript arrives while Sound plays an active answer.

    """函数契约说明.

    功能: 验证 interrupt cancels active turn
    and waits for matching flush ack
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    classifier = _Classifier(['{"decision":"interrupt"}'])

    sender = _FlushSender()

    gate = SemanticBargeInGate(classifier=classifier, clock=_Clock(), sender=sender)

    active = _active_answer()

    gate.activate(active)

    # When: the classifier explicitly approves interruption.

    gate.handle(_transcript("Please stop and explain that again."))

    # Then: only the active answer is cancelled and its replacement remains blocked.

    request = classifier.requests[0]

    assert request.transcript == "Please stop and explain that again."

    assert request.active_turn_id == active.turn_id

    assert request.active_segment_id == active.segment_id

    assert request.active_answer_excerpt == active.answer_excerpt

    assert request.timeout_ms == 400

    assert active.cancellation.cancelled is True

    assert active.cancellation.reason == "semantic_interrupt"

    assert gate.cancellations[0].targets == ("llm", "tts", "rtp")

    assert len(sender.flushes) == 1

    flush = sender.flushes[0]

    assert gate.cancellation_epoch == CancellationEpoch(8)

    assert gate.pop_admitted_replacement() is None

    gate.acknowledge(FlushAcknowledgement.from_flush(flush))

    assert gate.pop_admitted_replacement() == _transcript(
        "Please stop and explain that again."
    )


def test_continue_retains_active_playback_and_replaces_queued_utterance() -> None:
    # Given: an active answer and an older queued utterance on the same stream.

    """函数契约说明.

    功能: 验证 continue retains active
    playback and replaces queued
    utterance 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    classifier = _Classifier(['{"decision":"continue"}', '{"decision":"continue"}'])

    sender = _FlushSender()

    gate = SemanticBargeInGate(classifier=classifier, clock=_Clock(), sender=sender)

    active = _active_answer()

    gate.activate(active)

    gate.handle(_transcript("First follow-up."))

    # When: a newer stable utterance also classifies as continue.

    newest = _transcript(
        "Actually, make that concise.", turn="turn-new", segment="seg-new"
    )

    gate.handle(newest)

    # Then: playback continues, no flush is sent, and newest-wins queueing is retained.

    assert active.cancellation.cancelled is False

    assert sender.flushes == []

    assert gate.pop_queued_utterance() == newest

    assert gate.pop_queued_utterance() is None


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (TimeoutError("deadline"), "timeout"),
        ('{"decision":"replace"}', "malformed"),
        (ProviderResponseError(stage="barge_in", reason="status"), "unavailable"),
    ],
)
def test_classifier_failure_defaults_to_continue_with_one_correlated_record(
    response: str | Exception,
    reason: Literal["timeout", "malformed", "unavailable"],
) -> None:
    # Given: an active answer and a classifier that cannot provide a valid decision.

    """函数契约说明.

    功能: 验证 classifier failure defaults
    to continue with one correlated
    record 的回归场景和可观察结果。
    参数: response: str | Exception。 必填。
    reason: Literal['timeout',
    'malformed', 'unavailable']。 必填。
    契约: 同步调用。 返回 `None`。 可能抛出
    ProviderResponseError、TimeoutError。
    """

    classifier = _Classifier([response])

    sender = _FlushSender()

    gate = SemanticBargeInGate(classifier=classifier, clock=_Clock(), sender=sender)

    active = _active_answer()

    gate.activate(active)

    utterance = _transcript("Could you clarify the last point?")

    # When: the stable utterance is evaluated.

    gate.handle(utterance)

    # Then: the fallback preserves playback and emits exactly one correlated failure.

    assert active.cancellation.cancelled is False

    assert sender.flushes == []

    assert gate.pop_queued_utterance() == utterance

    assert gate.failures == (
        BargeInClassifierFailure(
            stream=utterance.stream,
            turn_id=active.turn_id,
            segment_id=active.segment_id,
            reason=reason,
        ),
    )


def test_stale_interrupt_result_cannot_cancel_a_replaced_active_answer() -> None:
    # Given: classification starts for one active answer, which then completes.

    """函数契约说明.

    功能: 验证 stale interrupt result cannot
    cancel a replaced active answer
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    classifier = _Classifier(['{"decision":"interrupt"}'])

    sender = _FlushSender()

    gate = SemanticBargeInGate(classifier=classifier, clock=_Clock(), sender=sender)

    first = _active_answer()

    gate.activate(first)

    decision = gate.classify(_transcript("Interrupt the old answer."))

    second = _active_answer(turn="turn-2", segment="seg-2", epoch=8)

    gate.activate(second)

    # When: the old explicit interrupt result arrives after active playback changed.

    gate.apply(decision)

    # Then: neither active playback nor the new answer is cancelled.

    assert first.cancellation.cancelled is False

    assert second.cancellation.cancelled is False

    assert sender.flushes == []


def _active_answer(
    *, turn: str = "turn-1", segment: str = "seg-1", epoch: int = 7
) -> ActiveAnswer:
    """函数契约说明.

    功能: 执行 _active_answer 的同步逻辑,并协调
    ActiveAnswer, StreamKey, TurnId,
    SegmentId。
    参数: turn: str。 可省略。 segment: str。
    可省略。 epoch: int。 可省略。
    契约: 同步调用。 返回 `ActiveAnswer`。
    """

    return ActiveAnswer(
        stream=StreamKey(session_id="session-1", stream_id="stream-1"),
        turn_id=TurnId(turn),
        segment_id=SegmentId(segment),
        cancellation_epoch=CancellationEpoch(epoch),
        answer_excerpt="A bounded active answer excerpt.",
        target_generated_ssrc=GeneratedSsrc(0x1234_5678),
    )


def _transcript(
    text: str, *, turn: str = "turn-next", segment: str = "seg-next"
) -> EndpointedTranscript:
    """函数契约说明.

    功能: 执行 _transcript 的同步逻辑,并协调
    EndpointedTranscript, StreamKey,
    TurnId, SegmentId。
    参数: text: str。 必填。 turn: str。 可省略。
    segment: str。 可省略。
    契约: 同步调用。 返回 `EndpointedTranscript`。
    """

    return EndpointedTranscript(
        stream=StreamKey(session_id="session-1", stream_id="stream-1"),
        text=text,
        turn_id=TurnId(turn),
        segment_id=SegmentId(segment),
    )
