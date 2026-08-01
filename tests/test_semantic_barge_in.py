
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

    responses: list[str | Exception]

    requests: list[BargeInClassifierRequest] = field(default_factory=list)

    def classify(self, request: BargeInClassifierRequest) -> str:

        self.requests.append(request)

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


@dataclass
class _FlushSender:

    flushes: list[StreamFlush] = field(default_factory=list)

    def send_flush(self, flush: StreamFlush) -> None:

        self.flushes.append(flush)


@dataclass
class _Clock:

    now_ms: int = 0


def test_interrupt_cancels_active_turn_and_waits_for_matching_flush_ack() -> None:
    # Given: a stable endpoint transcript arrives while Sound plays an active answer.


    classifier = _Classifier(['{"decision":"interrupt"}'])

    sender = _FlushSender()

    gate = SemanticBargeInGate(classifier=classifier, clock=_Clock(), sender=sender)

    active = _active_answer(epoch=40)

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

    # Recognition only cancels obsolete compute.  Current Sound audio stays
    # live until the replacement has produced its first valid RTP frame.
    assert sender.flushes == []

    assert gate.replacement_audio_ready(active.stream) is True

    assert len(sender.flushes) == 1

    flush = sender.flushes[0]

    assert gate.cancellation_epoch == CancellationEpoch(41)

    assert gate.pop_admitted_replacement() is None

    gate.acknowledge(FlushAcknowledgement.from_flush(flush))

    assert gate.pop_admitted_replacement() == _transcript(
        "Please stop and explain that again."
    )


def test_continue_retains_active_playback_and_replaces_queued_utterance() -> None:
    # Given: an active answer and an older queued utterance on the same stream.


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

    return EndpointedTranscript(
        stream=StreamKey(session_id="session-1", stream_id="stream-1"),
        text=text,
        turn_id=TurnId(turn),
        segment_id=SegmentId(segment),
    )
