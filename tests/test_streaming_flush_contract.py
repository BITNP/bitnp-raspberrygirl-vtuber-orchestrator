from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    FlushAdmission,
    FlushFailure,
    FlushRequestId,
    GeneratedSsrc,
    SegmentId,
    StreamFlush,
    StreamKey,
    TurnId,
)
from orchestrator.transport_control import (
    ControlEnvelopeError,
    EnvelopeCorrelation,
    VoiceEvidence,
    parse_control_event,
)


def _flush_envelope(*, session_id: str = "session-001", epoch: int = 3) -> str:

    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": "media.stream.flush",
            "event_id": "flush-event-001",
            "source": "orchestrator",
            "time": "2026-07-28T00:00:00Z",
            "trace_id": "trace-001",
            "session_id": session_id,
            "turn_id": "turn-001",
            "segment_id": "segment-001",
            "seq": 1,
            "data": {
                "stream_id": "stream-001",
                "cancellation_epoch": epoch,
                "request_id": "flush-request-001",
                "target_generated_ssrc": 305419896,
            },
        }
    )


def test_flush_envelope_parses_every_epoch_correlated_identity() -> None:
    # Given: a canonical generated-media flush envelope.

    # When: the WSS boundary parses it.

    flush = parse_control_event(_flush_envelope())

    # Then: all replacement-admission correlation identities survive parsing.

    assert flush == StreamFlush(
        stream=StreamKey(session_id="session-001", stream_id="stream-001"),
        turn_id=TurnId("turn-001"),
        segment_id=SegmentId("segment-001"),
        cancellation_epoch=CancellationEpoch(3),
        request_id=FlushRequestId("flush-request-001"),
        target_generated_ssrc=GeneratedSsrc(305419896),
        correlation=EnvelopeCorrelation(
            trace_id="trace-001", session_id="session-001", seq=1
        ),
    )

    correlation = flush.correlation

    assert correlation is not None

    assert (
        correlation.trace_id,
        correlation.session_id,
        correlation.seq,
    ) == ("trace-001", "session-001", 1)


def test_flush_acknowledgement_parses_every_envelope_and_command_correlation() -> None:
    # Given: a Sound acknowledgement preserving a generated-media flush identity.

    acknowledgement_envelope = (
        _flush_envelope()
        .replace('"media.stream.flush"', '"media.stream.flush.ack"')
        .replace('"orchestrator"', '"sound"')
    )

    # When: the WSS boundary parses the acknowledgement.

    acknowledgement = parse_control_event(acknowledgement_envelope)

    # Then: envelope and replacement identities remain exactly joinable.

    assert acknowledgement == FlushAcknowledgement(
        stream=StreamKey(session_id="session-001", stream_id="stream-001"),
        turn_id=TurnId("turn-001"),
        segment_id=SegmentId("segment-001"),
        cancellation_epoch=CancellationEpoch(3),
        request_id=FlushRequestId("flush-request-001"),
        target_generated_ssrc=GeneratedSsrc(305419896),
        correlation=EnvelopeCorrelation(
            trace_id="trace-001", session_id="session-001", seq=1
        ),
    )

    correlation = acknowledgement.correlation

    assert correlation is not None

    assert (
        correlation.trace_id,
        correlation.session_id,
        correlation.seq,
    ) == ("trace-001", "session-001", 1)


def test_flush_envelope_rejects_missing_epoch() -> None:
    # Given: a flush missing its cancellation epoch.

    envelope = _flush_envelope().replace('"cancellation_epoch": 3, ', "")

    # When: the boundary parses the malformed event.

    with pytest.raises(ControlEnvelopeError) as error:
        _ = parse_control_event(envelope)

    # Then: uncorrelated flush cannot reach Sound.

    assert error.value.field_name == "data.cancellation_epoch"


def test_voice_evidence_is_bounded_and_keeps_rtp_correlation() -> None:
    envelope = json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": "voice.evidence",
            "event_id": "voice-1",
            "source": "mic",
            "time": "2026-08-02T00:00:00Z",
            "trace_id": "trace-001",
            "session_id": "session-001",
            "seq": 7,
            "data": {
                "stream_id": "stream-001",
                "rtp_start_timestamp": 1_000,
                "rtp_end_timestamp": 4_200,
                "embedding_model_revision": "camplusplus-onnx-v1",
                "embedding": [0.25, -0.5],
                "quality": {"speech_ms": 200, "score": 0.91},
            },
        }
    )

    evidence = parse_control_event(envelope)

    assert evidence == VoiceEvidence(
        session_id="session-001",
        stream_id="stream-001",
        rtp_start_timestamp=1_000,
        rtp_end_timestamp=4_200,
        embedding_model_revision="camplusplus-onnx-v1",
        embedding=(0.25, -0.5),
        speech_ms=200,
        quality_score=0.91,
        correlation=EnvelopeCorrelation("trace-001", "session-001", 7),
    )


@dataclass
class _FakeClock:
    now_ms: int = 0

    def advance(self, milliseconds: int) -> None:

        self.now_ms += milliseconds


@dataclass
class _RecordingFlushSender:
    sent: list[StreamFlush] = field(default_factory=list)

    def send_flush(self, flush: StreamFlush) -> None:

        self.sent.append(flush)


def _flush() -> StreamFlush:

    return StreamFlush(
        stream=StreamKey(session_id="session-001", stream_id="stream-001"),
        turn_id=TurnId("turn-001"),
        segment_id=SegmentId("segment-001"),
        cancellation_epoch=CancellationEpoch(3),
        request_id=FlushRequestId("flush-request-001"),
        target_generated_ssrc=GeneratedSsrc(305419896),
    )


def test_replacement_admission_retries_once_then_accepts_matching_ack() -> None:
    # Given: a flush request whose Sound acknowledgement is delayed past its retry.

    clock = _FakeClock()

    sender = _RecordingFlushSender()

    admission = FlushAdmission(clock=clock, sender=sender)

    flush = _flush()

    # When: the fake clock reaches 250ms and Sound returns the matching acknowledgement.

    admission.begin(flush)

    clock.advance(250)

    admission.advance()

    _ = admission.acknowledge(FlushAcknowledgement.from_flush(flush))

    # Then: exactly one retry precedes replacement admission.

    assert sender.sent == [flush, flush]

    assert admission.admitted(flush) is True

    assert admission.failures == []


def test_replacement_admission_rejects_invalid_ack_and_fake_clock_timeout() -> None:
    # Given: a pending flush and an acknowledgement for a different session.

    clock = _FakeClock()

    sender = _RecordingFlushSender()

    admission = FlushAdmission(clock=clock, sender=sender)

    flush = _flush()

    admission.begin(flush)

    # When: the invalid acknowledgement arrives and the fake clock reaches 750ms.

    _ = admission.acknowledge(
        FlushAcknowledgement.from_flush(
            StreamFlush(
                stream=StreamKey(session_id="other-session", stream_id="stream-001"),
                turn_id=flush.turn_id,
                segment_id=flush.segment_id,
                cancellation_epoch=flush.cancellation_epoch,
                request_id=flush.request_id,
                target_generated_ssrc=flush.target_generated_ssrc,
            )
        )
    )

    clock.advance(750)

    admission.advance()

    # Then: replacement remains blocked and reports the typed invalid-ack failure.

    assert admission.admitted(flush) is False

    assert admission.failures == [
        FlushFailure(flush=flush, reason="invalid_ack"),
        FlushFailure(flush=flush, reason="timeout"),
    ]


def test_replacement_admission_requires_the_exact_acknowledged_flush_identity() -> None:

    clock = _FakeClock()

    sender = _RecordingFlushSender()

    admission = FlushAdmission(clock=clock, sender=sender)

    acknowledged = _flush()

    replacement = StreamFlush(
        stream=acknowledged.stream,
        turn_id=acknowledged.turn_id,
        segment_id=SegmentId("segment-replacement"),
        cancellation_epoch=acknowledged.cancellation_epoch,
        request_id=FlushRequestId("flush-request-replacement"),
        target_generated_ssrc=acknowledged.target_generated_ssrc,
    )

    admission.begin(acknowledged)

    assert admission.acknowledge(FlushAcknowledgement.from_flush(acknowledged)) is True

    assert admission.admitted(acknowledged) is True

    assert admission.admitted(replacement) is False
