from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.config import load_fake_config
from orchestrator.ids import SegmentId as PipelineSegmentId
from orchestrator.ids import TurnId as PipelineTurnId
from orchestrator.observability import (
    OnsiteObservability,
    StageCorrelation,
    StageRecord,
)
from orchestrator.onsite_stream_actor import OnsiteStreamActor
from orchestrator.pipeline_contracts import ASRAudienceEvent, TurnResult
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    SegmentId,
    StreamKey,
    TurnId,
)
from orchestrator.streaming_endpoint import EndpointedUtterance, EndpointReason
from orchestrator.transport_control import EnvelopeCorrelation
from orchestrator.tts_rtp import Pcm16leChunk

if TYPE_CHECKING:
    from orchestrator.llm import CancellationToken


@dataclass(slots=True)
class _Stages:
    outputs: list[bytes] = field(default_factory=list)

    def transcribe(
        self, endpoint: EndpointedUtterance, cancellation: CancellationToken
    ) -> ASRAudienceEvent:
        _ = cancellation
        return ASRAudienceEvent("question", 0, str(endpoint.segment_id), 1)

    def answer(
        self, event: ASRAudienceEvent, cancellation: CancellationToken
    ) -> TurnResult:
        _ = (event, cancellation)
        return TurnResult(
            PipelineTurnId("turn-17"),
            PipelineSegmentId("segment-23"),
            "answer",
            used_fallback=False,
        )

    def synthesize(
        self, turn: TurnResult, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...]:
        _ = (turn, cancellation)
        return (Pcm16leChunk(b"\x10\x20" * 320),)

    def complete(self, turn: TurnResult, chunks: tuple[Pcm16leChunk, ...]) -> None:
        _ = (turn, chunks)

    async def output(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        _ = (stream, epoch)
        self.outputs.append(packet)


def test_fake_turn_emits_correlation_complete_stage_records() -> None:
    asyncio.run(_correlation_proof())


async def _correlation_proof() -> None:
    # Given: one fake turn with its ingress trace bound to the stream.
    stream = StreamKey("session-17", "stream-23")
    observability = OnsiteObservability(load_fake_config())
    observability.bind_correlation(
        stream,
        EnvelopeCorrelation(trace_id="trace-19", session_id="session-17", seq=23),
    )
    actor = OnsiteStreamActor(
        stream,
        CancellationEpoch(5),
        _Stages(),
        observability=observability,
    )
    endpoint = EndpointedUtterance(
        stream=stream,
        payload=b"\x01\x02" * 320,
        reason=EndpointReason.FORCED,
        turn_id=TurnId("turn-17"),
        segment_id=SegmentId("segment-23"),
        cancellation_epoch=CancellationEpoch(5),
    )

    # When: the endpoint is processed through ASR, answer, TTS, and RTP output.
    actor.submit(endpoint, CancellationEpoch(5))
    await actor.wait_quiescent()

    # Then: every stage is trace/session/turn/segment/epoch correlated.
    assert {record.stage for record in observability.records} >= {
        "endpoint",
        "asr_final",
        "answer",
        "tts",
        "rtp_egress",
    }
    assert all(record.trace_id == "trace-19" for record in observability.records)
    assert all(record.session_id == "session-17" for record in observability.records)
    assert all(record.turn_id == "turn-17" for record in observability.records)
    assert all(record.segment_id == "segment-23" for record in observability.records)
    assert all(record.cancellation_epoch == 5 for record in observability.records)
    assert all(record.seq == 23 for record in observability.records)
    assert {record.stage for record in observability.journal.records} >= {
        "endpoint",
        "asr_final",
        "answer",
        "tts",
        "rtp_egress",
    }
    assert "trace-19" not in repr(observability.journal.records)
    assert "session-17" not in repr(observability.journal.records)
    assert "turn-17" not in repr(observability.journal.records)
    assert "segment-23" not in repr(observability.journal.records)


def test_transport_stages_use_bound_envelope_and_real_command_correlation() -> None:
    # Given: a source registration correlation and a real Sound playback command.
    stream = StreamKey("session-17", "stream-23")
    observability = OnsiteObservability(load_fake_config())
    observability.bind_correlation(
        stream,
        EnvelopeCorrelation(trace_id="trace-19", session_id="session-17", seq=23),
    )

    # When: ingress and playback state transitions are recorded at their boundaries.
    observability.record_stream("rtp_ingress", stream)
    observability.record_stream(
        "playback_state",
        stream,
        command=StageCorrelation(
            trace_id="trace-19",
            session_id="session-17",
            seq=23,
            turn_id="turn-17",
            segment_id="segment-23",
            cancellation_epoch=5,
        ),
    )

    # Then: no transport-derived identifiers replace the received correlation.
    assert observability.records == [
        StageRecord(
            stage="rtp_ingress",
            trace_id="trace-19",
            session_id="session-17",
            seq=23,
            turn_id=None,
            segment_id=None,
            cancellation_epoch=None,
        ),
        StageRecord(
            stage="playback_state",
            trace_id="trace-19",
            session_id="session-17",
            seq=23,
            turn_id="turn-17",
            segment_id="segment-23",
            cancellation_epoch=5,
        ),
    ]
