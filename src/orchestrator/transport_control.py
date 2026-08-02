from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

from orchestrator.json_boundary import JsonValue, parse_json_value
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    FlushRequestId,
    GeneratedSsrc,
    SegmentId,
    StreamFlush,
    StreamKey,
    TurnId,
)

if TYPE_CHECKING:
    from orchestrator.config import TrustedLanToken


MAX_SSRC = 4_294_967_295

MAX_UDP_PORT = 65_535

MAX_VOICE_EMBEDDING_DIMENSIONS = 1_024

MAX_ASR_TEXT_LENGTH = 4_000


type ControlEvent = (
    MicInputRegistration
    | SourceRegistration
    | SinkRegistration
    | StreamReady
    | StreamState
    | StreamFlush
    | FlushAcknowledgement
    | VoiceEvidence
    | AsrPartial
    | AsrFinal
)


@dataclass(frozen=True, slots=True)
class ControlEnvelopeError(Exception):
    field_name: str

    @override
    def __str__(self) -> str:
        return f"invalid control envelope: {self.field_name}"


@dataclass(frozen=True, slots=True)
class EnvelopeCorrelation:
    trace_id: str

    session_id: str

    seq: int


@dataclass(frozen=True, slots=True)
class SourceRegistration:
    session_id: str

    stream_id: str

    ssrc: int

    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class MicInputRegistration:
    """Authenticated Mic input identity; it deliberately has no RTP endpoint."""

    session_id: str
    stream_id: str
    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class SinkRegistration:
    session_id: str

    stream_id: str

    udp_port: int

    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class StreamReady:
    session_id: str

    stream_id: str

    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class StreamState:
    session_id: str

    stream_id: str

    state: str

    correlation: EnvelopeCorrelation

    turn_id: TurnId | None = None

    segment_id: SegmentId | None = None

    cancellation_epoch: CancellationEpoch | None = None


@dataclass(frozen=True, slots=True)
class VoiceEvidence:
    """Ephemeral CAM++ evidence; callers must not persist raw embeddings."""

    session_id: str
    stream_id: str
    rtp_start_timestamp: int
    rtp_end_timestamp: int
    embedding_model_revision: str
    embedding: tuple[float, ...]
    speech_ms: int
    quality_score: float
    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class AsrPartial:
    """Diagnostic-only recognition update emitted by the registered Mic."""

    session_id: str
    stream_id: str
    segment_id: str
    rtp_start_timestamp: int
    rtp_end_timestamp: int
    cancellation_epoch: CancellationEpoch
    text: str
    received_at_ms: int
    confidence: float | None
    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class AsrFinal(AsrPartial):
    """A finalized Mic recognition result eligible for the audience gate."""


class ControlReceiver(Protocol):
    def register_control(self, raw_message: str, peer_ip: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AuthenticatedControl:
    receiver: ControlReceiver

    token: TrustedLanToken | None

    def register(
        self, raw_message: str, peer_ip: str, authorization: str | None
    ) -> bool:
        if not bearer_token_matches(self.token, authorization):
            return False

        self.receiver.register_control(raw_message, peer_ip)

        return True


def bearer_token_matches(
    token: TrustedLanToken | None, authorization: str | None
) -> bool:
    if token is None:
        return authorization is None

    prefix = "Bearer "

    if authorization is None or not authorization.startswith(prefix):
        return False

    return hmac.compare_digest(authorization.removeprefix(prefix), token)


def parse_control_event(raw_message: str) -> ControlEvent:  # noqa: C901
    value = parse_json_value(raw_message)

    if not isinstance(value, dict):
        raise ControlEnvelopeError(field_name="$")

    _validate_envelope(value)

    event_type = _text(value, "event_type")

    data = _mapping(value, "data")

    correlation = EnvelopeCorrelation(
        trace_id=_text(value, "trace_id"),
        session_id=_text(value, "session_id"),
        seq=_nonnegative_int(value, "seq"),
    )

    match event_type:
        case "mic.input.register":
            _validate_mic_input_registration(data, _text(value, "source"))
            parsed = MicInputRegistration(
                _text(value, "session_id"), _text(data, "stream_id"), correlation
            )

        # Read-only wire compatibility for an already-deployed Mic. The RTP
        # hub rejects all UDP ingress, so accepting this envelope cannot
        # recreate an audio path.
        case "media.rtp.source.register":
            _validate_source_registration(data, _text(value, "source"))
            parsed = SourceRegistration(
                _text(value, "session_id"),
                _text(data, "stream_id"),
                _ssrc(data),
                correlation,
            )

        case "media.rtp.sink.register":
            _validate_sink_registration(data, _text(value, "source"))

            parsed = SinkRegistration(
                session_id=_text(value, "session_id"),
                stream_id=_text(data, "stream_id"),
                udp_port=_endpoint_port(data),
                correlation=correlation,
            )

        case "media.rtp.sink.ready":
            _validate_sink_ready(data, _text(value, "source"))

            parsed = StreamReady(
                _text(value, "session_id"), _text(data, "stream_id"), correlation
            )

        case "media.rtp.source.ready":
            _validate_source_ready(data, _text(value, "source"))
            parsed = StreamReady(
                _text(value, "session_id"), _text(data, "stream_id"), correlation
            )

        case "media.stream.state":
            _validate_stream_state(data, _text(value, "source"))

            parsed = StreamState(
                session_id=_text(value, "session_id"),
                stream_id=_text(data, "stream_id"),
                state=_stream_state(data),
                correlation=correlation,
                turn_id=_optional_turn_id(value),
                segment_id=_optional_segment_id(value),
                cancellation_epoch=_optional_cancellation_epoch(data),
            )

        case "media.stream.flush":
            _validate_flush(data, _text(value, "source"), "orchestrator")

            parsed = StreamFlush(
                stream=StreamKey(_text(value, "session_id"), _text(data, "stream_id")),
                turn_id=TurnId(_text(value, "turn_id")),
                segment_id=SegmentId(_text(value, "segment_id")),
                cancellation_epoch=CancellationEpoch(
                    _nonnegative_int(data, "cancellation_epoch")
                ),
                request_id=FlushRequestId(_text(data, "request_id")),
                target_generated_ssrc=GeneratedSsrc(
                    _ssrc_field(data, "target_generated_ssrc")
                ),
                correlation=correlation,
            )

        case "media.stream.flush.ack":
            _validate_flush(data, _text(value, "source"), "sound")

            parsed = FlushAcknowledgement(
                stream=StreamKey(_text(value, "session_id"), _text(data, "stream_id")),
                turn_id=TurnId(_text(value, "turn_id")),
                segment_id=SegmentId(_text(value, "segment_id")),
                cancellation_epoch=CancellationEpoch(
                    _nonnegative_int(data, "cancellation_epoch")
                ),
                request_id=FlushRequestId(_text(data, "request_id")),
                target_generated_ssrc=GeneratedSsrc(
                    _ssrc_field(data, "target_generated_ssrc")
                ),
                correlation=correlation,
            )

        case "voice.evidence":
            _validate_voice_evidence(data, _text(value, "source"))
            quality = _mapping(data, "quality")
            parsed = VoiceEvidence(
                session_id=_text(value, "session_id"),
                stream_id=_text(data, "stream_id"),
                rtp_start_timestamp=_nonnegative_int(data, "rtp_start_timestamp"),
                rtp_end_timestamp=_nonnegative_int(data, "rtp_end_timestamp"),
                embedding_model_revision=_text(data, "embedding_model_revision"),
                embedding=tuple(_embedding(data)),
                speech_ms=_nonnegative_int(quality, "speech_ms"),
                quality_score=_quality_score(quality),
                correlation=correlation,
            )

        case "asr.partial" | "asr.final" as asr_event_type:
            asr = _parse_asr_event(data, _text(value, "source"), correlation)
            parsed = (
                AsrPartial(**asr)
                if asr_event_type == "asr.partial"
                else AsrFinal(**asr)
            )

        case _:
            raise ControlEnvelopeError(field_name="event_type")

    return parsed


def _validate_envelope(value: dict[str, JsonValue]) -> None:
    required = (
        "schema_version",
        "event_type",
        "event_id",
        "source",
        "time",
        "trace_id",
        "session_id",
        "seq",
        "data",
    )

    allowed = {*required, "turn_id", "segment_id", "traceparent"}

    if set(value).difference(allowed):
        raise ControlEnvelopeError(field_name="$")

    for field_name in required:
        if field_name not in value:
            raise ControlEnvelopeError(field_name=field_name)

    if value["schema_version"] != "1.0.0":
        raise ControlEnvelopeError(field_name="schema_version")

    for field_name in (
        "event_type",
        "event_id",
        "source",
        "time",
        "trace_id",
        "session_id",
    ):
        _ = _text(value, field_name)

    sequence = value["seq"]

    if type(sequence) is not int or sequence < 0:
        raise ControlEnvelopeError(field_name="seq")

    _ = _mapping(value, "data")


def _validate_mic_input_registration(
    data: dict[str, JsonValue], source: str
) -> None:
    if source != "mic" or set(data) != {"stream_id"}:
        raise ControlEnvelopeError(field_name="data")
    _ = _text(data, "stream_id")


def _validate_source_registration(data: dict[str, JsonValue], source: str) -> None:
    if source != "mic" or set(data) != {"stream_id", "ssrc", "codec", "rtp_endpoint"}:
        raise ControlEnvelopeError(field_name="source")
    _ = _text(data, "stream_id")
    _ = _ssrc(data)
    _ = _endpoint_port(data)


def _validate_sink_registration(data: dict[str, JsonValue], source: str) -> None:
    if source != "sound" or set(data) != {"stream_id", "codec", "rtp_endpoint"}:
        raise ControlEnvelopeError(field_name="source")

    _ = _text(data, "stream_id")

    _ = _endpoint_port(data)


def _validate_source_ready(data: dict[str, JsonValue], source: str) -> None:
    if source != "mic" or set(data) != {"stream_id", "ssrc"}:
        raise ControlEnvelopeError(field_name="source")
    _ = _text(data, "stream_id")
    _ = _ssrc(data)


def _validate_sink_ready(data: dict[str, JsonValue], source: str) -> None:
    if source != "sound" or set(data) != {"stream_id"}:
        raise ControlEnvelopeError(field_name="source")

    _ = _text(data, "stream_id")


def _validate_stream_state(data: dict[str, JsonValue], source: str) -> None:
    if source != "sound":
        raise ControlEnvelopeError(field_name="source")

    _ = _text(data, "stream_id")

    _ = _stream_state(data)

    if set(data).difference({"stream_id", "state", "cancellation_epoch"}):
        raise ControlEnvelopeError(field_name="data")

    _ = _optional_cancellation_epoch(data)


def _optional_turn_id(value: dict[str, JsonValue]) -> TurnId | None:
    turn_id = value.get("turn_id")

    if turn_id is None:
        return None

    if not isinstance(turn_id, str) or turn_id == "":
        raise ControlEnvelopeError(field_name="turn_id")

    return TurnId(turn_id)


def _optional_segment_id(value: dict[str, JsonValue]) -> SegmentId | None:
    segment_id = value.get("segment_id")

    if segment_id is None:
        return None

    if not isinstance(segment_id, str) or segment_id == "":
        raise ControlEnvelopeError(field_name="segment_id")

    return SegmentId(segment_id)


def _optional_cancellation_epoch(
    data: dict[str, JsonValue],
) -> CancellationEpoch | None:
    if "cancellation_epoch" not in data:
        return None

    return CancellationEpoch(_nonnegative_int(data, "cancellation_epoch"))


def _validate_flush(
    data: dict[str, JsonValue], source: str, expected_source: str
) -> None:
    if source != expected_source:
        raise ControlEnvelopeError(field_name="source")

    required = {
        "stream_id",
        "cancellation_epoch",
        "request_id",
        "target_generated_ssrc",
    }

    for field_name in required:
        if field_name not in data:
            raise ControlEnvelopeError(field_name=f"data.{field_name}")

    if set(data) != required:
        raise ControlEnvelopeError(field_name="data")

    _ = _text(data, "stream_id")

    _ = _nonnegative_int(data, "cancellation_epoch")

    _ = _text(data, "request_id")

    _ = _ssrc_field(data, "target_generated_ssrc")


def _validate_voice_evidence(data: dict[str, JsonValue], source: str) -> None:
    required = {
        "stream_id",
        "rtp_start_timestamp",
        "rtp_end_timestamp",
        "embedding_model_revision",
        "embedding",
        "quality",
    }
    if source != "mic" or set(data) != required:
        raise ControlEnvelopeError(field_name="data")
    _ = _text(data, "stream_id")
    start = _nonnegative_int(data, "rtp_start_timestamp")
    if _nonnegative_int(data, "rtp_end_timestamp") < start:
        raise ControlEnvelopeError(field_name="data.rtp_end_timestamp")
    _ = _text(data, "embedding_model_revision")
    _ = _embedding(data)
    quality = _mapping(data, "quality")
    if set(quality) != {"speech_ms", "score"}:
        raise ControlEnvelopeError(field_name="data.quality")
    _ = _nonnegative_int(quality, "speech_ms")
    _ = _quality_score(quality)


def _parse_asr_event(
    data: dict[str, JsonValue], source: str, correlation: EnvelopeCorrelation
) -> dict[str, object]:
    """Parse the wire ASR contract without trusting Mic-provided routing data."""
    required = {
        "stream_id",
        "segment_id",
        "rtp_start_timestamp",
        "rtp_end_timestamp",
        "cancellation_epoch",
        "text",
        "received_at_ms",
    }
    if (
        source != "mic"
        or not required.issubset(data)
        or set(data) - (required | {"confidence"})
    ):
        raise ControlEnvelopeError(field_name="data")
    start = _nonnegative_int(data, "rtp_start_timestamp")
    if _nonnegative_int(data, "rtp_end_timestamp") < start:
        raise ControlEnvelopeError(field_name="data.rtp_end_timestamp")
    text = _text(data, "text")
    if len(text) > MAX_ASR_TEXT_LENGTH:
        raise ControlEnvelopeError(field_name="data.text")
    confidence: float | None = None
    if "confidence" in data:
        value = data["confidence"]
        if not _is_number(value) or not 0 <= _number(value) <= 1:
            raise ControlEnvelopeError(field_name="data.confidence")
        confidence = _number(value)
    return {
        "session_id": correlation.session_id,
        "stream_id": _text(data, "stream_id"),
        "segment_id": _text(data, "segment_id"),
        "rtp_start_timestamp": start,
        "rtp_end_timestamp": _nonnegative_int(data, "rtp_end_timestamp"),
        "cancellation_epoch": CancellationEpoch(
            _nonnegative_int(data, "cancellation_epoch")
        ),
        "text": text,
        "received_at_ms": _nonnegative_int(data, "received_at_ms"),
        "confidence": confidence,
        "correlation": correlation,
    }


def _embedding(data: dict[str, JsonValue]) -> tuple[float, ...]:
    value = data.get("embedding")
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= MAX_VOICE_EMBEDDING_DIMENSIONS
        or any(not _is_number(item) for item in value)
    ):
        raise ControlEnvelopeError(field_name="data.embedding")
    return tuple(_number(item) for item in value)


def _quality_score(quality: dict[str, JsonValue]) -> float:
    value = quality.get("score")
    if not _is_number(value) or not 0 <= _number(value) <= 1:
        raise ControlEnvelopeError(field_name="data.quality.score")
    return _number(value)


def _is_number(value: JsonValue | None) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _number(value: JsonValue | None) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ControlEnvelopeError(field_name="number")
    return float(value)


def _ssrc(data: dict[str, JsonValue]) -> int:
    return _ssrc_field(data, "ssrc")


def _ssrc_field(data: dict[str, JsonValue], field_name: str) -> int:
    value = data.get(field_name)

    if type(value) is not int or not 0 <= value <= MAX_SSRC:
        raise ControlEnvelopeError(field_name=f"data.{field_name}")

    return value


def _nonnegative_int(data: dict[str, JsonValue], field_name: str) -> int:
    value = data.get(field_name)

    if type(value) is not int or value < 0:
        raise ControlEnvelopeError(field_name=f"data.{field_name}")

    return value


def _endpoint_port(data: dict[str, JsonValue]) -> int:
    _validate_codec(data.get("codec"))

    endpoint = data.get("rtp_endpoint")

    if not isinstance(endpoint, dict) or set(endpoint) != {"host", "port"}:
        raise ControlEnvelopeError(field_name="data.rtp_endpoint")

    _ = _text(endpoint, "host")

    port = endpoint["port"]

    if type(port) is not int or not 1 <= port <= MAX_UDP_PORT:
        raise ControlEnvelopeError(field_name="data.rtp_endpoint.port")

    return port


def _validate_codec(value: JsonValue | None) -> None:
    expected: dict[str, JsonValue] = {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }

    if not isinstance(value, dict) or value != expected:
        raise ControlEnvelopeError(field_name="data.codec")


def _stream_state(data: dict[str, JsonValue]) -> str:
    value = _text(data, "state")

    if value not in {"queued", "playing", "paused", "finished", "cancelled", "error"}:
        raise ControlEnvelopeError(field_name="data.state")

    return value


def _mapping(value: dict[str, JsonValue], field_name: str) -> dict[str, JsonValue]:
    nested = value.get(field_name)

    if not isinstance(nested, dict):
        raise ControlEnvelopeError(field_name=field_name)

    return nested


def _text(value: dict[str, JsonValue], field_name: str) -> str:
    text = value.get(field_name)

    if not isinstance(text, str) or text.strip() == "":
        raise ControlEnvelopeError(field_name=field_name)

    return text
