"""Canonical control-envelope parsing for the Orchestrator media transport."""

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

type ControlEvent = (
    SourceRegistration
    | SinkRegistration
    | StreamReady
    | StreamState
    | StreamFlush
    | FlushAcknowledgement
)


@dataclass(frozen=True, slots=True)
class ControlEnvelopeError(Exception):
    """Raised when a control message violates the canonical envelope contract."""

    field_name: str

    @override
    def __str__(self) -> str:
        return f"invalid control envelope: {self.field_name}"


@dataclass(frozen=True, slots=True)
class EnvelopeCorrelation:
    """Authenticated envelope identity retained across control and media stages."""

    trace_id: str
    session_id: str
    seq: int


@dataclass(frozen=True, slots=True)
class SourceRegistration:
    """Canonical Mic source registration parsed at the WSS boundary."""

    session_id: str
    stream_id: str
    ssrc: int
    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class SinkRegistration:
    """Canonical Sound sink registration parsed at the WSS boundary."""

    session_id: str
    stream_id: str
    udp_port: int
    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class StreamReady:
    """Canonical Mic or Sound stream readiness parsed at the WSS boundary."""

    session_id: str
    stream_id: str
    correlation: EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class StreamState:
    """Canonical Sound stream state parsed at the WSS boundary."""

    session_id: str
    stream_id: str
    state: str
    correlation: EnvelopeCorrelation
    turn_id: TurnId | None = None
    segment_id: SegmentId | None = None
    cancellation_epoch: CancellationEpoch | None = None


class ControlReceiver(Protocol):
    """Accepts trusted canonical control messages from one authenticated peer."""

    def register_control(self, raw_message: str, peer_ip: str) -> None:
        """Parse and apply one authenticated control message."""


@dataclass(frozen=True, slots=True)
class AuthenticatedControl:
    """Bearer-token gate applied before a control message reaches the hub."""

    receiver: ControlReceiver
    token: TrustedLanToken | None

    def register(
        self, raw_message: str, peer_ip: str, authorization: str | None
    ) -> bool:
        """Forward a control envelope only when its bearer token is accepted."""
        if not bearer_token_matches(self.token, authorization):
            return False
        self.receiver.register_control(raw_message, peer_ip)
        return True


def bearer_token_matches(
    token: TrustedLanToken | None, authorization: str | None
) -> bool:
    """Return whether the header has the configured Bearer token."""
    if token is None:
        return authorization is None
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        return False
    return hmac.compare_digest(authorization.removeprefix(prefix), token)


def parse_control_event(raw_message: str) -> ControlEvent:
    """Parse one canonical RTP control envelope from the untrusted WSS boundary."""
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
        case "media.rtp.source.register":
            _validate_source_registration(data, _text(value, "source"))
            parsed: ControlEvent = SourceRegistration(
                session_id=_text(value, "session_id"),
                stream_id=_text(data, "stream_id"),
                ssrc=_ssrc(data),
                correlation=correlation,
            )
        case "media.rtp.sink.register":
            _validate_sink_registration(data, _text(value, "source"))
            parsed = SinkRegistration(
                session_id=_text(value, "session_id"),
                stream_id=_text(data, "stream_id"),
                udp_port=_endpoint_port(data),
                correlation=correlation,
            )
        case "media.rtp.source.ready":
            _validate_source_ready(data, _text(value, "source"))
            parsed = StreamReady(
                _text(value, "session_id"), _text(data, "stream_id"), correlation
            )
        case "media.rtp.sink.ready":
            _validate_sink_ready(data, _text(value, "source"))
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
