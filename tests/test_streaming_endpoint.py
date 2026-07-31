
from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.streaming_contracts import StreamKey
from orchestrator.streaming_endpoint import (
    EndpointedUtterance,
    EndpointReason,
    PartialUtterance,
    StreamEndpointer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_endpointer_emits_preroll_partial_and_silence_final() -> None:
    # Given: ten quiet frames before one voiced frame on one canonical RTP stream.


    actor = StreamEndpointer(StreamKey("session-a", "stream-a"))

    quiet = [_packet(sequence, sequence * 320, 0) for sequence in range(1, 11)]

    # When: speech begins then exactly 600 ms of quiet audio follows.

    events = [event for frame in quiet for event in actor.push(frame)]

    events.extend(actor.push(_packet(11, 11 * 320, 1_000)))

    events.extend(
        event
        for sequence in range(12, 42)
        for event in actor.push(_packet(sequence, sequence * 320, 0))
    )

    # Then: the partial and final retain precisely 200 ms of pre-roll.

    partial = _last_partial(events)

    final = _last_final(events)

    assert partial.payload == b"\x00\x00" * (320 * 10) + b"\x03\xe8" * 320

    assert final.reason is EndpointReason.SILENCE

    assert final.payload == (
        b"\x00\x00" * (320 * 10) + b"\x03\xe8" * 320 + b"\x00\x00" * (320 * 30)
    )


def test_endpointer_forces_a_final_after_fifteen_seconds_of_speech() -> None:
    # Given: an active stream receiving continuous voiced canonical frames.


    actor = StreamEndpointer(StreamKey("session-a", "stream-a"))

    # When: 15 seconds of speech are accepted.

    events = [
        event
        for sequence in range(1, 751)
        for event in actor.push(_packet(sequence, sequence * 320, 1_000))
    ]

    # Then: the final is bounded and carries the forced endpoint reason.

    final = _last_final(events)

    assert final.reason is EndpointReason.FORCED

    assert len(final.payload) == 640 * 750


def test_endpointer_rejects_an_isolated_noise_peak_as_speech() -> None:
    # Given: a quiet canonical frame containing one loud click.


    actor = StreamEndpointer(StreamKey("session-a", "stream-a"))
    payload = bytearray(640)
    payload[:2] = (10_000).to_bytes(2, "big", signed=True)
    packet = b"\x80\x60" + (1).to_bytes(2, "big") + (320).to_bytes(4, "big")
    packet += b"\x12\x34\x56\x78" + bytes(payload)

    # When: the click reaches the endpointer.


    events = actor.push(packet)

    # Then: it remains pre-roll rather than opening an endless speech turn.


    assert events == ()


def test_endpointer_drops_duplicate_and_accepts_one_frame_reorder() -> None:
    # Given: a stream with a one-frame reordering window.


    actor = StreamEndpointer(StreamKey("session-a", "stream-a"))

    # When: frame three arrives before frame two and frame two is repeated.

    events = list(actor.push(_packet(1, 320, 1_000)))

    events.extend(actor.push(_packet(3, 960, 3_000)))

    events.extend(actor.push(_packet(2, 640, 2_000)))

    events.extend(actor.push(_packet(2, 640, 2_000)))

    # Then: only the ordered three-frame partial is observed.

    assert len(events) == 3

    assert events[-1].payload == (
        b"\x03\xe8" * 320 + b"\x07\xd0" * 320 + b"\x0b\xb8" * 320
    )


def test_endpointer_finalizes_on_gap_and_accepts_timestamp_wrap() -> None:
    # Given: an active stream just before the unsigned RTP timestamp wraps.


    actor = StreamEndpointer(StreamKey("session-a", "stream-a"))

    # When: a continuous wrap is followed by a two-frame sequence gap.

    events = list(actor.push(_packet(65_534, 4_294_966_976, 1_000)))

    events.extend(actor.push(_packet(65_535, 0, 1_000)))

    events.extend(actor.push(_packet(0, 320, 1_000)))

    events.extend(actor.push(_packet(1, 640, 1_000)))

    events.extend(actor.push(_packet(4, 1_600, 1_000)))

    # Then: wrap is continuous, while the gap finalizes the prior utterance.

    final = _last_final(events)

    assert final.reason is EndpointReason.GAP

    assert final.payload == b"\x03\xe8" * (320 * 4)


def test_endpointer_keeps_turn_segment_and_epoch_namespaces_per_stream() -> None:
    # Given: two concurrent route identities.


    first = StreamEndpointer(StreamKey("session-a", "stream-a"))

    second = StreamEndpointer(StreamKey("session-b", "stream-b"))

    # When: each stream reaches a silence endpoint.

    first_final = _silence_final(first)

    second_final = _silence_final(second)

    # Then: no bytes or turn namespace crosses stream boundaries.

    assert first_final.payload == b"\x03\xe8" * 320 + b"\x00\x00" * (320 * 30)

    assert second_final.payload == b"\x07\xd0" * 320 + b"\x00\x00" * (320 * 30)

    assert first_final.turn_id != second_final.turn_id

    assert first_final.segment_id != second_final.segment_id

    assert first_final.cancellation_epoch == second_final.cancellation_epoch == 0


def test_endpointer_disconnect_finalizes_and_clears_state() -> None:
    # Given: an active stream whose route is about to disconnect.


    actor = StreamEndpointer(StreamKey("session-a", "stream-a"))

    _ = actor.push(_packet(1, 320, 1_000))

    # When: the owning route disconnects.

    final = actor.disconnect()

    # Then: its pending speech is finalized once and retained state is gone.

    assert final is not None

    assert final.reason is EndpointReason.DISCONNECT

    assert actor.disconnect() is None


def _silence_final(actor: StreamEndpointer) -> EndpointedUtterance:

    sample = 1_000 if actor.stream.stream_id == "stream-a" else 2_000

    events = list(actor.push(_packet(1, 320, sample)))

    events.extend(
        event
        for sequence in range(2, 32)
        for event in actor.push(_packet(sequence, sequence * 320, 0))
    )

    return _last_final(events)


def _last_partial(
    events: Sequence[PartialUtterance | EndpointedUtterance],
) -> PartialUtterance:

    return next(
        event for event in reversed(events) if isinstance(event, PartialUtterance)
    )


def _last_final(
    events: Sequence[PartialUtterance | EndpointedUtterance],
) -> EndpointedUtterance:

    return next(
        event for event in reversed(events) if isinstance(event, EndpointedUtterance)
    )


def _packet(sequence: int, timestamp: int, sample: int) -> bytes:

    payload = sample.to_bytes(2, "big", signed=True) * 320

    return (
        b"\x80\x60"
        + sequence.to_bytes(2, "big")
        + timestamp.to_bytes(4, "big")
        + b"\x12\x34\x56\x78"
        + payload
    )
