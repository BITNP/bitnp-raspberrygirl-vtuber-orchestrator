
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from orchestrator.streaming_contracts import (
    CancellationEpoch,
    SegmentId,
    StreamKey,
    TurnId,
)
from orchestrator.transport_hub import L16_FRAME_BYTES, RTP_HEADER_BYTES

_FRAME_SAMPLES: Final = 320

_PRE_ROLL_FRAMES: Final = 10

_SILENCE_FRAMES: Final = 30

_FORCED_FRAMES: Final = 750

_SPEECH_ENERGY: Final = 400

_SEQUENCE_HALF_RANGE: Final = 32_768


class EndpointReason(StrEnum):

    SILENCE = "silence"

    FORCED = "forced"

    GAP = "gap"

    DISCONNECT = "disconnect"


@dataclass(frozen=True, slots=True)
class PartialUtterance:

    stream: StreamKey

    payload: bytes

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch


@dataclass(frozen=True, slots=True)
class EndpointedUtterance:

    stream: StreamKey

    payload: bytes

    reason: EndpointReason

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch


type EndpointEvent = PartialUtterance | EndpointedUtterance


@dataclass(slots=True)
class StreamEndpointer:

    stream: StreamKey

    _pre_roll: deque[bytes] = field(
        default_factory=lambda: deque(maxlen=_PRE_ROLL_FRAMES)
    )

    _utterance: list[bytes] = field(default_factory=list)

    _pending: dict[int, bytes] = field(default_factory=dict)

    _expected_sequence: int | None = None

    _expected_timestamp: int | None = None

    _silence_frames: int = 0

    _turn_index: int = 0

    _segment_index: int = 0

    _epoch: int = 0

    def push(self, packet: bytes) -> tuple[EndpointEvent, ...]:
        sequence = int.from_bytes(packet[2:4], "big")

        if self._expected_sequence is None:
            self._expected_sequence = sequence

        expected = self._expected_sequence

        events: list[EndpointEvent]

        if sequence == expected:
            events = self._accept_ordered(packet)

            events.extend(self._drain_pending())

            return tuple(events)

        if sequence == (expected + 1) % 65_536:
            _ = self._pending.setdefault(sequence, packet)

            return ()

        if sequence in self._pending or self._is_stale(sequence, expected):
            return ()

        events = []

        events.extend(self._finalize(EndpointReason.GAP))

        self._reset_ordering(sequence)

        events.extend(self._accept_ordered(packet))

        return tuple(events)

    def disconnect(self) -> EndpointedUtterance | None:
        events = self._finalize(EndpointReason.DISCONNECT)

        self._pre_roll.clear()

        self._pending.clear()

        self._expected_sequence = None

        self._expected_timestamp = None

        return events[0] if events else None

    def _accept_ordered(self, packet: bytes) -> list[EndpointEvent]:
        timestamp = int.from_bytes(packet[4:8], "big")

        if (
            self._expected_timestamp is not None
            and timestamp != self._expected_timestamp
        ):
            events: list[EndpointEvent] = []

            events.extend(self._finalize(EndpointReason.GAP))

        else:
            events = []

        self._expected_sequence = (int.from_bytes(packet[2:4], "big") + 1) % 65_536

        self._expected_timestamp = (timestamp + _FRAME_SAMPLES) % (2**32)

        payload = packet[RTP_HEADER_BYTES:]

        if self._is_speech(payload):
            if not self._utterance:
                self._utterance.extend(self._pre_roll)

            self._utterance.append(payload)

            self._silence_frames = 0

            events.append(
                PartialUtterance(
                    self.stream,
                    b"".join(self._utterance),
                    TurnId(
                        ":".join(
                            (
                                self.stream.session_id,
                                self.stream.stream_id,
                                "turn",
                                str(self._turn_index + 1),
                            )
                        )
                    ),
                    SegmentId(
                        ":".join(
                            (
                                self.stream.session_id,
                                self.stream.stream_id,
                                "segment",
                                str(self._segment_index + 1),
                            )
                        )
                    ),
                    CancellationEpoch(self._epoch),
                )
            )

            if len(self._utterance) >= _FORCED_FRAMES:
                events.extend(self._finalize(EndpointReason.FORCED))

        elif self._utterance:
            self._utterance.append(payload)

            self._silence_frames += 1

            if self._silence_frames == _SILENCE_FRAMES:
                events.extend(self._finalize(EndpointReason.SILENCE))

        else:
            self._pre_roll.append(payload)

        return events

    def _drain_pending(self) -> list[EndpointEvent]:
        events: list[EndpointEvent] = []

        while self._expected_sequence in self._pending:
            packet = self._pending.pop(self._expected_sequence)

            events.extend(self._accept_ordered(packet))

        return events

    def _finalize(self, reason: EndpointReason) -> list[EndpointedUtterance]:
        if not self._utterance:
            return []

        self._turn_index += 1

        self._segment_index += 1

        event = EndpointedUtterance(
            stream=self.stream,
            payload=b"".join(self._utterance),
            reason=reason,
            turn_id=TurnId(
                f"{self.stream.session_id}:{self.stream.stream_id}:turn:{self._turn_index}"
            ),
            segment_id=SegmentId(
                f"{self.stream.session_id}:{self.stream.stream_id}:segment:{self._segment_index}"
            ),
            cancellation_epoch=CancellationEpoch(self._epoch),
        )

        self._utterance.clear()

        self._silence_frames = 0

        self._pre_roll.clear()

        return [event]

    def _reset_ordering(self, sequence: int) -> None:
        self._pending.clear()

        self._expected_sequence = sequence

        self._expected_timestamp = None

    def _is_stale(self, sequence: int, expected: int) -> bool:
        return (expected - sequence) % 65_536 <= _SEQUENCE_HALF_RANGE

    def _is_speech(self, payload: bytes) -> bool:
        # A single peak is not speech: normal microphone noise and keyboard
        # taps can exceed a sample threshold.  Gate on a whole 20 ms frame's
        # mean absolute amplitude so real trailing silence reaches the 600 ms
        # endpoint instead of being forced into fifteen-second ASR batches.
        total_amplitude = sum(
            abs(int.from_bytes(payload[offset : offset + 2], "big", signed=True))
            for offset in range(0, L16_FRAME_BYTES, 2)
        )

        return total_amplitude >= _SPEECH_ENERGY * _FRAME_SAMPLES
