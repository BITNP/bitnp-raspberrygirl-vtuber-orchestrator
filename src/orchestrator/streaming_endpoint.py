"""模块契约说明.

职责: 提供 orchestrator.streaming_endpoint
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 定义 EndpointReason 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    SILENCE = "silence"

    FORCED = "forced"

    GAP = "gap"

    DISCONNECT = "disconnect"


@dataclass(frozen=True, slots=True)
class PartialUtterance:
    """类契约说明.

    职责: 保存 PartialUtterance
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、payload、turn_id、segme
    nt_id、cancellation_epoch。
    """

    stream: StreamKey

    payload: bytes

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch


@dataclass(frozen=True, slots=True)
class EndpointedUtterance:
    """类契约说明.

    职责: 保存 EndpointedUtterance
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、payload、reason、turn_i
    d、segment_id、cancellation_epoch。
    """

    stream: StreamKey

    payload: bytes

    reason: EndpointReason

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch


type EndpointEvent = PartialUtterance | EndpointedUtterance


@dataclass(slots=True)
class StreamEndpointer:
    """类契约说明.

    职责: 保存 StreamEndpointer
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、_pre_roll、_utterance、
    _pending、_expected_sequence、_expecte
    d_timestamp。 方法: push、disconnect、_ac
    cept_ordered、_drain_pending、_finaliz
    e、_reset_ordering。
    """

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
        """函数契约说明.

        功能: 执行 push 的同步逻辑,并协调
        from_bytes, extend,
        _reset_ordering, tuple。
        参数: self 表示当前实例。 packet: bytes。
        必填。
        契约: 同步调用。 返回
        `tuple[EndpointEvent, ...]`。
        """
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
        """函数契约说明.

        功能: 执行 disconnect 的同步逻辑,并协调
        _finalize, clear。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `EndpointedUtterance | None`。
        """
        events = self._finalize(EndpointReason.DISCONNECT)

        self._pre_roll.clear()

        self._pending.clear()

        self._expected_sequence = None

        self._expected_timestamp = None

        return events[0] if events else None

    def _accept_ordered(self, packet: bytes) -> list[EndpointEvent]:
        """函数契约说明.

        功能: 执行 _accept_ordered 的同步逻辑,并协调
        from_bytes, _is_speech, extend,
        append。
        参数: self 表示当前实例。 packet: bytes。
        必填。
        契约: 同步调用。 返回
        `list[EndpointEvent]`。
        """
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
        """函数契约说明.

        功能: 执行 _drain_pending 的同步逻辑,并协调
        pop, extend, _accept_ordered。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `list[EndpointEvent]`。
        """
        events: list[EndpointEvent] = []

        while self._expected_sequence in self._pending:
            packet = self._pending.pop(self._expected_sequence)

            events.extend(self._accept_ordered(packet))

        return events

    def _finalize(self, reason: EndpointReason) -> list[EndpointedUtterance]:
        """函数契约说明.

        功能: 执行 _finalize 的同步逻辑,并协调
        EndpointedUtterance, clear,
        join, TurnId。
        参数: self 表示当前实例。 reason:
        EndpointReason。 必填。
        契约: 同步调用。 返回
        `list[EndpointedUtterance]`。
        """
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
        """函数契约说明.

        功能: 执行 _reset_ordering 的同步逻辑,并协调
        clear。
        参数: self 表示当前实例。 sequence: int。
        必填。
        契约: 同步调用。 返回 `None`。
        """
        self._pending.clear()

        self._expected_sequence = sequence

        self._expected_timestamp = None

    def _is_stale(self, sequence: int, expected: int) -> bool:
        """函数契约说明.

        功能: 执行 _is_stale 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 sequence: int。
        必填。 expected: int。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        return (expected - sequence) % 65_536 <= _SEQUENCE_HALF_RANGE

    def _is_speech(self, payload: bytes) -> bool:
        """函数契约说明.

        功能: 执行 _is_speech 的同步逻辑,并协调
        range, any, abs, from_bytes。
        参数: self 表示当前实例。 payload: bytes。
        必填。
        契约: 同步调用。 返回 `bool`。
        """
        samples = range(0, L16_FRAME_BYTES, 2)

        return any(
            abs(int.from_bytes(payload[offset : offset + 2], "big", signed=True))
            >= _SPEECH_ENERGY
            for offset in samples
        )
