"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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

    """函数契约说明.

    功能: 验证 endpointer emits preroll
    partial and silence final
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 endpointer forces a final
    after fifteen seconds of speech
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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


def test_endpointer_drops_duplicate_and_accepts_one_frame_reorder() -> None:
    # Given: a stream with a one-frame reordering window.

    """函数契约说明.

    功能: 验证 endpointer drops duplicate
    and accepts one frame reorder
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 endpointer finalizes on gap
    and accepts timestamp wrap
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 endpointer keeps turn segment
    and epoch namespaces per stream
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 endpointer disconnect
    finalizes and clears state
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    actor = StreamEndpointer(StreamKey("session-a", "stream-a"))

    _ = actor.push(_packet(1, 320, 1_000))

    # When: the owning route disconnects.

    final = actor.disconnect()

    # Then: its pending speech is finalized once and retained state is gone.

    assert final is not None

    assert final.reason is EndpointReason.DISCONNECT

    assert actor.disconnect() is None


def _silence_final(actor: StreamEndpointer) -> EndpointedUtterance:
    """函数契约说明.

    功能: 执行 _silence_final 的同步逻辑,并协调
    list, extend, _last_final, push。
    参数: actor: StreamEndpointer。 必填。
    契约: 同步调用。 返回 `EndpointedUtterance`。
    """

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
    """函数契约说明.

    功能: 执行 _last_partial 的同步逻辑,并协调 next,
    reversed, isinstance。
    参数: events:
    Sequence[PartialUtterance |
    EndpointedUtterance]。 必填。
    契约: 同步调用。 返回 `PartialUtterance`。
    """

    return next(
        event for event in reversed(events) if isinstance(event, PartialUtterance)
    )


def _last_final(
    events: Sequence[PartialUtterance | EndpointedUtterance],
) -> EndpointedUtterance:
    """函数契约说明.

    功能: 执行 _last_final 的同步逻辑,并协调 next,
    reversed, isinstance。
    参数: events:
    Sequence[PartialUtterance |
    EndpointedUtterance]。 必填。
    契约: 同步调用。 返回 `EndpointedUtterance`。
    """

    return next(
        event for event in reversed(events) if isinstance(event, EndpointedUtterance)
    )


def _packet(sequence: int, timestamp: int, sample: int) -> bytes:
    """函数契约说明.

    功能: 执行 _packet 的同步逻辑,并协调 to_bytes。
    参数: sequence: int。 必填。 timestamp:
    int。 必填。 sample: int。 必填。
    契约: 同步调用。 返回 `bytes`。
    """

    payload = sample.to_bytes(2, "big", signed=True) * 320

    return (
        b"\x80\x60"
        + sequence.to_bytes(2, "big")
        + timestamp.to_bytes(4, "big")
        + b"\x12\x34\x56\x78"
        + payload
    )
