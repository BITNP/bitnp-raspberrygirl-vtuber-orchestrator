"""模块契约说明.

职责: 提供 orchestrator.scheduler_reflex
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from orchestrator.ids import TraceId
from orchestrator.sessions import (
    EventCorrelation,
    EventSequence,
    SchedulerEvent,
    SessionScheduler,
    StartTurn,
    TransitionAccepted,
)
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
from orchestrator.tts_rtp import generated_ssrc

if TYPE_CHECKING:
    from orchestrator.transport_control import EnvelopeCorrelation


@dataclass(frozen=True, slots=True)
class OutputLease:
    """类契约说明.

    职责: 保存 OutputLease
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、turn_id、segment_id、ca
    ncellation_epoch、generation、target_g
    enerated_ssrc。
    """

    stream: StreamKey

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch

    generation: int

    target_generated_ssrc: GeneratedSsrc


@dataclass(frozen=True, slots=True)
class _PendingReplacement:
    """类契约说明.

    职责: 保存 _PendingReplacement
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: lease、flush。
    """

    lease: OutputLease

    flush: StreamFlush


@dataclass(frozen=True, slots=True)
class SchedulerReflexRejectionError(Exception):
    """类契约说明.

    职责: 保存 SchedulerReflexRejectionError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """


@final
class SchedulerOutputFence:
    """类契约说明.

    职责: 定义 SchedulerOutputFence
    的状态、行为和对外协作边界。
    契约: 方法: __init__、activate、interrupt、
    acknowledge、can_emit、_start_turn。
    """

    def __init__(self, scheduler: SessionScheduler) -> None:
        """函数契约说明.

        功能: 初始化 SchedulerOutputFence
        的字段并建立实例不变式。
        参数: self 表示当前实例。 scheduler:
        SessionScheduler。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._scheduler = scheduler

        self._leases: dict[StreamKey, OutputLease] = {}

        self._pending: dict[StreamKey, _PendingReplacement] = {}

        self._flush_sequence = 0

    def activate(
        self,
        *,
        stream: StreamKey,
        segment_id: SegmentId,
        target_generated_ssrc: GeneratedSsrc,
        correlation: EnvelopeCorrelation,
    ) -> OutputLease:
        """函数契约说明.

        功能: 执行 activate 的同步逻辑,并协调 get,
        CancellationEpoch, OutputLease,
        pop。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 segment_id:
        SegmentId。 必填。
        target_generated_ssrc:
        GeneratedSsrc。 必填。 correlation:
        EnvelopeCorrelation。 必填。
        契约: 同步调用。 返回 `OutputLease`。
        """
        previous = self._leases.get(stream)

        generation = 0 if previous is None else previous.generation + 1

        epoch = CancellationEpoch(
            0 if previous is None else int(previous.cancellation_epoch) + 1
        )

        lease = OutputLease(
            stream=stream,
            turn_id=self._start_turn(correlation),
            segment_id=segment_id,
            cancellation_epoch=epoch,
            generation=generation,
            target_generated_ssrc=target_generated_ssrc,
        )

        self._leases[stream] = lease

        _ = self._pending.pop(stream, None)

        return lease

    def interrupt(
        self,
        *,
        stream: StreamKey,
        segment_id: SegmentId,
        correlation: EnvelopeCorrelation,
    ) -> tuple[OutputLease, StreamFlush]:
        """函数契约说明.

        功能: 执行 interrupt 的同步逻辑,并协调
        activate, StreamFlush,
        _PendingReplacement,
        GeneratedSsrc。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 segment_id:
        SegmentId。 必填。 correlation:
        EnvelopeCorrelation。 必填。
        契约: 同步调用。 返回 `tuple[OutputLease,
        StreamFlush]`。
        """
        active = self._leases[stream]

        replacement = self.activate(
            stream=stream,
            segment_id=segment_id,
            target_generated_ssrc=GeneratedSsrc(
                generated_ssrc(
                    stream, CancellationEpoch(int(active.cancellation_epoch) + 1)
                )
            ),
            correlation=correlation,
        )

        self._flush_sequence += 1

        flush = StreamFlush(
            stream=stream,
            turn_id=active.turn_id,
            segment_id=active.segment_id,
            cancellation_epoch=replacement.cancellation_epoch,
            request_id=FlushRequestId(
                f"{stream.session_id}:{stream.stream_id}:flush:{self._flush_sequence}"
            ),
            target_generated_ssrc=active.target_generated_ssrc,
            correlation=correlation,
        )

        self._pending[stream] = _PendingReplacement(replacement, flush)

        return replacement, flush

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> bool:
        """函数契约说明.

        功能: 执行 acknowledge 的同步逻辑,并协调
        get, from_flush。
        参数: self 表示当前实例。
        acknowledgement:
        FlushAcknowledgement。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        pending = self._pending.get(acknowledgement.stream)

        if pending is None or acknowledgement != FlushAcknowledgement.from_flush(
            pending.flush
        ):
            return False

        del self._pending[acknowledgement.stream]

        return True

    def can_emit(self, stream: StreamKey, epoch: CancellationEpoch) -> bool:
        """函数契约说明.

        功能: 执行 can_emit 的同步逻辑,并协调 get,
        str。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 epoch:
        CancellationEpoch。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        lease = self._leases.get(stream)

        return (
            lease is not None
            and stream not in self._pending
            and epoch == lease.cancellation_epoch
            and str(lease.turn_id) == str(self._scheduler.snapshot.active_turn_id)
        )

    def _start_turn(self, correlation: EnvelopeCorrelation) -> TurnId:
        """函数契约说明.

        功能: 执行 _start_turn 的同步逻辑,并协调
        apply, TurnId, StartTurn,
        isinstance。
        参数: self 表示当前实例。 correlation:
        EnvelopeCorrelation。 必填。
        契约: 同步调用。 返回 `TurnId`。
        """
        result = self._scheduler.apply(
            StartTurn(
                expected_revision=self._scheduler.snapshot.revision,
                event=SchedulerEvent(
                    event_type="asr.final",
                    correlation=EventCorrelation(
                        trace_id=TraceId(correlation.trace_id),
                        session_id=self._scheduler.snapshot.session_id,
                        sequence=EventSequence(correlation.seq),
                    ),
                ),
            )
        )

        if not isinstance(result, TransitionAccepted):
            raise SchedulerReflexRejectionError

        return TurnId(str(result.accepted_event.turn_id))
