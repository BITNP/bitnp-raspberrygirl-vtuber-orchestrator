"""模块契约说明.

职责: 提供 orchestrator.streaming_contracts
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NewType, Protocol, final

StreamingContractVersion = NewType("StreamingContractVersion", str)

TurnId = NewType("TurnId", str)

SegmentId = NewType("SegmentId", str)

CancellationEpoch = NewType("CancellationEpoch", int)

FlushRequestId = NewType("FlushRequestId", str)

GeneratedSsrc = NewType("GeneratedSsrc", int)


STREAMING_CONTRACT_VERSION: Final = StreamingContractVersion("1.0.0")

_RETRY_AFTER_MS: Final = 250

_TIMEOUT_AFTER_MS: Final = 750


class EnvelopeIdentity(Protocol):
    """类契约说明.

    职责: 声明 EnvelopeIdentity
    协议接口,约束实现方必须提供的行为。
    契约: 方法: trace_id、session_id、seq。
    """

    @property
    def trace_id(self) -> str:
        """函数契约说明.

        功能: 执行 trace_id 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        ...

    @property
    def session_id(self) -> str:
        """函数契约说明.

        功能: 执行 session_id 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        ...

    @property
    def seq(self) -> int:
        """函数契约说明.

        功能: 执行 seq 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `int`。
        """
        ...


@dataclass(frozen=True, slots=True)
class StreamKey:
    """类契约说明.

    职责: 保存 StreamKey
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: session_id、stream_id。
    """

    session_id: str

    stream_id: str


@dataclass(frozen=True, slots=True)
class StreamFlush:
    """类契约说明.

    职责: 保存 StreamFlush
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、turn_id、segment_id、ca
    ncellation_epoch、request_id、target_g
    enerated_ssrc。
    """

    stream: StreamKey

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch

    request_id: FlushRequestId

    target_generated_ssrc: GeneratedSsrc

    version: StreamingContractVersion = STREAMING_CONTRACT_VERSION

    correlation: EnvelopeIdentity | None = None


@dataclass(frozen=True, slots=True)
class FlushAcknowledgement:
    """类契约说明.

    职责: 保存 FlushAcknowledgement
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、turn_id、segment_id、ca
    ncellation_epoch、request_id、target_g
    enerated_ssrc。 方法: from_flush。
    """

    stream: StreamKey

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch

    request_id: FlushRequestId

    target_generated_ssrc: GeneratedSsrc

    version: StreamingContractVersion = STREAMING_CONTRACT_VERSION

    correlation: EnvelopeIdentity | None = None

    @classmethod
    def from_flush(cls, flush: StreamFlush) -> FlushAcknowledgement:
        """函数契约说明.

        功能: 执行 from_flush 的同步逻辑,并协调 cls。
        参数: cls 表示当前类。 flush:
        StreamFlush。 必填。
        契约: 同步调用。 返回
        `FlushAcknowledgement`。
        """
        return cls(
            stream=flush.stream,
            turn_id=flush.turn_id,
            segment_id=flush.segment_id,
            cancellation_epoch=flush.cancellation_epoch,
            request_id=flush.request_id,
            target_generated_ssrc=flush.target_generated_ssrc,
            version=flush.version,
            correlation=flush.correlation,
        )


@dataclass(frozen=True, slots=True)
class FlushFailure:
    """类契约说明.

    职责: 保存 FlushFailure
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: flush、reason。
    """

    flush: StreamFlush

    reason: str


class FlushClock(Protocol):
    """类契约说明.

    职责: 声明 FlushClock 协议接口,约束实现方必须提供的行为。
    契约: 方法: now_ms。
    """

    @property
    def now_ms(self) -> int:
        """函数契约说明.

        功能: 执行 now_ms 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `int`。
        """
        ...


class FlushSender(Protocol):
    """类契约说明.

    职责: 声明 FlushSender
    协议接口,约束实现方必须提供的行为。
    契约: 方法: send_flush。
    """

    def send_flush(self, flush: StreamFlush) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 同步调用。 返回 `None`。
        """
        ...


@dataclass(frozen=True, slots=True)
class _PendingFlush:
    """类契约说明.

    职责: 保存 _PendingFlush
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: flush、started_at_ms、retried。
    """

    flush: StreamFlush

    started_at_ms: int

    retried: bool = False


@final
class FlushAdmission:
    """类契约说明.

    职责: 定义 FlushAdmission 的状态、行为和对外协作边界。
    契约: 方法: __init__、begin、acknowledge、_
    flush_for_request、advance、admitted。
    """

    def __init__(self, *, clock: FlushClock, sender: FlushSender) -> None:
        """函数契约说明.

        功能: 初始化 FlushAdmission
        的字段并建立实例不变式。
        参数: self 表示当前实例。 clock:
        FlushClock。 必填。 sender:
        FlushSender。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._clock = clock

        self._sender = sender

        self._pending: dict[StreamKey, _PendingFlush] = {}

        self._admitted: set[StreamFlush] = set()

        self.failures: list[FlushFailure] = []

    def begin(self, flush: StreamFlush) -> None:
        """函数契约说明.

        功能: 执行 begin 的同步逻辑,并协调
        _PendingFlush, send_flush。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._admitted = {
            admitted for admitted in self._admitted if admitted.stream != flush.stream
        }

        self._pending[flush.stream] = _PendingFlush(flush, self._clock.now_ms)

        self._sender.send_flush(flush)

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> bool:
        """函数契约说明.

        功能: 执行 acknowledge 的同步逻辑,并协调
        get, add, append, from_flush。
        参数: self 表示当前实例。
        acknowledgement:
        FlushAcknowledgement。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        pending = self._pending.get(acknowledgement.stream)

        if pending is None or acknowledgement != FlushAcknowledgement.from_flush(
            pending.flush
        ):
            flush = (
                pending.flush
                if pending is not None
                else self._flush_for_request(acknowledgement)
            )

            self.failures.append(FlushFailure(flush=flush, reason="invalid_ack"))

            return False

        del self._pending[acknowledgement.stream]

        self._admitted.add(pending.flush)

        return True

    def _flush_for_request(self, acknowledgement: FlushAcknowledgement) -> StreamFlush:
        """函数契约说明.

        功能: 执行 _flush_for_request
        的同步逻辑,并协调 values, StreamFlush。
        参数: self 表示当前实例。
        acknowledgement:
        FlushAcknowledgement。 必填。
        契约: 同步调用。 返回 `StreamFlush`。
        """
        for pending in self._pending.values():
            if pending.flush.request_id == acknowledgement.request_id:
                return pending.flush

        return StreamFlush(
            stream=acknowledgement.stream,
            turn_id=acknowledgement.turn_id,
            segment_id=acknowledgement.segment_id,
            cancellation_epoch=acknowledgement.cancellation_epoch,
            request_id=acknowledgement.request_id,
            target_generated_ssrc=acknowledgement.target_generated_ssrc,
            version=acknowledgement.version,
            correlation=acknowledgement.correlation,
        )

    def advance(self) -> None:
        """函数契约说明.

        功能: 执行 advance 的同步逻辑,并协调 tuple,
        items, append, FlushFailure。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        now_ms = self._clock.now_ms

        for stream, pending in tuple(self._pending.items()):
            elapsed_ms = now_ms - pending.started_at_ms

            if elapsed_ms >= _TIMEOUT_AFTER_MS:
                del self._pending[stream]

                self.failures.append(
                    FlushFailure(flush=pending.flush, reason="timeout")
                )

            elif elapsed_ms >= _RETRY_AFTER_MS and not pending.retried:
                self._sender.send_flush(pending.flush)

                self._pending[stream] = _PendingFlush(
                    flush=pending.flush,
                    started_at_ms=pending.started_at_ms,
                    retried=True,
                )

    def admitted(self, flush: StreamFlush) -> bool:
        """函数契约说明.

        功能: 执行 admitted 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 flush:
        StreamFlush。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        return flush in self._admitted
