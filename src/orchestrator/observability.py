"""模块契约说明.

职责: 提供 orchestrator.observability
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from typing import Final, Literal, TypedDict

from orchestrator.config import OrchestratorConfig
from orchestrator.operational_journal import OperationalJournal, OperationalRecord
from orchestrator.streaming_contracts import CancellationEpoch, StreamKey
from orchestrator.transport_control import EnvelopeCorrelation

type OnsiteStage = Literal[
    "endpoint",
    "asr_partial",
    "asr_final",
    "asr_failure",
    "answer",
    "tts",
    "queue",
    "drop",
    "rtp_egress",
    "rtp_ingress",
    "classifier_decision",
    "classifier_failure",
    "cancellation",
    "flush",
    "flush_ack",
    "playback_state",
]


class JsonLogRecord(TypedDict):
    """类契约说明.

    职责: 定义 JsonLogRecord 的状态、行为和对外协作边界。
    契约: 字段: service、service_version、leve
    l、message、trace_id、session_id。
    """

    service: str

    service_version: str

    level: Literal["debug", "info", "warning", "error"]

    message: str

    trace_id: str

    session_id: str


@dataclass(frozen=True, slots=True)
class LatencyMetric:
    """类契约说明.

    职责: 保存 LatencyMetric
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    service、operation、latency_ms。
    """

    service: str

    operation: str

    latency_ms: float


@dataclass(frozen=True, slots=True)
class QueueMetric:
    """类契约说明.

    职责: 保存 QueueMetric
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: service、queue_name、depth。
    """

    service: str

    queue_name: str

    depth: int


@dataclass(frozen=True, slots=True)
class StageCorrelation:
    """类契约说明.

    职责: 保存 StageCorrelation
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: trace_id、session_id、seq、turn
    _id、segment_id、cancellation_epoch。
    """

    trace_id: str

    session_id: str

    seq: int

    turn_id: str | None = None

    segment_id: str | None = None

    cancellation_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class StageRecord:
    """类契约说明.

    职责: 保存 StageRecord
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stage、trace_id、session_id、se
    q、turn_id、segment_id。
    """

    stage: OnsiteStage

    trace_id: str

    session_id: str

    seq: int

    turn_id: str | None = None

    segment_id: str | None = None

    cancellation_epoch: int | None = None

    latency_ms: float | None = None

    queue_name: str | None = None

    queue_depth: int | None = None

    drop_count: int | None = None


@dataclass(frozen=True, slots=True)
class StageDetails:
    """类契约说明.

    职责: 保存 StageDetails
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: latency_ms、queue_name、queue_
    depth、drop_count。
    """

    latency_ms: float | None = None

    queue_name: str | None = None

    queue_depth: int | None = None

    drop_count: int | None = None


DEFAULT_STAGE_DETAILS: Final = StageDetails()


@dataclass(slots=True)
class OnsiteObservability:
    """类契约说明.

    职责: 保存 OnsiteObservability
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: config、records、logs、latencie
    s、queues、journal。 方法: __init__、bind_
    correlation、correlation、record、recor
    d_stream。
    """

    config: OrchestratorConfig

    records: list[StageRecord]

    logs: list[JsonLogRecord]

    latencies: list[LatencyMetric]

    queues: list[QueueMetric]

    journal: OperationalJournal

    _envelopes: dict[StreamKey, EnvelopeCorrelation]

    def __init__(self, config: OrchestratorConfig) -> None:
        """函数契约说明.

        功能: 初始化 OnsiteObservability
        的字段并建立实例不变式。
        参数: self 表示当前实例。 config:
        OrchestratorConfig。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self.config = config

        self.records = []

        self.logs = []

        self.latencies = []

        self.queues = []

        self.journal = OperationalJournal()

        self._envelopes = {}

    def bind_correlation(
        self, stream: StreamKey, correlation: EnvelopeCorrelation
    ) -> None:
        """函数契约说明.

        功能: 执行 bind_correlation
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 correlation:
        EnvelopeCorrelation。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._envelopes[stream] = correlation

    def correlation(
        self,
        stream: StreamKey,
        turn_id: str | None,
        segment_id: str | None,
        epoch: CancellationEpoch | None,
    ) -> StageCorrelation | None:
        """函数契约说明.

        功能: 执行 correlation 的同步逻辑,并协调
        get, StageCorrelation, int。
        参数: self 表示当前实例。 stream:
        StreamKey。 必填。 turn_id: str |
        None。 必填。 segment_id: str |
        None。 必填。 epoch:
        CancellationEpoch | None。 必填。
        契约: 同步调用。 返回 `StageCorrelation |
        None`。
        """
        envelope = self._envelopes.get(stream)

        if envelope is None:
            return None

        return StageCorrelation(
            trace_id=envelope.trace_id,
            session_id=envelope.session_id,
            seq=envelope.seq,
            turn_id=turn_id,
            segment_id=segment_id,
            cancellation_epoch=int(epoch) if epoch is not None else None,
        )

    def record(
        self,
        stage: OnsiteStage,
        correlation: StageCorrelation,
        details: StageDetails = DEFAULT_STAGE_DETAILS,
    ) -> None:
        """函数契约说明.

        功能: 执行 record 的同步逻辑,并协调 append,
        StageRecord, json_log_record,
        OperationalRecord。
        参数: self 表示当前实例。 stage:
        OnsiteStage。 必填。 correlation:
        StageCorrelation。 必填。 details:
        StageDetails。 可省略。
        契约: 同步调用。 返回 `None`。
        """
        self.records.append(
            StageRecord(
                stage=stage,
                trace_id=correlation.trace_id,
                session_id=correlation.session_id,
                seq=correlation.seq,
                turn_id=correlation.turn_id,
                segment_id=correlation.segment_id,
                cancellation_epoch=correlation.cancellation_epoch,
                latency_ms=details.latency_ms,
                queue_name=details.queue_name,
                queue_depth=details.queue_depth,
                drop_count=details.drop_count,
            )
        )

        self.logs.append(
            json_log_record(
                self.config,
                level="info",
                message=stage,
                trace_id=correlation.trace_id,
                session_id=correlation.session_id,
            )
        )

        self.journal.append(
            OperationalRecord(
                stage=stage,
                trace_id=correlation.trace_id,
                session_id=correlation.session_id,
                turn_id=correlation.turn_id,
                segment_id=correlation.segment_id,
                task_id=None,
                outcome=_stage_outcome(stage),
            )
        )

        if details.latency_ms is not None:
            self.latencies.append(
                latency_metric(
                    self.config, operation=stage, latency_ms=details.latency_ms
                )
            )

        if details.queue_name is not None and details.queue_depth is not None:
            self.queues.append(
                queue_metric(
                    self.config,
                    queue_name=details.queue_name,
                    depth=details.queue_depth,
                )
            )

    def record_stream(
        self,
        stage: OnsiteStage,
        stream: StreamKey,
        command: StageCorrelation | None = None,
        details: StageDetails = DEFAULT_STAGE_DETAILS,
    ) -> None:
        """函数契约说明.

        功能: 执行 record_stream 的同步逻辑,并协调
        correlation, record。
        参数: self 表示当前实例。 stage:
        OnsiteStage。 必填。 stream:
        StreamKey。 必填。 command:
        StageCorrelation | None。 可省略。
        details: StageDetails。 可省略。
        契约: 同步调用。 返回 `None`。
        """
        correlation = command or self.correlation(stream, None, None, None)

        if correlation is not None:
            self.record(stage, correlation, details)


def json_log_record(
    config: OrchestratorConfig,
    *,
    level: Literal["debug", "info", "warning", "error"],
    message: str,
    trace_id: str,
    session_id: str,
) -> JsonLogRecord:
    """函数契约说明.

    功能: 执行 json_log_record
    的同步逻辑,并维持签名契约。
    参数: config: OrchestratorConfig。 必填。
    level: Literal['debug', 'info',
    'warning', 'error']。 必填。 message:
    str。 必填。 trace_id: str。 必填。
    session_id: str。 必填。
    契约: 同步调用。 返回 `JsonLogRecord`。
    """
    return {
        "service": config.service_name,
        "service_version": config.service_version,
        "level": level,
        "message": message,
        "trace_id": trace_id,
        "session_id": session_id,
    }


def latency_metric(
    config: OrchestratorConfig,
    *,
    operation: str,
    latency_ms: float,
) -> LatencyMetric:
    """函数契约说明.

    功能: 执行 latency_metric 的同步逻辑,并协调
    LatencyMetric。
    参数: config: OrchestratorConfig。 必填。
    operation: str。 必填。 latency_ms:
    float。 必填。
    契约: 同步调用。 返回 `LatencyMetric`。
    """
    return LatencyMetric(
        service=config.service_name,
        operation=operation,
        latency_ms=latency_ms,
    )


def queue_metric(
    config: OrchestratorConfig,
    *,
    queue_name: str,
    depth: int,
) -> QueueMetric:
    """函数契约说明.

    功能: 执行 queue_metric 的同步逻辑,并协调
    QueueMetric。
    参数: config: OrchestratorConfig。 必填。
    queue_name: str。 必填。 depth: int。 必填。
    契约: 同步调用。 返回 `QueueMetric`。
    """
    return QueueMetric(service=config.service_name, queue_name=queue_name, depth=depth)


def _stage_outcome(stage: OnsiteStage) -> str:
    """函数契约说明.

    功能: 执行 _stage_outcome 的同步逻辑,并维持签名契约。
    参数: stage: OnsiteStage。 必填。
    契约: 同步调用。 返回 `str`。
    """
    if stage in {"asr_failure", "classifier_failure"}:
        return "failure"

    if stage == "cancellation":
        return "cancelled"

    return "observed"
