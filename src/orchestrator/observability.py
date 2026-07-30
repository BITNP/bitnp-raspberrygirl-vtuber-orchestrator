"""Dependency-free observability value helpers."""

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
    """Structured JSON log shape shared by tests and service code."""

    service: str
    service_version: str
    level: Literal["debug", "info", "warning", "error"]
    message: str
    trace_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class LatencyMetric:
    """Single-operation latency measurement."""

    service: str
    operation: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class QueueMetric:
    """Queue depth measurement for backpressure visibility."""

    service: str
    queue_name: str
    depth: int


@dataclass(frozen=True, slots=True)
class StageCorrelation:
    """Identifiers that join one onsite stage record to its stream turn."""

    trace_id: str
    session_id: str
    seq: int
    turn_id: str | None = None
    segment_id: str | None = None
    cancellation_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One structured onsite streaming state transition."""

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
    """Optional measurement values attached to one stage transition."""

    latency_ms: float | None = None
    queue_name: str | None = None
    queue_depth: int | None = None
    drop_count: int | None = None


DEFAULT_STAGE_DETAILS: Final = StageDetails()


@dataclass(slots=True)
class OnsiteObservability:
    """Collects correlated records for the bounded onsite pipeline."""

    config: OrchestratorConfig
    records: list[StageRecord]
    logs: list[JsonLogRecord]
    latencies: list[LatencyMetric]
    queues: list[QueueMetric]
    journal: OperationalJournal
    _envelopes: dict[StreamKey, EnvelopeCorrelation]

    def __init__(self, config: OrchestratorConfig) -> None:
        """Create an empty recorder for one configured Orchestrator instance."""
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
        """Retain the authenticated source envelope for RTP-only later stages."""
        self._envelopes[stream] = correlation

    def correlation(
        self,
        stream: StreamKey,
        turn_id: str | None,
        segment_id: str | None,
        epoch: CancellationEpoch | None,
    ) -> StageCorrelation | None:
        """Resolve a complete correlation value for one stream-local stage."""
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
        """Record one correlated transition using the existing helper values."""
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
        """Record a transition from identifiers held at a transport boundary."""
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
    """Build a trace-aware JSON log record."""
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
    """Build a service-scoped latency metric."""
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
    """Build a service-scoped queue depth metric."""
    return QueueMetric(service=config.service_name, queue_name=queue_name, depth=depth)


def _stage_outcome(stage: OnsiteStage) -> str:
    if stage in {"asr_failure", "classifier_failure"}:
        return "failure"
    if stage == "cancellation":
        return "cancelled"
    return "observed"
