"""Dependency-free observability value helpers."""

from dataclasses import dataclass
from typing import Literal, TypedDict

from orchestrator.config import OrchestratorConfig


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
