"""Health and readiness report values for the Orchestrator."""

from dataclasses import dataclass
from enum import StrEnum, unique

from orchestrator.registry import ModuleIdentity


@unique
class ServiceStatus(StrEnum):
    """Lifecycle states exposed by health and readiness surfaces."""

    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Minimal liveness report for the central service."""

    service: str
    service_version: str
    status: ServiceStatus


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Dependency-aware readiness report for local LAN deployment."""

    service: str
    service_version: str
    status: ServiceStatus
    active_modules: tuple[ModuleIdentity, ...]
    peer_dependency_count: int
    failure_reasons: tuple[str, ...] = ()
