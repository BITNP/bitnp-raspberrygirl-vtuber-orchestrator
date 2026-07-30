"""Health and readiness report values for the Orchestrator."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Protocol

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


class TransportReadinessSource(Protocol):
    """Supplies separately meaningful listener and route readiness facts."""

    @property
    def listener_ready(self) -> bool:
        """Return whether the UDP and WSS listeners are active."""
        ...

    @property
    def route_ready(self) -> bool:
        """Return whether a Mic source has a paired Sound route."""
        ...


class ProviderCapabilityProbe(Protocol):
    """Executes a provider capability probe within the supplied deadline."""

    def probe(self, timeout_ms: int) -> bool:
        """Return capability success before the supplied timeout expires."""
        ...


@dataclass(frozen=True, slots=True)
class AuthenticatedReadinessReport:
    """Token-gated readiness facts that never infer remote provider health."""

    listener_ready: bool
    route_ready: bool
    provider_capable: bool
