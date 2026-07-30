
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Protocol

from orchestrator.registry import ModuleIdentity


@unique
class ServiceStatus(StrEnum):

    STARTING = "starting"

    READY = "ready"

    DEGRADED = "degraded"

    STOPPING = "stopping"

    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class HealthReport:

    service: str

    service_version: str

    status: ServiceStatus


@dataclass(frozen=True, slots=True)
class ReadinessReport:

    service: str

    service_version: str

    status: ServiceStatus

    active_modules: tuple[ModuleIdentity, ...]

    peer_dependency_count: int

    failure_reasons: tuple[str, ...] = ()


class TransportReadinessSource(Protocol):

    @property
    def listener_ready(self) -> bool:
        ...

    @property
    def route_ready(self) -> bool:
        ...


class ProviderCapabilityProbe(Protocol):

    def probe(self, timeout_ms: int) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class AuthenticatedReadinessReport:

    listener_ready: bool

    route_ready: bool

    provider_capable: bool
