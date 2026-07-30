"""Central Orchestrator server shell used by tests and local runbook commands."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Final

from orchestrator.config import DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH, OrchestratorConfig
from orchestrator.health import (
    AuthenticatedReadinessReport,
    HealthReport,
    ProviderCapabilityProbe,
    ReadinessReport,
    ServiceStatus,
    TransportReadinessSource,
)
from orchestrator.registry import ConnectionRegistry
from orchestrator.security import trusted_lan_token_is_valid
from orchestrator.sessions import SessionManager

_PROVIDER_PROBE_TIMEOUT_SECONDS: Final = 0.45
_PROVIDER_PROBE_TIMEOUT_MS: Final = 500


@dataclass(frozen=True, slots=True)
class OrchestratorServer:
    """In-process Orchestrator shell with health/readiness surfaces."""

    config: OrchestratorConfig
    registry: ConnectionRegistry
    sessions: SessionManager

    @classmethod
    def from_config(cls, config: OrchestratorConfig) -> "OrchestratorServer":
        """Create a server shell from typed configuration."""
        return cls(
            config=config,
            registry=ConnectionRegistry(),
            sessions=SessionManager(session_id_prefix=config.session_id_prefix),
        )

    def health(self) -> HealthReport:
        """Return the liveness report without requiring peer modules."""
        return HealthReport(
            service=self.config.service_name,
            service_version=self.config.service_version,
            status=ServiceStatus.READY,
        )

    def readiness(self) -> ReadinessReport:
        """Return readiness while preserving zero peer dependencies."""
        active_modules = self.registry.active_identities()
        failure_reasons = self._readiness_failure_reasons()
        status = (
            ServiceStatus.READY
            if len(failure_reasons) == 0
            else ServiceStatus.DEGRADED
        )
        return ReadinessReport(
            service=self.config.service_name,
            service_version=self.config.service_version,
            status=status,
            active_modules=active_modules,
            peer_dependency_count=0,
            failure_reasons=failure_reasons,
        )

    def authenticated_readiness(
        self,
        authorization: str | None,
        transport: TransportReadinessSource | None,
        provider_probe: ProviderCapabilityProbe | None,
    ) -> AuthenticatedReadinessReport | None:
        """Return readiness details only to the configured trusted-LAN bearer."""
        if not trusted_lan_token_is_valid(self.config, authorization):
            return None
        listener_ready = transport.listener_ready if transport is not None else False
        route_ready = transport.route_ready if transport is not None else False
        provider_capable = _provider_capable(provider_probe)
        return AuthenticatedReadinessReport(
            listener_ready=listener_ready,
            route_ready=route_ready,
            provider_capable=provider_capable,
        )

    def _readiness_failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if (
            self.config.llm_provider == "openai_compatible"
            and self.config.llm_api_key is None
        ):
            reasons.append("ORCHESTRATOR_LLM_API_KEY is required for openai_compatible")
        if (
            self.config.trusted_lan_token is not None
            and len(self.config.trusted_lan_token)
            < DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH
        ):
            reasons.append("TRUSTED_LAN_TOKEN must be at least 12 characters")
        return tuple(reasons)


def _provider_capable(provider_probe: ProviderCapabilityProbe | None) -> bool:
    """Bound an isolated synchronous probe so readiness never blocks a caller."""
    if provider_probe is None:
        return False
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="provider-probe")
    future = executor.submit(provider_probe.probe, _PROVIDER_PROBE_TIMEOUT_MS)
    try:
        return future.result(timeout=_PROVIDER_PROBE_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        return False
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
