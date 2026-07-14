"""Central Orchestrator server shell used by tests and local runbook commands."""

from dataclasses import dataclass

from orchestrator.config import DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH, OrchestratorConfig
from orchestrator.health import HealthReport, ReadinessReport, ServiceStatus
from orchestrator.registry import ConnectionRegistry
from orchestrator.sessions import SessionManager


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
