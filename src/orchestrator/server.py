"""模块契约说明.

职责: 提供 orchestrator.server
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 OrchestratorServer
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: config、registry、sessions。
    方法: from_config、health、readiness、aut
    henticated_readiness、_readiness_fail
    ure_reasons。
    """

    config: OrchestratorConfig

    registry: ConnectionRegistry

    sessions: SessionManager

    @classmethod
    def from_config(cls, config: OrchestratorConfig) -> "OrchestratorServer":
        """函数契约说明.

        功能: 执行 from_config 的同步逻辑,并协调
        cls, ConnectionRegistry,
        SessionManager。
        参数: cls 表示当前类。 config:
        OrchestratorConfig。 必填。
        契约: 同步调用。 返回
        `'OrchestratorServer'`。
        """
        return cls(
            config=config,
            registry=ConnectionRegistry(),
            sessions=SessionManager(session_id_prefix=config.session_id_prefix),
        )

    def health(self) -> HealthReport:
        """函数契约说明.

        功能: 执行 health 的同步逻辑,并协调
        HealthReport。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `HealthReport`。
        """
        return HealthReport(
            service=self.config.service_name,
            service_version=self.config.service_version,
            status=ServiceStatus.READY,
        )

    def readiness(self) -> ReadinessReport:
        """函数契约说明.

        功能: 执行 readiness 的同步逻辑,并协调
        active_identities,
        _readiness_failure_reasons,
        ReadinessReport, len。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `ReadinessReport`。
        """
        active_modules = self.registry.active_identities()

        failure_reasons = self._readiness_failure_reasons()

        status = (
            ServiceStatus.READY if len(failure_reasons) == 0 else ServiceStatus.DEGRADED
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
        """函数契约说明.

        功能: 执行 authenticated_readiness
        的同步逻辑,并协调 _provider_capable,
        AuthenticatedReadinessReport,
        trusted_lan_token_is_valid。
        参数: self 表示当前实例。 authorization:
        str | None。 必填。 transport:
        TransportReadinessSource | None。
        必填。 provider_probe:
        ProviderCapabilityProbe | None。
        必填。
        契约: 同步调用。 返回
        `AuthenticatedReadinessReport |
        None`。
        """
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
        """函数契约说明.

        功能: 执行
        _readiness_failure_reasons
        的同步逻辑,并协调 tuple, append, len。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[str, ...]`。
        """
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
    """函数契约说明.

    功能: 执行 _provider_capable 的同步逻辑,并协调
    ThreadPoolExecutor, submit, result,
    shutdown。
    参数: provider_probe:
    ProviderCapabilityProbe | None。 必填。
    契约: 同步调用。 返回 `bool`。
    """
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
