"""模块契约说明.

职责: 提供 orchestrator.health
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Protocol

from orchestrator.registry import ModuleIdentity


@unique
class ServiceStatus(StrEnum):
    """类契约说明.

    职责: 定义 ServiceStatus 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    STARTING = "starting"

    READY = "ready"

    DEGRADED = "degraded"

    STOPPING = "stopping"

    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class HealthReport:
    """类契约说明.

    职责: 保存 HealthReport
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    service、service_version、status。
    """

    service: str

    service_version: str

    status: ServiceStatus


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """类契约说明.

    职责: 保存 ReadinessReport
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: service、service_version、stat
    us、active_modules、peer_dependency_co
    unt、failure_reasons。
    """

    service: str

    service_version: str

    status: ServiceStatus

    active_modules: tuple[ModuleIdentity, ...]

    peer_dependency_count: int

    failure_reasons: tuple[str, ...] = ()


class TransportReadinessSource(Protocol):
    """类契约说明.

    职责: 声明 TransportReadinessSource
    协议接口,约束实现方必须提供的行为。
    契约: 方法: listener_ready、route_ready。
    """

    @property
    def listener_ready(self) -> bool:
        """函数契约说明.

        功能: 执行 listener_ready
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `bool`。
        """
        ...

    @property
    def route_ready(self) -> bool:
        """函数契约说明.

        功能: 执行 route_ready
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `bool`。
        """
        ...


class ProviderCapabilityProbe(Protocol):
    """类契约说明.

    职责: 声明 ProviderCapabilityProbe
    协议接口,约束实现方必须提供的行为。
    契约: 方法: probe。
    """

    def probe(self, timeout_ms: int) -> bool:
        """函数契约说明.

        功能: 执行 probe 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 timeout_ms:
        int。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        ...


@dataclass(frozen=True, slots=True)
class AuthenticatedReadinessReport:
    """类契约说明.

    职责: 保存 AuthenticatedReadinessReport
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: listener_ready、route_ready、p
    rovider_capable。
    """

    listener_ready: bool

    route_ready: bool

    provider_capable: bool
