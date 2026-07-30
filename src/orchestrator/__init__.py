"""模块契约说明.

职责: 定义 orchestrator 包的导入边界和对外可见包语义。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from orchestrator.config import OrchestratorConfig, load_fake_config
from orchestrator.health import HealthReport, ReadinessReport, ServiceStatus
from orchestrator.registry import ConnectionRegistry, ModuleIdentity
from orchestrator.server import OrchestratorServer

__all__ = [
    "ConnectionRegistry",
    "HealthReport",
    "ModuleIdentity",
    "OrchestratorConfig",
    "OrchestratorServer",
    "ReadinessReport",
    "ServiceStatus",
    "load_fake_config",
]
