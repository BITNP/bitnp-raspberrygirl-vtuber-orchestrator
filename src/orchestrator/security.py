"""模块契约说明.

职责: 提供 orchestrator.security
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from orchestrator.config import DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH, OrchestratorConfig


def trusted_lan_token_is_configured(config: OrchestratorConfig) -> bool:
    """函数契约说明.

    功能: 执行
    trusted_lan_token_is_configured
    的同步逻辑,并协调 len。
    参数: config: OrchestratorConfig。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    return (
        config.trusted_lan_token is not None
        and len(config.trusted_lan_token) >= DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH
    )


def trusted_lan_token_is_valid(
    config: OrchestratorConfig,
    authorization: str | None,
) -> bool:
    """函数契约说明.

    功能: 执行 trusted_lan_token_is_valid
    的同步逻辑,并协调
    trusted_lan_token_is_configured。
    参数: config: OrchestratorConfig。 必填。
    authorization: str | None。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    if not trusted_lan_token_is_configured(config):
        return False

    return authorization == f"Bearer {config.trusted_lan_token}"
