
from orchestrator.config import DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH, OrchestratorConfig


def trusted_lan_token_is_configured(config: OrchestratorConfig) -> bool:
    return (
        config.trusted_lan_token is not None
        and len(config.trusted_lan_token) >= DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH
    )


def trusted_lan_token_is_valid(
    config: OrchestratorConfig,
    authorization: str | None,
) -> bool:
    if not trusted_lan_token_is_configured(config):
        return False

    return authorization == f"Bearer {config.trusted_lan_token}"
