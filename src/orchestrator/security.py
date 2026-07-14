"""Trusted-LAN token checks for local Orchestrator connections."""

from orchestrator.config import DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH, OrchestratorConfig


def trusted_lan_token_is_configured(config: OrchestratorConfig) -> bool:
    """Return whether a configured token is long enough to enforce."""
    return (
        config.trusted_lan_token is not None
        and len(config.trusted_lan_token) >= DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH
    )


def trusted_lan_token_is_valid(
    config: OrchestratorConfig,
    authorization: str | None,
) -> bool:
    """Check a bearer authorization header against configured LAN token."""
    if not trusted_lan_token_is_configured(config):
        return False
    return authorization == f"Bearer {config.trusted_lan_token}"
