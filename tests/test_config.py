from pathlib import Path

import pytest

from orchestrator.config import ConfigParseError, load_config_from_env
from orchestrator.response_execution_mode import ResponseExecutionMode


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"ORCHESTRATOR_TLS_CA_PATH": "   "},
    ],
)
def test_load_config_from_env_uses_no_ca_bundle_when_path_is_absent_or_blank(
    env: dict[str, str],
) -> None:
    config = load_config_from_env(env)

    assert config.tls_ca_path is None


def test_load_config_from_env_uses_configured_ca_bundle_path() -> None:
    config = load_config_from_env(
        {"ORCHESTRATOR_TLS_CA_PATH": "/etc/bitnp/internal-ca.pem"}
    )

    assert config.tls_ca_path == Path("/etc/bitnp/internal-ca.pem")


def test_load_config_from_env_parses_response_execution_mode() -> None:
    config = load_config_from_env(
        {"ORCHESTRATOR_RESPONSE_EXECUTION_MODE": "new_shadow"}
    )

    assert config.response_execution_mode is ResponseExecutionMode.NEW_SHADOW

    with pytest.raises(ConfigParseError, match="ORCHESTRATOR_RESPONSE_EXECUTION_MODE"):
        _ = load_config_from_env(
            {"ORCHESTRATOR_RESPONSE_EXECUTION_MODE": "unsupported"}
        )


def test_load_config_from_env_rejects_invalid_provider_with_ca_bundle_path() -> None:
    with pytest.raises(ConfigParseError, match="ORCHESTRATOR_LLM_PROVIDER"):
        _ = load_config_from_env(
            {
                "ORCHESTRATOR_LLM_PROVIDER": "unsupported",
                "ORCHESTRATOR_TLS_CA_PATH": "/etc/bitnp/internal-ca.pem",
            }
        )
