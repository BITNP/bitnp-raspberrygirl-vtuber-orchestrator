from pathlib import Path

import pytest

from orchestrator.config import ConfigParseError, load_config_from_env


@pytest.mark.parametrize("env", [{}, {"ORCHESTRATOR_TLS_CA_PATH": "   "}])
def test_load_config_from_env_uses_no_ca_bundle_when_absent(
    env: dict[str, str],
) -> None:
    assert load_config_from_env(env).tls_ca_path is None


def test_load_config_from_env_uses_configured_ca_bundle_path() -> None:
    config = load_config_from_env(
        {"ORCHESTRATOR_TLS_CA_PATH": "/etc/bitnp/internal-ca.pem"}
    )
    assert config.tls_ca_path == Path("/etc/bitnp/internal-ca.pem")


def test_retired_gate_and_execution_mode_environment_are_ignored() -> None:
    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_GATE_MODEL": "retired",
            "ORCHESTRATOR_RESPONSE_EXECUTION_MODE": "new_shadow",
        }
    )
    assert not hasattr(config, "llm_gate_model")
    assert not hasattr(config, "response_execution_mode")


@pytest.mark.parametrize("dialect", [None, "", "unsupported"])
def test_real_llm_requires_explicit_reasoning_dialect(dialect: str | None) -> None:
    env = {"ORCHESTRATOR_LLM_PROVIDER": "openai_compatible"}
    if dialect is not None:
        env["ORCHESTRATOR_LLM_REASONING_DIALECT"] = dialect
    with pytest.raises(ConfigParseError, match="ORCHESTRATOR_LLM_REASONING_DIALECT"):
        _ = load_config_from_env(env)


def test_real_llm_parses_brain_and_maintenance_routes() -> None:
    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
            "ORCHESTRATOR_LLM_REASONING_DIALECT": "openai",
            "ORCHESTRATOR_LLM_MODEL": "default-model",
            "ORCHESTRATOR_LLM_BRAIN_MODEL": "brain-model",
            "ORCHESTRATOR_LLM_MAINTENANCE_MODEL": "   ",
        }
    )
    assert config.llm_brain_model == "brain-model"
    assert config.llm_maintenance_model is None


def test_ppt_deck_catalog_is_bounded_and_controlled() -> None:
    config = load_config_from_env(
        {"ORCHESTRATOR_PPT_DECK_CATALOG": "launch-deck,product.v2"}
    )
    assert config.ppt_deck_catalog == frozenset({"launch-deck", "product.v2"})

    for invalid in ("launch-deck,launch-deck", "../../secret", "deck/path"):
        with pytest.raises(ConfigParseError, match="ORCHESTRATOR_PPT_DECK_CATALOG"):
            _ = load_config_from_env({"ORCHESTRATOR_PPT_DECK_CATALOG": invalid})
