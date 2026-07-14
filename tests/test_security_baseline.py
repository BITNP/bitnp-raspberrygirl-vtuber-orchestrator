from pathlib import Path

from orchestrator.config import load_config_from_env
from orchestrator.health import ServiceStatus
from orchestrator.observability import json_log_record, latency_metric, queue_metric
from orchestrator.security import trusted_lan_token_is_valid
from orchestrator.server import OrchestratorServer


def test_real_provider_readiness_reports_missing_llm_key_without_crashing() -> None:
    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
            "TRUSTED_LAN_TOKEN": "placeholder-token-123",
        },
    )

    readiness = OrchestratorServer.from_config(config).readiness()

    assert readiness.status is ServiceStatus.DEGRADED
    assert readiness.failure_reasons == (
        "ORCHESTRATOR_LLM_API_KEY is required for openai_compatible",
    )


def test_real_provider_readiness_reports_invalid_token_without_crashing() -> None:
    config = load_config_from_env(
        {
            "ORCHESTRATOR_LLM_PROVIDER": "openai_compatible",
            "ORCHESTRATOR_LLM_API_KEY": "placeholder-test-key",
            "TRUSTED_LAN_TOKEN": "short",
        },
    )

    readiness = OrchestratorServer.from_config(config).readiness()

    assert readiness.status is ServiceStatus.DEGRADED
    assert readiness.failure_reasons == (
        "TRUSTED_LAN_TOKEN must be at least 12 characters",
    )
    assert not trusted_lan_token_is_valid(config, "Bearer short")


def test_observability_helpers_emit_json_logs_and_metrics() -> None:
    config = load_config_from_env({})

    log_record = json_log_record(
        config,
        level="info",
        message="ready",
        trace_id="trace-001",
        session_id="session-001",
    )
    latency = latency_metric(config, operation="readiness", latency_ms=3.5)
    queue = queue_metric(config, queue_name="turns", depth=0)

    assert log_record["service"] == "orchestrator"
    assert log_record["service_version"] == "0.1.0"
    assert log_record["trace_id"] == "trace-001"
    assert latency.operation == "readiness"
    assert queue.queue_name == "turns"


def test_env_examples_do_not_commit_real_secrets() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples = sorted(repo_root.glob("*/.env.example"))

    assert {path.parent.name for path in examples} == {
        "asr",
        "comments",
        "mic",
        "orchestrator",
        "sound",
        "tts",
    }
    for path in examples:
        text = path.read_text(encoding="utf-8")
        assert "placeholder" in text
        assert "sk-" not in text
        assert "local-dev-token" not in text
