"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from pathlib import Path

from orchestrator.config import load_config_from_env
from orchestrator.health import ServiceStatus
from orchestrator.observability import json_log_record, latency_metric, queue_metric
from orchestrator.security import trusted_lan_token_is_valid
from orchestrator.server import OrchestratorServer


def test_real_provider_readiness_reports_missing_llm_key_without_crashing() -> None:
    """函数契约说明.

    功能: 验证 real provider readiness
    reports missing llm key without
    crashing 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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
    """函数契约说明.

    功能: 验证 real provider readiness
    reports invalid token without
    crashing 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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
    """函数契约说明.

    功能: 验证 observability helpers emit
    json logs and metrics 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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


def test_authenticated_readiness_rejects_wrong_token_without_leaking_state() -> None:
    # Given: a server whose readiness surface is protected by a LAN token.

    """函数契约说明.

    功能: 验证 authenticated readiness
    rejects wrong token without leaking
    state 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    config = load_config_from_env({"TRUSTED_LAN_TOKEN": "readiness-token-123"})

    server = OrchestratorServer.from_config(config)

    # When: an untrusted caller supplies a different bearer token.

    report = server.authenticated_readiness("Bearer wrong-token", None, None)

    # Then: no readiness details are exposed.

    assert report is None


def test_env_examples_do_not_commit_real_secrets() -> None:
    """函数契约说明.

    功能: 验证 env examples do not commit
    real secrets 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    repo_root = Path(__file__).resolve().parents[1]

    examples = (repo_root / ".env.example",)

    for path in examples:
        text = path.read_text(encoding="utf-8")

        assert "placeholder" in text

        assert "sk-" not in text

        assert "local-dev-token" not in text
