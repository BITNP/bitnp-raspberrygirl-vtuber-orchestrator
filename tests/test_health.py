from orchestrator.config import load_fake_config
from orchestrator.health import ServiceStatus
from orchestrator.server import OrchestratorServer


def test_health_reports_ready_with_fake_config_without_peer_modules() -> None:
    # Given: a central Orchestrator shell using only fake local config.
    server = OrchestratorServer.from_config(load_fake_config())

    # When: readiness is requested before any peer module connects.
    readiness = server.readiness()

    # Then: the shell is ready and reports only Orchestrator-owned metadata.
    assert readiness.service == "orchestrator"
    assert readiness.service_version == "0.1.0"
    assert readiness.status is ServiceStatus.READY
    assert readiness.active_modules == ()
    assert readiness.peer_dependency_count == 0


def test_fake_config_uses_orchestrator_identity() -> None:
    # Given: the fake config loader used by task-level smoke tests.
    config = load_fake_config()

    # When: the server health snapshot is built from that config.
    report = OrchestratorServer.from_config(config).health()

    # Then: the central service identity is stable and mode-free.
    assert report.service == "orchestrator"
    assert report.service_version == "0.1.0"
    assert report.status is ServiceStatus.READY
