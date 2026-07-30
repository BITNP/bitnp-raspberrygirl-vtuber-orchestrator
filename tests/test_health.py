from dataclasses import dataclass, replace
from threading import Event
from time import monotonic

from orchestrator.config import TrustedLanToken, load_fake_config
from orchestrator.health import ServiceStatus
from orchestrator.server import OrchestratorServer


@dataclass(frozen=True, slots=True)
class _Transport:
    listener_ready: bool
    route_ready: bool


@dataclass(frozen=True, slots=True)
class _Probe:
    capable: bool
    calls: int = 0

    def probe(self, timeout_ms: int) -> bool:
        _ = timeout_ms
        return self.capable


@dataclass(slots=True)
class _NoncooperativeProbe:
    started: Event
    release: Event

    def probe(self, timeout_ms: int) -> bool:
        _ = timeout_ms
        self.started.set()
        return self.release.wait()


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


def test_authenticated_readiness_distinguishes_listener_route_and_provider_capability(
) -> None:
    # Given: a token-protected fake server with live listeners but no Sound route.
    config = replace(
        load_fake_config(), trusted_lan_token=TrustedLanToken("readiness-token-123")
    )
    server = OrchestratorServer.from_config(config)

    # When: the trusted caller requests each independently changing state.
    listener_only = server.authenticated_readiness(
        "Bearer readiness-token-123",
        _Transport(listener_ready=True, route_ready=False),
        _Probe(capable=True),
    )
    route_ready = server.authenticated_readiness(
        "Bearer readiness-token-123",
        _Transport(listener_ready=True, route_ready=True),
        _Probe(capable=True),
    )

    # Then: listener-only is not route-ready, while a probed fake provider is capable.
    assert listener_only is not None
    assert listener_only.listener_ready is True
    assert listener_only.route_ready is False
    assert listener_only.provider_capable is True
    assert route_ready is not None
    assert route_ready.listener_ready is True
    assert route_ready.route_ready is True
    assert route_ready.provider_capable is True


def test_authenticated_readiness_requires_a_successful_bounded_remote_probe() -> None:
    # Given: a token-protected server and a failed explicit provider probe.
    config = replace(
        load_fake_config(), trusted_lan_token=TrustedLanToken("readiness-token-123")
    )
    server = OrchestratorServer.from_config(config)

    # When: the trusted caller reads readiness after the probe reports failure.
    report = server.authenticated_readiness(
        "Bearer readiness-token-123",
        _Transport(listener_ready=True, route_ready=True),
        _Probe(capable=False),
    )

    # Then: the provider is not claimed capable.
    assert report is not None
    assert report.provider_capable is False


def test_authenticated_readiness_rejects_noncooperative_probe_within_deadline() -> None:
    # Given: a provider probe that does not honor its supplied deadline.
    config = replace(
        load_fake_config(), trusted_lan_token=TrustedLanToken("readiness-token-123")
    )
    server = OrchestratorServer.from_config(config)
    probe = _NoncooperativeProbe(started=Event(), release=Event())

    # When: authenticated readiness invokes the blocking probe.
    started_at = monotonic()
    report = server.authenticated_readiness(
        "Bearer readiness-token-123",
        _Transport(listener_ready=True, route_ready=True),
        probe,
    )
    elapsed_seconds = monotonic() - started_at
    probe.release.set()

    # Then: readiness fails closed before the documented 500ms caller deadline.
    assert probe.started.is_set()
    assert report is not None
    assert report.provider_capable is False
    assert elapsed_seconds < 0.5
