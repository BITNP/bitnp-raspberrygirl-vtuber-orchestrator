
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import ConfigParseError
from orchestrator.transport_config import load_transport_config_from_env


def test_transport_config_requires_wss_tls_and_token_outside_loopback() -> None:
    # Given: production transport settings without TLS and bearer-token material.


    environment = {
        "ORCHESTRATOR_CONTROL_BIND_HOST": "control-bind.example.test",
        "ORCHESTRATOR_CONTROL_BIND_PORT": "8443",
        "ORCHESTRATOR_RTP_BIND_HOST": "rtp-bind.example.test",
        "ORCHESTRATOR_RTP_BIND_PORT": "5004",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST": "orchestrator.example.test",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT": "443",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT": "5004",
    }

    # When: the transport configuration crosses the environment boundary.

    with pytest.raises(ConfigParseError) as error:
        _ = load_transport_config_from_env(environment)

    # Then: production cannot silently downgrade the control plane from WSS.

    assert error.value.field_name == "ORCHESTRATOR_MIC_CONTROL_TOKEN"


def test_transport_config_allows_explicit_loopback_ws_for_tests() -> None:
    # Given: an explicit local-loopback transport test mode.


    environment = {
        "ORCHESTRATOR_CONTROL_BIND_HOST": "127.0.0.1",
        "ORCHESTRATOR_CONTROL_BIND_PORT": "8765",
        "ORCHESTRATOR_RTP_BIND_HOST": "127.0.0.1",
        "ORCHESTRATOR_RTP_BIND_PORT": "5004",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST": "127.0.0.1",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT": "8765",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT": "5004",
        "ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS": "true",
    }

    # When: the typed configuration is loaded.

    config = load_transport_config_from_env(environment)

    # Then: only the explicit loopback path may use unsecured WS without TLS material.

    assert config.control_scheme == "ws"

    assert config.control_bind_host == "127.0.0.1"

    assert config.udp_bind_port == 5004

    assert config.tls_cert_path is None

    assert config.tls_key_path is None


def test_transport_config_requires_tls_paths_outside_loopback() -> None:
    # Given: a production transport with bearer-token material but no TLS paths.


    environment = {
        "ORCHESTRATOR_CONTROL_BIND_HOST": "control-bind.example.test",
        "ORCHESTRATOR_CONTROL_BIND_PORT": "8443",
        "ORCHESTRATOR_RTP_BIND_HOST": "rtp-bind.example.test",
        "ORCHESTRATOR_RTP_BIND_PORT": "5004",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST": "orchestrator.example.test",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT": "443",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT": "5004",
        **_role_tokens(),
    }

    # When: the typed configuration is loaded.

    with pytest.raises(ConfigParseError) as error:
        _ = load_transport_config_from_env(environment)

    # Then: production requires a certificate path before it may expose WSS.

    assert error.value.field_name == "ORCHESTRATOR_CONTROL_TLS_CERT_PATH"


def test_transport_config_rejects_ws_on_nonloopback_hosts() -> None:
    # Given: an explicit WS flag paired with a network-reachable advertised host.


    environment = {
        "ORCHESTRATOR_CONTROL_BIND_HOST": "127.0.0.1",
        "ORCHESTRATOR_CONTROL_BIND_PORT": "8765",
        "ORCHESTRATOR_RTP_BIND_HOST": "127.0.0.1",
        "ORCHESTRATOR_RTP_BIND_PORT": "5004",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST": "orchestrator.example.test",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT": "8765",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT": "5004",
        "ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS": "true",
    }

    # When: the typed configuration is loaded.

    with pytest.raises(ConfigParseError) as error:
        _ = load_transport_config_from_env(environment)

    # Then: the WS escape hatch cannot expose an insecure nonloopback endpoint.

    assert error.value.field_name == "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST"


def test_transport_config_exposes_deployable_wss_and_udp_endpoints() -> None:
    # Given: complete production transport environment values.


    environment = {
        "ORCHESTRATOR_CONTROL_BIND_HOST": "control-bind.example.test",
        "ORCHESTRATOR_CONTROL_BIND_PORT": "8443",
        "ORCHESTRATOR_RTP_BIND_HOST": "rtp-bind.example.test",
        "ORCHESTRATOR_RTP_BIND_PORT": "5004",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST": "orchestrator.example.test",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT": "443",
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT": "5004",
        "ORCHESTRATOR_CONTROL_TLS_CERT_PATH": "/run/secrets/control.crt",
        "ORCHESTRATOR_CONTROL_TLS_KEY_PATH": "/run/secrets/control.key",
        **_role_tokens(),
    }

    # When: the typed configuration is loaded.

    config = load_transport_config_from_env(environment)

    # Then: control remains WSS and the advertised UDP endpoint is explicit.

    assert config.control_scheme == "wss"

    assert config.advertised_host == "orchestrator.example.test"

    assert config.advertised_control_port == 443

    assert config.advertised_udp_port == 5004

    assert config.tls_cert_path == Path("/run/secrets/control.crt")

    assert config.tls_key_path == Path("/run/secrets/control.key")


def test_transport_config_rejects_duplicate_role_tokens() -> None:
    environment = {
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST": "orchestrator.example.test",
        "ORCHESTRATOR_CONTROL_TLS_CERT_PATH": "/run/secrets/control.crt",
        "ORCHESTRATOR_CONTROL_TLS_KEY_PATH": "/run/secrets/control.key",
        **_role_tokens(),
        "ORCHESTRATOR_SOUND_CONTROL_TOKEN": "mic-role-token",
    }

    with pytest.raises(ConfigParseError) as error:
        _ = load_transport_config_from_env(environment)

    assert error.value.field_name == "ORCHESTRATOR_*_CONTROL_TOKEN"


def _role_tokens() -> dict[str, str]:
    return {
        "ORCHESTRATOR_MIC_CONTROL_TOKEN": "mic-role-token",
        "ORCHESTRATOR_SOUND_CONTROL_TOKEN": "sound-role-token",
        "ORCHESTRATOR_COMMENTS_CONTROL_TOKEN": "comments-role-token",
        "ORCHESTRATOR_FRONTEND_CONTROL_TOKEN": "frontend-role-token",
        "ORCHESTRATOR_OPERATOR_CONTROL_TOKEN": "operator-role-token",
    }
