
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[1]

WORKSPACE: Final = ROOT.parent

SCRIPT: Final = ROOT / "scripts" / "verify_topology.py"

GOOD_DEPLOYMENT: Final = ROOT / "tests" / "fixtures" / "deployment" / "good"

DIRECT_PEER_DEPLOYMENT: Final = (
    ROOT / "tests" / "fixtures" / "deployment" / "direct_peer"
)

MISMATCHED_IDENTITIES_DEPLOYMENT: Final = (
    ROOT / "tests" / "fixtures" / "deployment" / "mismatched_identities"
)

MISSING_CA_PATH_DEPLOYMENT: Final = (
    ROOT / "tests" / "fixtures" / "deployment" / "missing_ca_path"
)


def test_deployment_validator_accepts_repository_artifacts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: the checked-in systemd bundle and sanitized sibling environment manifests.

    # When: the deployment acceptance validator scans the workspace.


    exit_code, output = _run(monkeypatch, capsys, "--deployment-root", str(WORKSPACE))

    # Then: Mic and Sound are routed only through the configured Orchestrator.

    assert exit_code == 0

    assert "deployment topology accepted" in output


def test_deployment_validator_rejects_direct_peer_endpoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a fixture where Mic names a direct Sound endpoint.

    # When: the deployment acceptance validator scans the fixture.


    exit_code, output = _run(
        monkeypatch, capsys, "--deployment-root", str(DIRECT_PEER_DEPLOYMENT)
    )

    # Then: direct Mic-to-Sound configuration is rejected.

    assert exit_code == 1

    assert "direct peer endpoint" in output


def test_deployment_validator_rejects_mismatched_stream_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a fixture whose Mic and Sound stream IDs differ.

    # When: the deployment acceptance validator scans the fixture.


    exit_code, output = _run(
        monkeypatch, capsys, "--deployment-root", str(MISMATCHED_IDENTITIES_DEPLOYMENT)
    )

    # Then: the shared routing identity is rejected before deployment.

    assert exit_code == 1

    assert "stream IDs differ" in output


def test_deployment_validator_rejects_missing_ca_bundle_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a deployment fixture whose service environment examples omit the CA bundle.

    # When: the deployment acceptance validator scans the fixture.

    exit_code, output = _run(
        monkeypatch, capsys, "--deployment-root", str(MISSING_CA_PATH_DEPLOYMENT)
    )

    # Then: every service missing the shared CA bundle contract is rejected.

    assert exit_code == 1

    assert "ORCHESTRATOR_TLS_CA_PATH must name a PEM CA bundle" in output


def test_deployment_validator_accepts_sanitized_fixture(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a complete sanitized deployment fixture.

    # When: the deployment acceptance validator scans it.


    exit_code, output = _run(
        monkeypatch, capsys, "--deployment-root", str(GOOD_DEPLOYMENT)
    )

    # Then: matching Orchestrator endpoints and identities pass.

    assert exit_code == 0

    assert "deployment topology accepted" in output


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *arguments: str,
) -> tuple[int, str]:

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])

    with pytest.raises(SystemExit) as result:
        _ = runpy.run_path(str(SCRIPT), run_name="__main__")

    match result.value.code:
        case int() as exit_code:
            return exit_code, capsys.readouterr().out

        case _:
            message = "topology verifier must exit with an integer status"

            raise AssertionError(message)
