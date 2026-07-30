"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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


def test_deployment_validator_accepts_repository_artifacts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: the checked-in systemd bundle and sanitized sibling environment manifests.

    # When: the deployment acceptance validator scans the workspace.

    """函数契约说明.

    功能: 验证 deployment validator accepts
    repository artifacts 的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。 capsys:
    pytest.CaptureFixture[str]。 必填。
    契约: 同步调用。 返回 `None`。
    """

    exit_code, output = _run(monkeypatch, capsys, "--deployment-root", str(WORKSPACE))

    # Then: Mic and Sound are routed only through the configured Orchestrator.

    assert exit_code == 0

    assert "deployment topology accepted" in output


def test_deployment_validator_rejects_direct_peer_endpoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a fixture where Mic names a direct Sound endpoint.

    # When: the deployment acceptance validator scans the fixture.

    """函数契约说明.

    功能: 验证 deployment validator rejects
    direct peer endpoint 的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。 capsys:
    pytest.CaptureFixture[str]。 必填。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 deployment validator rejects
    mismatched stream identity
    的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。 capsys:
    pytest.CaptureFixture[str]。 必填。
    契约: 同步调用。 返回 `None`。
    """

    exit_code, output = _run(
        monkeypatch, capsys, "--deployment-root", str(MISMATCHED_IDENTITIES_DEPLOYMENT)
    )

    # Then: the shared routing identity is rejected before deployment.

    assert exit_code == 1

    assert "stream IDs differ" in output


def test_deployment_validator_accepts_sanitized_fixture(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a complete sanitized deployment fixture.

    # When: the deployment acceptance validator scans it.

    """函数契约说明.

    功能: 验证 deployment validator accepts
    sanitized fixture 的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。 capsys:
    pytest.CaptureFixture[str]。 必填。
    契约: 同步调用。 返回 `None`。
    """

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
    """函数契约说明.

    功能: 执行 _run 的同步逻辑,并协调 setattr,
    raises, run_path, str。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。 capsys:
    pytest.CaptureFixture[str]。 必填。
    *arguments: str。 必填。
    契约: 同步调用。 返回 `tuple[int, str]`。 可能抛出
    AssertionError。
    """

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])

    with pytest.raises(SystemExit) as result:
        _ = runpy.run_path(str(SCRIPT), run_name="__main__")

    match result.value.code:
        case int() as exit_code:
            return exit_code, capsys.readouterr().out

        case _:
            message = "topology verifier must exit with an integer status"

            raise AssertionError(message)
