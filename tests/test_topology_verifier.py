"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]

WORKSPACE: Final = ROOT.parent

SCRIPT: Final = ROOT / "scripts" / "verify_topology.py"

MIC: Final = WORKSPACE / "bitnp-raspberrygirl-vtuber-mic"

COMMENTS: Final = WORKSPACE / "bitnp-raspberrygirl-vtuber-comments"

SOUND: Final = WORKSPACE / "bitnp-raspberrygirl-vtuber-sound"

FRONTEND: Final = WORKSPACE / "bitnp-raspberrygirl-vtuber-frontend"

BAD_FIXTURE: Final = ROOT / "tests" / "fixtures" / "topology" / "bad_peer_edge"

BAD_DOC_FIXTURE: Final = ROOT / "tests" / "fixtures" / "topology" / "bad_peer_doc"


def test_topology_verifier_accepts_explicit_sibling_paths(tmp_path: Path) -> None:
    # Given: all services and frontend are supplied as explicit sibling paths.

    # When: the verifier runs outside the parent repository.

    """函数契约说明.

    功能: 验证 topology verifier accepts
    explicit sibling paths 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--orchestrator-path",
            str(ROOT),
            "--mic-path",
            str(MIC),
            "--comments-path",
            str(COMMENTS),
            "--sound-path",
            str(SOUND),
            "--frontend-path",
            str(FRONTEND),
        ],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: the hub-and-spoke checkout passes.

    assert result.returncode == 0, result.stdout + result.stderr

    assert "0 direct non-orchestrator communication edges found" in result.stdout


def test_topology_verifier_preserves_bad_peer_fixture_mode(tmp_path: Path) -> None:
    # Given: the target-local fixture that connects Mic directly to Sound.

    # When: fixture mode scans it outside the parent repository.

    """函数契约说明.

    功能: 验证 topology verifier preserves
    bad peer fixture mode 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(BAD_FIXTURE)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: the forbidden direct edge remains rejected.

    assert result.returncode == 1

    assert "forbidden peer edge mic -> sound" in result.stdout


def test_topology_verifier_preserves_bad_document_fixture_mode(tmp_path: Path) -> None:
    # Given: a local documentation fixture with a direct peer edge.

    # When: fixture mode scans the documentation path.

    """函数契约说明.

    功能: 验证 topology verifier preserves
    bad document fixture mode
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(BAD_DOC_FIXTURE)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: the documented direct edge is rejected.

    assert result.returncode == 1

    assert "forbidden peer edge mic -> sound via documented edge" in result.stdout
