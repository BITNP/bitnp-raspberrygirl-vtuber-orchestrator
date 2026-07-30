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

FRONTEND: Final = ROOT.parent / "bitnp-raspberrygirl-vtuber-frontend"

SCRIPT: Final = ROOT / "scripts" / "verify_vtuber_contract.py"


def test_vtuber_fallback_requires_explicit_frontend_path(tmp_path: Path) -> None:
    # Given: the immutable frontend sibling is supplied explicitly.

    # When: the fallback verifier runs outside the workspace root.

    """函数契约说明.

    功能: 验证 vtuber fallback requires
    explicit frontend path 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--frontend-path", str(FRONTEND)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: the frontend contract passes without parent-root discovery.

    assert result.returncode == 0, result.stdout + result.stderr

    assert "vtuber fallback contract passed" in result.stdout
