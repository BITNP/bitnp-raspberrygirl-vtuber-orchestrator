
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
