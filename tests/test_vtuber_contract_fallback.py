
from __future__ import annotations

import shutil
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


def test_vtuber_fallback_requires_tls_ca_setting(tmp_path: Path) -> None:
    # Given: a frontend checkout without its configured Orchestrator TLS CA path.
    frontend = tmp_path / "frontend"
    _ = shutil.copytree(FRONTEND, frontend)
    project = frontend / "project.godot"
    _ = project.write_text(
        project.read_text(encoding="utf-8").replace(
            'run/orchestrator_tls_ca_path=""\n', ""
        ),
        encoding="utf-8",
    )

    # When: the fallback verifier checks the incomplete frontend contract.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--frontend-path", str(frontend)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: the missing TLS CA setting is rejected explicitly.
    assert result.returncode == 1
    assert "Orchestrator TLS CA setting is missing" in result.stdout
