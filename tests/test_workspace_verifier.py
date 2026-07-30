from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]

SCRIPT: Final = ROOT / "scripts" / "verify_workspace.sh"


def test_workspace_verifier_runs_contract_checks_without_frontend_freeze() -> None:
    # Given: the real sibling workspace and an invalid optional release baseline.

    # When: the workspace verifier runs its retained contract checks.
    result = subprocess.run(  # noqa: S603
        ["/usr/bin/bash", str(SCRIPT), "--sibling-root", str(ROOT.parent)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=os.environ | {"FRONTEND_FREEZE_BASELINE": "not-a-commit"},
    )

    # Then: the optional release input cannot affect the Contract gate.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "protocol schema fixtures passed" in result.stdout
