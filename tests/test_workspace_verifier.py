from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]

SCRIPT: Final = ROOT / "scripts" / "verify_workspace.sh"


def test_workspace_verifier_runs_contract_checks() -> None:
    # Given: the real sibling workspace.

    # When: the workspace verifier runs its retained contract checks.
    result = subprocess.run(  # noqa: S603
        ["/usr/bin/bash", str(SCRIPT), "--sibling-root", str(ROOT.parent)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: the composed Contract gate succeeds.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "protocol schema fixtures passed" in result.stdout
