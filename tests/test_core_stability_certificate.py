from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final

from orchestrator.release_certificate import (
    CertificateRequest,
    verify_certificate,
    write_certificate,
)

if TYPE_CHECKING:
    import pytest

ROOT: Final = Path(__file__).resolve().parents[1]
FREEZE_GUARD: Final = ROOT / "scripts" / "verify_frontend_freeze.py"


def test_certificate_verification_requires_matching_manifest_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: two successful, deterministic run manifests and an approved frontend SHA.
    first = tmp_path / "run-1.json"
    monkeypatch.setenv("TASK8_CERTIFICATE_KEY", "test-authority")
    second = tmp_path / "run-2.json"
    _ = first.write_text('{"commands":["schema"],"passed":true}\n', encoding="utf-8")
    _ = second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    certificate = tmp_path / "certificate.json"

    # When: the Task 8 producer binds both manifests into its canonical certificate.
    request = CertificateRequest(
        certificate, first, second, "frontend-baseline", _plan_digest()
    )
    write_certificate(request)
    accepted = verify_certificate(request)
    _ = first.write_text('{"commands":["schema"],"passed":false}\n', encoding="utf-8")
    stale = verify_certificate(request)

    # Then: only the unmodified two-run evidence verifies.
    assert accepted.accepted is True
    assert stale.accepted is False
    assert stale.code == "manifest_digest_mismatch"


def test_freeze_guard_rejects_task9_diff_with_locally_verified_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a frontend baseline, a subsequent Task 9 edit, and valid Task 8 evidence.
    frontend, baseline = _frontend_checkout(tmp_path)
    monkeypatch.setenv("TASK8_CERTIFICATE_KEY", "test-authority")
    manifests = _write_manifests(tmp_path)
    certificate = tmp_path / "certificate.json"
    write_certificate(
        CertificateRequest(certificate, *manifests, baseline, _plan_digest())
    )
    _ = (frontend / "scene.tscn").write_text("task-9", encoding="utf-8")

    # When: admission evaluates the Task 8 certificate against the frontend baseline.
    result = subprocess.run(
        [
            sys.executable,
            str(FREEZE_GUARD),
            "--frontend-path",
            str(frontend),
            "--baseline",
            baseline,
            "--certificate",
            str(certificate),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: self-issued local evidence cannot authorize a dirty Frontend checkout.
    assert result.returncode == 1
    assert '"code": "invalid_certificate"' in result.stdout


def test_freeze_guard_rejects_fabricated_certificate_for_frontend_diff(
    tmp_path: Path,
) -> None:
    # Given: a dirty frontend and an arbitrary JSON payload posing as a certificate.
    frontend, baseline = _frontend_checkout(tmp_path)
    _ = (frontend / "scene.tscn").write_text("changed", encoding="utf-8")
    certificate = tmp_path / "certificate.json"
    _ = certificate.write_text(
        json.dumps({"signature": "fabricated"}), encoding="utf-8"
    )

    # When: Task 9 admission checks the fabricated certificate.
    result = subprocess.run(
        [
            sys.executable,
            str(FREEZE_GUARD),
            "--frontend-path",
            str(frontend),
            "--baseline",
            baseline,
            "--certificate",
            str(certificate),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: arbitrary signatures never authorize a frozen checkout.
    assert result.returncode == 1
    assert '"code": "invalid_certificate"' in result.stdout


def test_certificate_verification_fails_closed_without_authority_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a locally issued certificate whose authority key is unavailable.
    monkeypatch.setenv("TASK8_CERTIFICATE_KEY", "test-authority")
    first, second = _write_manifests(tmp_path)
    certificate = tmp_path / "certificate.json"
    request = CertificateRequest(certificate, first, second, "frontend", _plan_digest())
    write_certificate(request)
    monkeypatch.delenv("TASK8_CERTIFICATE_KEY")

    # When: verification runs without an external certificate authority key.
    result = verify_certificate(request)

    # Then: the unavailable authority fails closed.
    assert result.accepted is False
    assert result.code == "certificate_authority_unavailable"


def _write_manifests(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "core-stability-run-1" / "manifest.json"
    second = tmp_path / "core-stability-run-2" / "manifest.json"
    first.parent.mkdir()
    second.parent.mkdir()
    payload = '{"commands":["schema"],"passed":true}\n'
    _ = first.write_text(payload, encoding="utf-8")
    _ = second.write_text(payload, encoding="utf-8")
    return first, second


def _frontend_checkout(tmp_path: Path) -> tuple[Path, str]:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    _ = _git(frontend, "init")
    _ = (frontend / "scene.tscn").write_text("baseline", encoding="utf-8")
    _ = _git(frontend, "add", ".")
    _ = _git(
        frontend,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-m",
        "baseline",
    )
    return frontend, _git(frontend, "rev-parse", "HEAD").stdout.strip()


def _git(path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=path, check=False, text=True, capture_output=True
    )


def _plan_digest() -> str:
    plan = ROOT.parent / ".omo" / "plans" / "core-loop-before-frontend.md"
    return sha256(plan.read_bytes()).hexdigest()
