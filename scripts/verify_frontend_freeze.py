#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]

WORKSPACE: Final = ROOT.parent

DEFAULT_FRONTEND: Final = WORKSPACE / "bitnp-raspberrygirl-vtuber-frontend"

DEFAULT_CERTIFICATE: Final = (
    WORKSPACE / ".omo/evidence/task-8-core-loop-before-frontend.json"
)

DEFAULT_PLAN: Final = WORKSPACE / ".omo/plans/core-loop-before-frontend.md"


sys.path.insert(0, str(ROOT / "src"))


from orchestrator.release_certificate import CertificateRequest, verify_certificate


@dataclass(frozen=True, slots=True)
class FreezeResult:

    accepted: bool

    code: str

    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FreezeRequest:

    frontend: Path

    baseline: str

    certificate: Path

    plan: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the pre-Task-8 Frontend freeze."
    )

    _ = parser.add_argument("--frontend-path", type=Path, default=DEFAULT_FRONTEND)

    _ = parser.add_argument("--baseline", default="HEAD")

    _ = parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)

    _ = parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)

    return parser.parse_args()


def _git(frontend: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=frontend, check=False, text=True, capture_output=True
    )


def _changed_paths(frontend: Path, baseline: str) -> tuple[str, ...] | None:
    baseline_check = _git(frontend, "rev-parse", "--verify", f"{baseline}^{{commit}}")

    if baseline_check.returncode != 0:
        return None

    tracked = _git(frontend, "diff", "--name-only", baseline).stdout.splitlines()

    untracked = _git(
        frontend, "ls-files", "--others", "--exclude-standard"
    ).stdout.splitlines()

    return tuple(sorted(set(tracked) | set(untracked)))


def verify(request: FreezeRequest) -> FreezeResult:
    paths = _changed_paths(request.frontend, request.baseline)

    if paths is None:
        return FreezeResult(False, "invalid_baseline", ())

    if not paths:
        return FreezeResult(True, "accepted", paths)

    certificate = verify_certificate(
        CertificateRequest(
            request.certificate,
            request.certificate.parent / "core-stability-run-1" / "manifest.json",
            request.certificate.parent / "core-stability-run-2" / "manifest.json",
            request.baseline,
            _sha256(request.plan),
            _source_digest(WORKSPACE),
        )
    )

    if not certificate.accepted:
        return FreezeResult(False, "invalid_certificate", paths)

    return FreezeResult(False, "local_certificate_not_authoritative", paths)


def main() -> int:
    args = parse_args()

    result = verify(
        FreezeRequest(
            args.frontend_path.resolve(),
            args.baseline,
            args.certificate.resolve(),
            args.plan.resolve(),
        )
    )

    print(
        json.dumps(
            {"accepted": result.accepted, "code": result.code, "paths": result.paths},
            sort_keys=True,
        )
    )

    return 0 if result.accepted else 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest(workspace: Path) -> str:
    digest = hashlib.sha256()

    ignored = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}

    for repository in (
        "bitnp-raspberrygirl-vtuber-orchestrator",
        "bitnp-raspberrygirl-vtuber-mic",
        "bitnp-raspberrygirl-vtuber-sound",
        "bitnp-raspberrygirl-vtuber-comments",
        "bitnp-raspberrygirl-vtuber-frontend",
    ):
        for path in sorted((workspace / repository).rglob("*")):
            if path.is_file() and not any(part in ignored for part in path.parts):
                digest.update(path.relative_to(workspace).as_posix().encode())

                digest.update(path.read_bytes())

    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
