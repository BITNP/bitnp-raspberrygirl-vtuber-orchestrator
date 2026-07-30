#!/usr/bin/env python3

"""模块契约说明.

职责: 提供命令行脚本的参数处理、验证或运维流程。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 FreezeResult
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: accepted、code、paths。
    """

    accepted: bool

    code: str

    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FreezeRequest:
    """类契约说明.

    职责: 保存 FreezeRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    frontend、baseline、certificate、plan。
    """

    frontend: Path

    baseline: str

    certificate: Path

    plan: Path


def parse_args() -> argparse.Namespace:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `argparse.Namespace`。
    """
    parser = argparse.ArgumentParser(
        description="Verify the pre-Task-8 Frontend freeze."
    )

    _ = parser.add_argument("--frontend-path", type=Path, default=DEFAULT_FRONTEND)

    _ = parser.add_argument("--baseline", default="HEAD")

    _ = parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)

    _ = parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)

    return parser.parse_args()


def _git(frontend: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """函数契约说明.

    功能: 执行 _git 的同步逻辑,并协调 run。
    参数: frontend: Path。 必填。 *arguments:
    str。 必填。
    契约: 同步调用。 返回
    `subprocess.CompletedProcess[str]`。
    """
    return subprocess.run(
        ["git", *arguments], cwd=frontend, check=False, text=True, capture_output=True
    )


def _changed_paths(frontend: Path, baseline: str) -> tuple[str, ...] | None:
    """函数契约说明.

    功能: 执行 _changed_paths 的同步逻辑,并协调
    _git, splitlines, tuple, sorted。
    参数: frontend: Path。 必填。 baseline:
    str。 必填。
    契约: 同步调用。 返回 `tuple[str, ...] |
    None`。
    """
    baseline_check = _git(frontend, "rev-parse", "--verify", f"{baseline}^{{commit}}")

    if baseline_check.returncode != 0:
        return None

    tracked = _git(frontend, "diff", "--name-only", baseline).stdout.splitlines()

    untracked = _git(
        frontend, "ls-files", "--others", "--exclude-standard"
    ).stdout.splitlines()

    return tuple(sorted(set(tracked) | set(untracked)))


def verify(request: FreezeRequest) -> FreezeResult:
    """函数契约说明.

    功能: 校验相关输入、协议或运行时约束。
    参数: request: FreezeRequest。 必填。
    契约: 同步调用。 返回 `FreezeResult`。
    """
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
    """函数契约说明.

    功能: 执行命令行或服务入口流程并返回进程级结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `int`。
    """
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
    """函数契约说明.

    功能: 执行 _sha256 的同步逻辑,并协调 hexdigest,
    sha256, read_bytes。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `str`。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest(workspace: Path) -> str:
    """函数契约说明.

    功能: 执行 _source_digest 的同步逻辑,并协调
    sha256, hexdigest, sorted, rglob。
    参数: workspace: Path。 必填。
    契约: 同步调用。 返回 `str`。
    """
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
