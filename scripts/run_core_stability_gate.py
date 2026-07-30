#!/usr/bin/env python3

"""模块契约说明.

职责: 提供命令行脚本的参数处理、验证或运维流程。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from orchestrator.release_certificate import (
    CertificateRequest,
    verify_certificate,
    write_certificate,
)

ROOT: Final = Path(__file__).resolve().parents[1]

WORKSPACE: Final = ROOT.parent

PLAN: Final = WORKSPACE / ".omo/plans/core-loop-before-frontend.md"

EVIDENCE: Final = WORKSPACE / ".omo/evidence"

REPOSITORIES: Final = (
    "bitnp-raspberrygirl-vtuber-orchestrator",
    "bitnp-raspberrygirl-vtuber-mic",
    "bitnp-raspberrygirl-vtuber-sound",
    "bitnp-raspberrygirl-vtuber-comments",
    "bitnp-raspberrygirl-vtuber-frontend",
)


@dataclass(frozen=True, slots=True)
class Command:
    """类契约说明.

    职责: 保存 Command 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: repository、arguments。
    """

    repository: str

    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateRequest:
    """类契约说明.

    职责: 保存 GateRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: workspace、evidence、plan、time
    out_seconds。
    """

    workspace: Path

    evidence: Path

    plan: Path

    timeout_seconds: int


def parse_args() -> argparse.Namespace:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `argparse.Namespace`。
    """
    parser = argparse.ArgumentParser(description="Run the Task 8 core stability gate.")

    _ = parser.add_argument("--workspace", type=Path, default=WORKSPACE)

    _ = parser.add_argument("--evidence", type=Path, default=EVIDENCE)

    _ = parser.add_argument("--plan", type=Path, default=PLAN)

    _ = parser.add_argument("--timeout-seconds", type=int, default=900)

    return parser.parse_args()


def run(request: GateRequest) -> int:
    """函数契约说明.

    功能: 运行流程并协调其依赖步骤。
    参数: request: GateRequest。 必填。
    契约: 同步调用。 返回 `int`。
    """
    certificate = request.evidence / "task-8-core-loop-before-frontend.json"

    certificate.unlink(missing_ok=True)

    baseline = _frontend_baseline(request.workspace)

    source_digest = _source_digest(request.workspace)

    try:
        first = _run_matrix(request, 1, baseline)

        second = _run_matrix(request, 2, baseline)

    except RuntimeError as error:
        print(str(error))

        return 1

    if _stable_manifest(first) != _stable_manifest(second):
        print("Task 8 failed: clean-run manifests differ")

        return 1

    certificate_request = CertificateRequest(
        certificate, first, second, baseline, _sha256(request.plan), source_digest
    )

    write_certificate(certificate_request)

    verification = verify_certificate(certificate_request)

    if not verification.accepted:
        certificate.unlink(missing_ok=True)

        print(f"Task 8 failed: certificate {verification.code}")

        return 1

    print(json.dumps({"certificate": str(certificate), "passed": True}, sort_keys=True))

    return 0


def _run_matrix(request: GateRequest, run_number: int, baseline: str) -> Path:
    """函数契约说明.

    功能: 执行 _run_matrix 的同步逻辑,并协调 exists,
    mkdir, write_text, rmtree。
    参数: request: GateRequest。 必填。
    run_number: int。 必填。 baseline: str。
    必填。
    契约: 同步调用。 返回 `Path`。 可能抛出
    RuntimeError。
    """
    artifact = request.evidence / f"core-stability-run-{run_number}"

    if artifact.exists():
        shutil.rmtree(artifact)

    _ = artifact.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix=f"task-8-run-{run_number}-") as temporary:
        sibling_root = Path(temporary) / "siblings"

        _copy_siblings(request.workspace, sibling_root)

        records = _execute_matrix(sibling_root, artifact, request.timeout_seconds)

    manifest = {
        "frontend_baseline": baseline,
        "passed": all(record["returncode"] == 0 for record in records),
        "records": records,
    }

    path = artifact / "manifest.json"

    _ = path.write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    if manifest["passed"] is not True:
        raise RuntimeError(f"Task 8 matrix {run_number} failed")

    return path


def _copy_siblings(source: Path, destination: Path) -> None:
    """函数契约说明.

    功能: 执行 _copy_siblings 的同步逻辑,并协调
    ignore_patterns, mkdir, copy2,
    copytree。
    参数: source: Path。 必填。 destination:
    Path。 必填。
    契约: 同步调用。 返回 `None`。
    """
    ignored = shutil.ignore_patterns(
        ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"
    )

    for repository in REPOSITORIES:
        _ = shutil.copytree(
            source / repository, destination / repository, ignore=ignored
        )

    plan_directory = destination / ".omo" / "plans"

    _ = plan_directory.mkdir(parents=True)

    _ = shutil.copy2(
        source / ".omo" / "plans" / "core-loop-before-frontend.md", plan_directory
    )


def _execute_matrix(
    root: Path, artifact: Path, timeout_seconds: int
) -> list[dict[str, str | int]]:
    """函数契约说明.

    功能: 执行 _execute_matrix 的同步逻辑,并协调
    enumerate, _commands, write_text,
    append。
    参数: root: Path。 必填。 artifact: Path。
    必填。 timeout_seconds: int。 必填。
    契约: 同步调用。 返回 `list[dict[str, str |
    int]]`。
    """
    records: list[dict[str, str | int]] = []

    for index, command in enumerate(_commands()):
        stdout = artifact / f"{index:02d}-{command.repository}.stdout.log"

        stderr = artifact / f"{index:02d}-{command.repository}.stderr.log"

        try:
            result = subprocess.run(
                command.arguments,
                cwd=root / command.repository,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )

            returncode = result.returncode

            output = result.stdout

            error = result.stderr

        except subprocess.TimeoutExpired as timeout:
            returncode = 124

            output = _timeout_text(timeout.stdout)

            error = _timeout_text(timeout.stderr)

        _ = stdout.write_text(output, encoding="utf-8")

        _ = stderr.write_text(error, encoding="utf-8")

        records.append(
            {
                "repository": command.repository,
                "arguments": "\u001f".join(command.arguments),
                "returncode": returncode,
                "stdout_path": stdout.name,
                "stderr_path": stderr.name,
                "stdout_raw_sha256": _sha256(stdout),
                "stderr_raw_sha256": _sha256(stderr),
                "stdout_sha256": _sha256_text(_stable_output(output, root)),
                "stderr_sha256": _sha256_text(_stable_output(error, root)),
            }
        )

        if returncode != 0:
            break

    return records


def _commands() -> tuple[Command, ...]:
    """函数契约说明.

    功能: 执行 _commands 的同步逻辑,并协调 Command。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `tuple[Command, ...]`。
    """
    orchestrator = "bitnp-raspberrygirl-vtuber-orchestrator"

    return (
        Command(orchestrator, ("uv", "sync", "--locked")),
        Command(orchestrator, ("uv", "run", "pytest")),
        Command(orchestrator, ("uv", "run", "basedpyright")),
        Command(orchestrator, ("uv", "run", "ruff", "check", "src", "tests")),
        Command(orchestrator, ("python", "scripts/verify_protocol_schema.py")),
        Command(
            orchestrator,
            ("python", "scripts/verify_topology.py", "--sibling-root", ".."),
        ),
        Command(
            orchestrator,
            (
                "python",
                "scripts/verify_vtuber_contract.py",
                "--frontend-path",
                "../bitnp-raspberrygirl-vtuber-frontend",
            ),
        ),
        Command(
            orchestrator,
            ("bash", "scripts/verify_workspace.sh", "--sibling-root", ".."),
        ),
        Command("bitnp-raspberrygirl-vtuber-mic", ("uv", "sync", "--locked")),
        Command("bitnp-raspberrygirl-vtuber-mic", ("uv", "run", "pytest")),
        Command("bitnp-raspberrygirl-vtuber-sound", ("uv", "sync", "--locked")),
        Command("bitnp-raspberrygirl-vtuber-sound", ("uv", "run", "pytest")),
        Command("bitnp-raspberrygirl-vtuber-comments", ("uv", "sync", "--locked")),
        Command("bitnp-raspberrygirl-vtuber-comments", ("uv", "run", "pytest")),
        Command(
            "bitnp-raspberrygirl-vtuber-frontend",
            (
                "godot",
                "--headless",
                "--path",
                ".",
                "--script",
                "res://tests/protocol_smoke.gd",
            ),
        ),
    )


def _frontend_baseline(workspace: Path) -> str:
    """函数契约说明.

    功能: 执行 _frontend_baseline 的同步逻辑,并协调
    run, strip。
    参数: workspace: Path。 必填。
    契约: 同步调用。 返回 `str`。
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace / "bitnp-raspberrygirl-vtuber-frontend",
        check=True,
        text=True,
        capture_output=True,
    )

    return result.stdout.strip()


def _source_digest(workspace: Path) -> str:
    """函数契约说明.

    功能: 执行 _source_digest 的同步逻辑,并协调
    sha256, hexdigest, sorted, rglob。
    参数: workspace: Path。 必填。
    契约: 同步调用。 返回 `str`。
    """
    digest = hashlib.sha256()

    ignored = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}

    for repository in REPOSITORIES:
        for path in sorted((workspace / repository).rglob("*")):
            if path.is_file() and not any(part in ignored for part in path.parts):
                digest.update(path.relative_to(workspace).as_posix().encode())

                digest.update(path.read_bytes())

    return digest.hexdigest()


def _sha256(path: Path) -> str:
    """函数契约说明.

    功能: 执行 _sha256 的同步逻辑,并协调 hexdigest,
    sha256, read_bytes。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `str`。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    """函数契约说明.

    功能: 执行 _sha256_text 的同步逻辑,并协调
    hexdigest, sha256, encode。
    参数: value: str。 必填。
    契约: 同步调用。 返回 `str`。
    """
    return hashlib.sha256(value.encode()).hexdigest()


def _stable_manifest(path: Path) -> str:
    """函数契约说明.

    功能: 执行 _stable_manifest 的同步逻辑,并协调
    loads, dumps, read_text。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `str`。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    for record in payload["records"]:
        del record["stdout_raw_sha256"]

        del record["stderr_raw_sha256"]

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _stable_output(value: str, root: Path) -> str:
    """函数契约说明.

    功能: 执行 _stable_output 的同步逻辑,并协调
    replace, sub, str。
    参数: value: str。 必填。 root: Path。 必填。
    契约: 同步调用。 返回 `str`。
    """
    root_normalized = value.replace(str(root), "<disposable-sibling-root>")

    return re.sub(r"\b\d+(?:\.\d+)?(?:ms|s)\b", "<duration>", root_normalized)


def _timeout_text(value: str | bytes | None) -> str:
    """函数契约说明.

    功能: 执行 _timeout_text 的同步逻辑,并协调
    decode。
    参数: value: str | bytes | None。 必填。
    契约: 同步调用。 返回 `str`。
    """
    match value:
        case str() as text:
            return text

        case bytes() as payload:
            return payload.decode(errors="replace")

        case None:
            return ""


if __name__ == "__main__":
    arguments = parse_args()

    raise SystemExit(
        run(
            GateRequest(
                arguments.workspace.resolve(),
                arguments.evidence.resolve(),
                arguments.plan.resolve(),
                arguments.timeout_seconds,
            )
        )
    )
