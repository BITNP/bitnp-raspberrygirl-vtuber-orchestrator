"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

from orchestrator.json_boundary import JsonValue, parse_json_value

ROOT: Final = Path(__file__).resolve().parents[1]

FIXTURES: Final = ROOT / "tests" / "fixtures" / "multimodal_benchmark"

BENCHMARK: Final = ROOT / "scripts" / "benchmark_multimodal.py"

PLAN_VERIFIER: Final = ROOT / "scripts" / "verify_plan_contracts.py"

SCOPE_VERIFIER: Final = ROOT / "scripts" / "verify_scheduler_scope.py"

FRONTEND_FREEZE_VERIFIER: Final = ROOT / "scripts" / "verify_frontend_freeze.py"

WORKSPACE_VERIFIER: Final = ROOT / "scripts" / "verify_workspace.sh"

PLAN: Final = ROOT.parent / ".omo" / "plans" / "multimodal-agent-scheduler.md"

GIT: Final = shutil.which("git")


def test_benchmark_emits_deterministic_release_report(tmp_path: Path) -> None:
    # Given: sanitized Chinese benchmark cases and a checked-in final-only baseline.

    """函数契约说明.

    功能: 验证 benchmark emits deterministic
    release report 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    report = tmp_path / "report.json"

    # When: the streaming candidate satisfies every configured release threshold.

    result = _run(
        BENCHMARK,
        "--fixtures",
        str(FIXTURES),
        "--baseline",
        str(FIXTURES / "baseline.json"),
        "--report",
        str(report),
        "--max-cer-regression-pp",
        "1.0",
        "--max-duplicate-turns",
        "0",
        "--require-p95-final-latency-improvement-percent",
        "20",
    )

    # Then: the stable report records identity, quality, latency, and shadow audit.

    assert result.returncode == 0, result.stderr

    payload = _read_json_object(report)

    assert payload["provider"] == "funasr"

    assert payload["model"] == "paraformer-zh-streaming"

    assert payload["config_version"] == "streaming-canary-v1"

    assert payload["corpus_version"] == "zh-synthetic-v1"

    assert payload["gates"] == {"breaches": [], "passed": True}

    memory_audit = payload["memory_audit"]

    assert isinstance(memory_audit, dict)

    assert memory_audit["shadow_mode"] is True

    assert payload["turns"] == {"duplicate": 0, "stale": 0, "total": 3}


def test_benchmark_rejects_quality_latency_and_duplicate_threshold_breaches(
    tmp_path: Path,
) -> None:
    # Given: three independent release-threshold breaches in a parseable corpus.

    """函数契约说明.

    功能: 验证 benchmark rejects quality
    latency and duplicate threshold
    breaches 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    fixtures = tmp_path / "fixtures"

    fixtures.mkdir()

    _write_json(fixtures / "baseline.json", _fixture_payload("baseline.json"))

    _write_json(fixtures / "cases.json", _failing_cases())

    # When: the benchmark evaluates the candidate against strict rollout gates.

    result = _run(
        BENCHMARK,
        "--fixtures",
        str(fixtures),
        "--baseline",
        str(fixtures / "baseline.json"),
        "--report",
        str(tmp_path / "failure.json"),
        "--max-cer-regression-pp",
        "1.0",
        "--max-duplicate-turns",
        "0",
        "--require-p95-final-latency-improvement-percent",
        "20",
    )

    # Then: failure is explicit rather than a fake successful report.

    assert result.returncode == 1

    assert "cer_regression_pp" in result.stdout

    assert "duplicate_turns" in result.stdout

    assert "p95_final_latency_improvement_percent" in result.stdout


def test_benchmark_rejects_sensitive_fixture_keys(tmp_path: Path) -> None:
    # Given: a fixture that would accidentally carry a prohibited biometric artifact.

    """函数契约说明.

    功能: 验证 benchmark rejects sensitive
    fixture keys 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    fixtures = tmp_path / "fixtures"

    fixtures.mkdir()

    _write_json(fixtures / "baseline.json", _fixture_payload("baseline.json"))

    _write_json(fixtures / "cases.json", {"voice_template": "forbidden"})

    # When: the release benchmark parses fixture input at its trust boundary.

    result = _run(
        BENCHMARK,
        "--fixtures",
        str(fixtures),
        "--baseline",
        str(fixtures / "baseline.json"),
        "--report",
        str(tmp_path / "report.json"),
    )

    # Then: it refuses sensitive data rather than writing a report.

    assert result.returncode == 1

    assert "prohibited sensitive fixture field" in result.stdout


def test_plan_verifier_rejects_non_chinese_prompt_requirement(tmp_path: Path) -> None:
    # Given: a plan whose source has a non-Chinese LLM system prompt.

    """函数契约说明.

    功能: 验证 plan verifier rejects non
    chinese prompt requirement
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    root = tmp_path / "root"

    source = root / "src" / "orchestrator"

    source.mkdir(parents=True)

    _ = (source / "prompt.py").write_text('SYSTEM_PROMPT = "answer in English"\n')

    plan = root / "plan.md"

    _ = plan.write_text("no raw Mic RTP to Sound\ntask snapshot validation\n")

    # When: the final-wave plan verifier enforces all requested contracts.

    result = _run(
        PLAN_VERIFIER,
        "--plan",
        str(plan),
        "--root",
        str(root),
        "--require-chinese-prompts",
        "--forbid-raw-mic-to-sound",
        "--require-task-snapshot-validation",
    )

    # Then: the unsupported prompt language blocks release.

    assert result.returncode == 1

    assert "Chinese LLM prompt" in result.stdout


def test_frontend_freeze_rejects_any_dirty_frontend_path(tmp_path: Path) -> None:
    # Given: an approved frontend baseline with a later source-path modification.

    """函数契约说明.

    功能: 验证 frontend freeze rejects any
    dirty frontend path 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    frontend = tmp_path / "frontend"

    frontend.mkdir()

    _ = _run_git(frontend, "init")

    source = frontend / "scene.tscn"

    _ = source.write_text("baseline", encoding="utf-8")

    _ = _run_git(frontend, "add", ".")

    _ = _run_git(
        frontend,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-m",
        "baseline",
    )

    baseline = _run_git(frontend, "rev-parse", "HEAD").stdout.strip()

    _ = source.write_text("changed", encoding="utf-8")

    # When: the pre-Task-8 release guard scans the frontend worktree.

    result = _run(
        FRONTEND_FREEZE_VERIFIER,
        "--frontend-path",
        str(frontend),
        "--baseline",
        baseline,
        "--certificate",
        str(tmp_path / "missing-certificate.json"),
    )

    # Then: missing Task 8 evidence rejects the changed frontend path.

    assert result.returncode == 1

    assert '"code": "invalid_certificate"' in result.stdout

    assert "scene.tscn" in result.stdout


def test_frontend_freeze_rejects_fabricated_signature(tmp_path: Path) -> None:
    # Given: a dirty checkout and a structurally convincing but unverified certificate.

    """函数契约说明.

    功能: 验证 frontend freeze rejects
    fabricated signature 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    frontend, baseline = _dirty_frontend_checkout(tmp_path)

    certificate = tmp_path / "task-8-core-loop-before-frontend.json"

    _write_json(
        certificate,
        {
            "plan": "core-loop-before-frontend",
            "task": 8,
            "frontend_baseline": baseline,
            "passed": True,
            "clean_run_manifests": ["one", "two"],
            "signature": "fabricated",
        },
    )

    # When: the freeze guard receives the fabricated authorization artifact.

    result = _run(
        FRONTEND_FREEZE_VERIFIER,
        "--frontend-path",
        str(frontend),
        "--baseline",
        baseline,
        "--certificate",
        str(certificate),
    )

    # Then: a fabricated digest contract cannot authorize the frozen checkout.

    assert result.returncode == 1

    assert '"code": "invalid_certificate"' in result.stdout


def test_scheduler_scope_verifier_rejects_biometric_authorization(
    tmp_path: Path,
) -> None:
    # Given: a scheduler source that attempts consequential authorization by voice.

    """函数契约说明.

    功能: 验证 scheduler scope verifier
    rejects biometric authorization
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    root = tmp_path / "root"

    source = root / "src" / "orchestrator"

    source.mkdir(parents=True)

    _ = (source / "identity.py").write_text(
        "def authorize_voice() -> bool:\n    return True\n"
    )

    # When: scope fidelity checks run against the isolated source tree.

    result = _run(
        SCOPE_VERIFIER,
        "--root",
        str(root),
        "--forbid-peer-links",
        "--forbid-biometric-authorization",
        "--require-closed-command-validation",
        "--require-memory-provenance",
    )

    # Then: the forbidden consequential use is surfaced as a release failure.

    assert result.returncode == 1

    assert "biometric authorization" in result.stdout


def test_scheduler_scope_verifier_rejects_peer_link_and_missing_controls(
    tmp_path: Path,
) -> None:
    # Given: a scope fixture that bypasses the Orchestrator and lacks reducer controls.

    """函数契约说明.

    功能: 验证 scheduler scope verifier
    rejects peer link and missing
    controls 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    root = tmp_path / "root"

    source = root / "src" / "orchestrator"

    source.mkdir(parents=True)

    _ = (source / "route.py").write_text(
        "def mic_to_sound() -> None:\n    return None\n"
    )

    # When: peer and closed-command/provenance requirements are enabled.

    result = _run(
        SCOPE_VERIFIER,
        "--root",
        str(root),
        "--forbid-peer-links",
        "--require-closed-command-validation",
        "--require-memory-provenance",
    )

    # Then: every absent scope control is a nonzero release failure.

    assert result.returncode == 1

    assert "peer link" in result.stdout

    assert "closed command validation" in result.stdout

    assert "memory provenance" in result.stdout


def _run(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """函数契约说明.

    功能: 执行 _run 的同步逻辑,并协调 run, str。
    参数: script: Path。 必填。 *arguments:
    str。 必填。
    契约: 同步调用。 返回
    `subprocess.CompletedProcess[str]`。
    """

    return subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False,
        text=True,
        capture_output=True,
    )


def _run_git(path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """函数契约说明.

    功能: 执行 _run_git 的同步逻辑,并协调 run。
    参数: path: Path。 必填。 *arguments: str。
    必填。
    契约: 同步调用。 返回
    `subprocess.CompletedProcess[str]`。
    """

    assert GIT is not None

    return subprocess.run(
        [GIT, *arguments],
        cwd=path,
        check=False,
        text=True,
        capture_output=True,
    )


def _dirty_frontend_checkout(tmp_path: Path) -> tuple[Path, str]:
    """函数契约说明.

    功能: 执行 _dirty_frontend_checkout
    的同步逻辑,并协调 mkdir, _run_git,
    write_text, strip。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `tuple[Path, str]`。
    """

    frontend = tmp_path / "frontend"

    frontend.mkdir()

    _ = _run_git(frontend, "init")

    source = frontend / "scene.tscn"

    _ = source.write_text("baseline", encoding="utf-8")

    _ = _run_git(frontend, "add", ".")

    _ = _run_git(
        frontend,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-m",
        "baseline",
    )

    baseline = _run_git(frontend, "rev-parse", "HEAD").stdout.strip()

    _ = source.write_text("changed", encoding="utf-8")

    return frontend, baseline


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    """函数契约说明.

    功能: 执行 _write_json 的同步逻辑,并协调
    write_text, dumps。
    参数: path: Path。 必填。 payload:
    dict[str, JsonValue]。 必填。
    契约: 同步调用。 返回 `None`。
    """

    _ = path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_payload(filename: str) -> dict[str, JsonValue]:
    """函数契约说明.

    功能: 执行 _fixture_payload 的同步逻辑,并协调
    _read_json_object。
    参数: filename: str。 必填。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """

    return _read_json_object(FIXTURES / filename)


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    """函数契约说明.

    功能: 执行 _read_json_object 的同步逻辑,并协调
    parse_json_value, isinstance,
    read_text。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """

    raw = parse_json_value(path.read_text(encoding="utf-8"))

    assert isinstance(raw, dict)

    return raw


def _failing_cases() -> dict[str, JsonValue]:
    """函数契约说明.

    功能: 执行 _failing_cases 的同步逻辑,并协调
    _fixture_payload, isinstance。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """

    payload = _fixture_payload("cases.json")

    cases = payload["cases"]

    assert isinstance(cases, list)

    first = cases[0]

    assert isinstance(first, dict)

    first["hypothesis"] = "错误"

    first["final_latency_ms"] = 300

    second = cases[1]

    assert isinstance(second, dict)

    second["turn_id"] = first["turn_id"]

    return payload
