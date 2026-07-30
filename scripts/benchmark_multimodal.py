#!/usr/bin/env python3

"""模块契约说明.

职责: 提供命令行脚本的参数处理、验证或运维流程。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SENSITIVE_FIELDS: Final = frozenset(
    {
        "api_key",
        "audio",
        "biometric",
        "credential",
        "recording",
        "secret",
        "token",
        "voice_template",
    }
)

ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Case:
    """类契约说明.

    职责: 保存 Case 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reference、hypothesis、final_l
    atency_ms、turn_id、stale、memory_decis
    ion。
    """

    reference: str

    hypothesis: str

    final_latency_ms: int

    turn_id: str

    stale: bool

    memory_decision: str

    provenance_id: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """类契约说明.

    职责: 保存 Candidate
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: provider、model、config_versio
    n、corpus_version、cases。
    """

    provider: str

    model: str

    config_version: str

    corpus_version: str

    cases: tuple[Case, ...]


@dataclass(frozen=True, slots=True)
class Baseline:
    """类契约说明.

    职责: 保存 Baseline 不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    cer_percent、p95_final_latency_ms。
    """

    cer_percent: float

    p95_final_latency_ms: int


def parse_args() -> argparse.Namespace:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `argparse.Namespace`。
    """
    parser = argparse.ArgumentParser(
        description="Benchmark sanitized Chinese multimodal fixtures."
    )

    _ = parser.add_argument("--fixtures", required=True, type=Path)

    _ = parser.add_argument("--baseline", required=True, type=Path)

    _ = parser.add_argument("--report", required=True, type=Path)

    _ = parser.add_argument("--max-cer-regression-pp", type=float, default=0.0)

    _ = parser.add_argument("--max-duplicate-turns", type=int, default=0)

    _ = parser.add_argument(
        "--require-p95-final-latency-improvement-percent", type=float, default=0.0
    )

    return parser.parse_args()


def main() -> int:
    """函数契约说明.

    功能: 执行命令行或服务入口流程并返回进程级结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `int`。
    """
    args = parse_args()

    try:
        fixtures = workspace_path(args.fixtures)

        candidate = read_candidate(fixtures / "cases.json")

        baseline = read_baseline(workspace_path(args.baseline))

        report, breaches = evaluate(candidate, baseline, args)

    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error))

        return 1

    write_json(workspace_path(args.report), report)

    if breaches:
        print(*breaches, sep="\n")

        return 1

    print("multimodal benchmark passed")

    return 0


def read_candidate(path: Path) -> Candidate:
    """函数契约说明.

    功能: 执行 read_candidate 的同步逻辑,并协调
    read_object,
    reject_sensitive_fields, get, tuple。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `Candidate`。 可能抛出
    ValueError。
    """
    source = read_object(path)

    reject_sensitive_fields(source)

    cases_value = source.get("cases")

    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("cases: expected non-empty list")

    cases = tuple(
        parse_case(value, index) for index, value in enumerate(cases_value, 1)
    )

    return Candidate(
        provider=require_text(source, "provider"),
        model=require_text(source, "model"),
        config_version=require_text(source, "config_version"),
        corpus_version=require_text(source, "corpus_version"),
        cases=cases,
    )


def read_baseline(path: Path) -> Baseline:
    """函数契约说明.

    功能: 执行 read_baseline 的同步逻辑,并协调
    read_object,
    reject_sensitive_fields,
    require_object, Baseline。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `Baseline`。
    """
    source = read_object(path)

    reject_sensitive_fields(source)

    quality = require_object(source, "quality")

    latency = require_object(source, "latency")

    return Baseline(
        cer_percent=require_number(quality, "cer_percent"),
        p95_final_latency_ms=require_integer(latency, "p95_final_latency_ms"),
    )


def parse_case(value: object, index: int) -> Case:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: value: object。 必填。 index: int。
    必填。
    契约: 同步调用。 返回 `Case`。 可能抛出
    ValueError。
    """
    if not isinstance(value, dict):
        raise ValueError(f"cases[{index}]: expected object")

    memory = require_object(value, "memory_decision")

    decision = require_text(memory, "decision")

    if decision not in {"accepted", "rejected"}:
        raise ValueError(
            f"cases[{index}].memory_decision.decision: expected accepted or rejected"
        )

    return Case(
        reference=require_text(value, "reference"),
        hypothesis=require_text(value, "hypothesis"),
        final_latency_ms=require_nonnegative_integer(value, "final_latency_ms"),
        turn_id=require_text(value, "turn_id"),
        stale=require_boolean(value, "stale"),
        memory_decision=decision,
        provenance_id=require_text(memory, "provenance_id"),
    )


def evaluate(
    candidate: Candidate, baseline: Baseline, args: argparse.Namespace
) -> tuple[dict[str, object], list[str]]:
    """函数契约说明.

    功能: 执行 evaluate 的同步逻辑,并协调 sum,
    round, percentile95, Counter。
    参数: candidate: Candidate。 必填。
    baseline: Baseline。 必填。 args:
    argparse.Namespace。 必填。
    契约: 同步调用。 返回 `tuple[dict[str,
    object], list[str]]`。
    """
    total_characters = sum(len(case.reference) for case in candidate.cases)

    errors = sum(
        edit_distance(case.reference, case.hypothesis) for case in candidate.cases
    )

    cer_percent = round(errors * 100 / total_characters, 4)

    p95_latency = percentile95([case.final_latency_ms for case in candidate.cases])

    counts = Counter(case.turn_id for case in candidate.cases)

    duplicates = sum(count - 1 for count in counts.values() if count > 1)

    stale = sum(case.stale for case in candidate.cases)

    improvement = round(
        (baseline.p95_final_latency_ms - p95_latency)
        * 100
        / baseline.p95_final_latency_ms,
        4,
    )

    provenance_complete = sum(bool(case.provenance_id) for case in candidate.cases)

    breaches = threshold_breaches(
        cer_percent=cer_percent,
        baseline=baseline,
        duplicates=duplicates,
        improvement=improvement,
        args=args,
    )

    report: dict[str, object] = {
        "config_version": candidate.config_version,
        "corpus_version": candidate.corpus_version,
        "gates": {"breaches": breaches, "passed": not breaches},
        "latency": {
            "p95_final_latency_improvement_percent": improvement,
            "p95_final_latency_ms": p95_latency,
        },
        "memory_audit": {
            "accepted": sum(
                case.memory_decision == "accepted" for case in candidate.cases
            ),
            "provenance_complete": provenance_complete,
            "rejected": sum(
                case.memory_decision == "rejected" for case in candidate.cases
            ),
            "shadow_mode": True,
            "total": len(candidate.cases),
        },
        "model": candidate.model,
        "provider": candidate.provider,
        "quality": {
            "baseline_cer_percent": baseline.cer_percent,
            "cer_percent": cer_percent,
            "cer_regression_pp": round(cer_percent - baseline.cer_percent, 4),
        },
        "turns": {
            "duplicate": duplicates,
            "stale": stale,
            "total": len(candidate.cases),
        },
    }

    return report, breaches


def threshold_breaches(
    *,
    cer_percent: float,
    baseline: Baseline,
    duplicates: int,
    improvement: float,
    args: argparse.Namespace,
) -> list[str]:
    """函数契约说明.

    功能: 执行 threshold_breaches 的同步逻辑,并协调
    append。
    参数: cer_percent: float。 必填。
    baseline: Baseline。 必填。 duplicates:
    int。 必填。 improvement: float。 必填。
    args: argparse.Namespace。 必填。
    契约: 同步调用。 返回 `list[str]`。
    """
    breaches: list[str] = []

    regression = cer_percent - baseline.cer_percent

    if regression > args.max_cer_regression_pp:
        breaches.append(f"cer_regression_pp={regression:.4f}")

    if duplicates > args.max_duplicate_turns:
        breaches.append(f"duplicate_turns={duplicates}")

    if improvement < args.require_p95_final_latency_improvement_percent:
        breaches.append(f"p95_final_latency_improvement_percent={improvement:.4f}")

    return breaches


def edit_distance(reference: str, hypothesis: str) -> int:
    """函数契约说明.

    功能: 执行 edit_distance 的同步逻辑,并协调 list,
    enumerate, range, append。
    参数: reference: str。 必填。 hypothesis:
    str。 必填。
    契约: 同步调用。 返回 `int`。
    """
    previous = list(range(len(hypothesis) + 1))

    for reference_index, reference_character in enumerate(reference, 1):
        current = [reference_index]

        for hypothesis_index, hypothesis_character in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_character != hypothesis_character),
                )
            )

        previous = current

    return previous[-1]


def percentile95(values: list[int]) -> int:
    """函数契约说明.

    功能: 执行 percentile95 的同步逻辑,并协调
    sorted, ceil, len。
    参数: values: list[int]。 必填。
    契约: 同步调用。 返回 `int`。
    """
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def read_object(path: Path) -> dict[str, object]:
    """函数契约说明.

    功能: 执行 read_object 的同步逻辑,并协调 loads,
    read_text, isinstance, ValueError。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `dict[str, object]`。
    可能抛出 ValueError。
    """
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")

    return value


def reject_sensitive_fields(value: object) -> None:
    """函数契约说明.

    功能: 执行 reject_sensitive_fields
    的同步逻辑,并协调 isinstance, items, any,
    reject_sensitive_fields。
    参数: value: object。 必填。
    契约: 同步调用。 返回 `None`。 可能抛出
    ValueError。
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(term in key.lower() for term in SENSITIVE_FIELDS):
                raise ValueError(f"prohibited sensitive fixture field: {key}")

            reject_sensitive_fields(nested)

    elif isinstance(value, list):
        for nested in value:
            reject_sensitive_fields(nested)


def require_object(source: dict[str, object], field: str) -> dict[str, object]:
    """函数契约说明.

    功能: 执行 require_object 的同步逻辑,并协调 get,
    isinstance, ValueError。
    参数: source: dict[str, object]。 必填。
    field: str。 必填。
    契约: 同步调用。 返回 `dict[str, object]`。
    可能抛出 ValueError。
    """
    value = source.get(field)

    if not isinstance(value, dict):
        raise ValueError(f"{field}: expected object")

    return value


def require_text(source: dict[str, object], field: str) -> str:
    """函数契约说明.

    功能: 执行 require_text 的同步逻辑,并协调 get,
    ValueError, isinstance。
    参数: source: dict[str, object]。 必填。
    field: str。 必填。
    契约: 同步调用。 返回 `str`。 可能抛出 ValueError。
    """
    value = source.get(field)

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected non-empty string")

    return value


def require_integer(source: dict[str, object], field: str) -> int:
    """函数契约说明.

    功能: 执行 require_integer 的同步逻辑,并协调
    get, type, ValueError。
    参数: source: dict[str, object]。 必填。
    field: str。 必填。
    契约: 同步调用。 返回 `int`。 可能抛出 ValueError。
    """
    value = source.get(field)

    if type(value) is not int:
        raise ValueError(f"{field}: expected integer")

    return value


def require_nonnegative_integer(source: dict[str, object], field: str) -> int:
    """函数契约说明.

    功能: 执行 require_nonnegative_integer
    的同步逻辑,并协调 require_integer,
    ValueError。
    参数: source: dict[str, object]。 必填。
    field: str。 必填。
    契约: 同步调用。 返回 `int`。 可能抛出 ValueError。
    """
    value = require_integer(source, field)

    if value < 0:
        raise ValueError(f"{field}: expected non-negative integer")

    return value


def require_number(source: dict[str, object], field: str) -> float:
    """函数契约说明.

    功能: 执行 require_number 的同步逻辑,并协调 get,
    float, isinstance, ValueError。
    参数: source: dict[str, object]。 必填。
    field: str。 必填。
    契约: 同步调用。 返回 `float`。 可能抛出
    ValueError。
    """
    value = source.get(field)

    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field}: expected number")

    return float(value)


def require_boolean(source: dict[str, object], field: str) -> bool:
    """函数契约说明.

    功能: 执行 require_boolean 的同步逻辑,并协调
    get, isinstance, ValueError。
    参数: source: dict[str, object]。 必填。
    field: str。 必填。
    契约: 同步调用。 返回 `bool`。 可能抛出
    ValueError。
    """
    value = source.get(field)

    if not isinstance(value, bool):
        raise ValueError(f"{field}: expected boolean")

    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    """函数契约说明.

    功能: 执行 write_json 的同步逻辑,并协调 mkdir,
    write_text, dumps。
    参数: path: Path。 必填。 payload:
    dict[str, object]。 必填。
    契约: 同步调用。 返回 `None`。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def workspace_path(path: Path) -> Path:
    """函数契约说明.

    功能: 执行 workspace_path 的同步逻辑,并协调
    is_absolute。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `Path`。
    """
    if not path.is_absolute() and path.parts[:1] == (".omo",):
        return ROOT.parent / path

    return path


if __name__ == "__main__":
    _ = main()

    raise SystemExit(_)
