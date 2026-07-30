#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SENSITIVE_FIELDS: Final = frozenset(
    {"api_key", "audio", "biometric", "credential", "recording", "secret", "token", "voice_template"}
)
ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Case:
    reference: str
    hypothesis: str
    final_latency_ms: int
    turn_id: str
    stale: bool
    memory_decision: str
    provenance_id: str


@dataclass(frozen=True, slots=True)
class Candidate:
    provider: str
    model: str
    config_version: str
    corpus_version: str
    cases: tuple[Case, ...]


@dataclass(frozen=True, slots=True)
class Baseline:
    cer_percent: float
    p95_final_latency_ms: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sanitized Chinese multimodal fixtures.")
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
    source = read_object(path)
    reject_sensitive_fields(source)
    cases_value = source.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("cases: expected non-empty list")
    cases = tuple(parse_case(value, index) for index, value in enumerate(cases_value, 1))
    return Candidate(
        provider=require_text(source, "provider"),
        model=require_text(source, "model"),
        config_version=require_text(source, "config_version"),
        corpus_version=require_text(source, "corpus_version"),
        cases=cases,
    )


def read_baseline(path: Path) -> Baseline:
    source = read_object(path)
    reject_sensitive_fields(source)
    quality = require_object(source, "quality")
    latency = require_object(source, "latency")
    return Baseline(
        cer_percent=require_number(quality, "cer_percent"),
        p95_final_latency_ms=require_integer(latency, "p95_final_latency_ms"),
    )


def parse_case(value: object, index: int) -> Case:
    if not isinstance(value, dict):
        raise ValueError(f"cases[{index}]: expected object")
    memory = require_object(value, "memory_decision")
    decision = require_text(memory, "decision")
    if decision not in {"accepted", "rejected"}:
        raise ValueError(f"cases[{index}].memory_decision.decision: expected accepted or rejected")
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
    total_characters = sum(len(case.reference) for case in candidate.cases)
    errors = sum(edit_distance(case.reference, case.hypothesis) for case in candidate.cases)
    cer_percent = round(errors * 100 / total_characters, 4)
    p95_latency = percentile95([case.final_latency_ms for case in candidate.cases])
    counts = Counter(case.turn_id for case in candidate.cases)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    stale = sum(case.stale for case in candidate.cases)
    improvement = round(
        (baseline.p95_final_latency_ms - p95_latency) * 100 / baseline.p95_final_latency_ms,
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
            "accepted": sum(case.memory_decision == "accepted" for case in candidate.cases),
            "provenance_complete": provenance_complete,
            "rejected": sum(case.memory_decision == "rejected" for case in candidate.cases),
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
        "turns": {"duplicate": duplicates, "stale": stale, "total": len(candidate.cases)},
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
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def reject_sensitive_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(term in key.lower() for term in SENSITIVE_FIELDS):
                raise ValueError(f"prohibited sensitive fixture field: {key}")
            reject_sensitive_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_sensitive_fields(nested)


def require_object(source: dict[str, object], field: str) -> dict[str, object]:
    value = source.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field}: expected object")
    return value


def require_text(source: dict[str, object], field: str) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected non-empty string")
    return value


def require_integer(source: dict[str, object], field: str) -> int:
    value = source.get(field)
    if type(value) is not int:
        raise ValueError(f"{field}: expected integer")
    return value


def require_nonnegative_integer(source: dict[str, object], field: str) -> int:
    value = require_integer(source, field)
    if value < 0:
        raise ValueError(f"{field}: expected non-negative integer")
    return value


def require_number(source: dict[str, object], field: str) -> float:
    value = source.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field}: expected number")
    return float(value)


def require_boolean(source: dict[str, object], field: str) -> bool:
    value = source.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field}: expected boolean")
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def workspace_path(path: Path) -> Path:
    if not path.is_absolute() and path.parts[:1] == (".omo",):
        return ROOT.parent / path
    return path


if __name__ == "__main__":
    _ = main()
    raise SystemExit(_)
