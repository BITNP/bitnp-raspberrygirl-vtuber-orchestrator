#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]

CHINESE: Final = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify final-wave plan contracts.")

    _ = parser.add_argument("--plan", required=True, type=Path)

    _ = parser.add_argument("--root", type=Path, default=ROOT)

    _ = parser.add_argument("--require-chinese-prompts", action="store_true")

    _ = parser.add_argument("--forbid-raw-mic-to-sound", action="store_true")

    _ = parser.add_argument("--require-task-snapshot-validation", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = args.root.resolve()

    try:
        plan = workspace_path(args.plan).read_text(encoding="utf-8")

    except OSError as error:
        print(str(error))

        return 1

    errors: list[str] = []

    if args.require_chinese_prompts:
        errors.extend(chinese_prompt_errors(root))

    if args.forbid_raw_mic_to_sound:
        errors.extend(raw_mic_errors(plan, root))

    if args.require_task_snapshot_validation:
        errors.extend(task_snapshot_errors(plan, root))

    write_evidence(root, errors)

    if errors:
        print(*errors, sep="\n")

        return 1

    print("plan contracts accepted")

    return 0


def chinese_prompt_errors(root: Path) -> list[str]:
    prompt_files = tuple((root / "src").rglob("*prompt*.py"))

    if not prompt_files:
        return ["Chinese LLM prompt: no prompt source found"]

    return [
        f"Chinese LLM prompt missing: {path}"
        for path in prompt_files
        if CHINESE.search(path.read_text(encoding="utf-8")) is None
    ]


def raw_mic_errors(plan: str, root: Path) -> list[str]:
    errors: list[str] = []

    if "no raw mic rtp to sound" not in plan.lower():
        errors.append("plan: missing no raw Mic RTP to Sound guardrail")

    pattern = re.compile(r"raw.{0,80}mic.{0,80}sound", re.IGNORECASE)

    errors.extend(
        f"raw Mic RTP to Sound path: {path}:{number}"
        for path in (root / "src").rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    )

    return errors


def task_snapshot_errors(plan: str, root: Path) -> list[str]:
    errors: list[str] = []

    if "snapshot" not in plan.lower():
        errors.append("plan: missing task snapshot validation requirement")

    reducer = root / "src" / "orchestrator" / "task_reducer.py"

    if not reducer.is_file() or "STALE_DATA_SNAPSHOT" not in reducer.read_text(
        encoding="utf-8"
    ):
        errors.append("task snapshot validation: stale data snapshot rejection missing")

    return errors


def write_evidence(root: Path, errors: list[str]) -> None:
    path = root.parent / ".omo" / "evidence" / "f1-plan-compliance.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    _ = path.write_text(
        json.dumps({"errors": errors, "passed": not errors}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def workspace_path(path: Path) -> Path:
    if not path.is_absolute() and path.parts[:1] == (".omo",):
        return ROOT.parent / path

    return path


if __name__ == "__main__":
    _ = main()

    raise SystemExit(_)
