#!/usr/bin/env -S uv run --script



# /// script

# requires-python = ">=3.12"

# dependencies = []

# ///


# ─── How to run ───

# 1. Install uv (if not installed):

#      curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Run directly from the repository root:

#      python scripts/run_lecturer_demo.py --script samples/lecturer/bitnet_intro_zh.json --evidence .omo/evidence/lecturer-demo.json

# 3. Or make executable and run:

#      chmod +x scripts/run_lecturer_demo.py && ./scripts/run_lecturer_demo.py --script samples/lecturer/bitnet_intro_zh.json --evidence .omo/evidence/lecturer-demo.json

# ──────────────────

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from lecturer_demo_lib import write_demo_evidence


@dataclass(frozen=True, slots=True)
class CliArgs:

    script: Path

    evidence: Path


def main() -> int:
    args = parse_args(sys.argv[1:])

    write_demo_evidence(args.script, args.evidence)

    print(f"LECTURER DEMO PASSED: {args.evidence}")

    return 0


def parse_args(argv: list[str]) -> CliArgs:
    parser = argparse.ArgumentParser(description="Run the local lecture-script protocol demo.")

    parser.add_argument("--script", required=True, type=Path)

    parser.add_argument("--evidence", required=True, type=Path)

    parsed = parser.parse_args(argv)

    return CliArgs(script=parsed.script, evidence=parsed.evidence)


if __name__ == "__main__":
    raise SystemExit(main())
