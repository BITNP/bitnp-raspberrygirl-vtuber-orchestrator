#!/usr/bin/env -S uv run --script

"""模块契约说明.

职责: 提供命令行脚本的参数处理、验证或运维流程。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""


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
    """类契约说明.

    职责: 保存 CliArgs 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: script、evidence。
    """

    script: Path

    evidence: Path


def main() -> int:
    """函数契约说明.

    功能: 执行命令行或服务入口流程并返回进程级结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `int`。
    """
    args = parse_args(sys.argv[1:])

    write_demo_evidence(args.script, args.evidence)

    print(f"LECTURER DEMO PASSED: {args.evidence}")

    return 0


def parse_args(argv: list[str]) -> CliArgs:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: argv: list[str]。 必填。
    契约: 同步调用。 返回 `CliArgs`。
    """
    parser = argparse.ArgumentParser(
        description="Run the local lecturer-mode protocol demo."
    )

    parser.add_argument("--script", required=True, type=Path)

    parser.add_argument("--evidence", required=True, type=Path)

    parsed = parser.parse_args(argv)

    return CliArgs(script=parsed.script, evidence=parsed.evidence)


if __name__ == "__main__":
    raise SystemExit(main())
