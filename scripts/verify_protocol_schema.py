#!/usr/bin/env python3

"""模块契约说明.

职责: 提供命令行脚本的参数处理、验证或运维流程。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

try:
    from protocol_schema_validation import ProtocolValidationError, validate_file

except ModuleNotFoundError as error:
    if error.name != "jsonschema":
        raise

    completed = subprocess.run(
        ["uv", "run", "python", str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )

    raise SystemExit(completed.returncode) from None


ROOT: Final = Path(__file__).resolve().parents[1]

SCHEMA_ROOT: Final = ROOT / "schemas" / "protocol"

VALID_FIXTURE: Final = ROOT / "schemas" / "fixtures" / "valid" / "protocol-events.json"

INVALID_FIXTURES: Final = tuple(
    sorted((ROOT / "schemas" / "fixtures" / "invalid").glob("*.json"))
)


def parse_args() -> argparse.Namespace:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `argparse.Namespace`。
    """
    parser = argparse.ArgumentParser(
        description="Validate canonical protocol schemas and fixtures."
    )

    _ = parser.add_argument("--expect-invalid", type=Path)

    return parser.parse_args()


def _report(errors: list[ProtocolValidationError]) -> None:
    """函数契约说明.

    功能: 执行 _report 的同步逻辑,并协调 print,
    dumps, as_json。
    参数: errors:
    list[ProtocolValidationError]。 必填。
    契约: 同步调用。 返回 `None`。
    """
    print(
        json.dumps(
            {"accepted": False, "errors": [error.as_json() for error in errors]},
            sort_keys=True,
        )
    )


def main() -> int:
    """函数契约说明.

    功能: 执行命令行或服务入口流程并返回进程级结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `int`。
    """
    args = parse_args()

    if args.expect_invalid is not None:
        errors = validate_file(args.expect_invalid.resolve(), SCHEMA_ROOT)

        if errors:
            _report(errors)

            return 0

        print(json.dumps({"accepted": True, "errors": []}, sort_keys=True))

        return 1

    valid_errors = validate_file(VALID_FIXTURE, SCHEMA_ROOT)

    invalid_failures = [
        path for path in INVALID_FIXTURES if not validate_file(path, SCHEMA_ROOT)
    ]

    if valid_errors or invalid_failures:
        _report(valid_errors)

        if invalid_failures:
            print(
                json.dumps(
                    {
                        "accepted": True,
                        "invalid_fixtures": [str(path) for path in invalid_failures],
                    },
                    sort_keys=True,
                )
            )

        return 1

    print(json.dumps({"accepted": True, "fixture": str(VALID_FIXTURE)}, sort_keys=True))

    print("protocol schema fixtures passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
