#!/usr/bin/env python3


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
    parser = argparse.ArgumentParser(
        description="Validate canonical protocol schemas and fixtures."
    )

    _ = parser.add_argument("--expect-invalid", type=Path)

    return parser.parse_args()


def _report(errors: list[ProtocolValidationError]) -> None:
    print(
        json.dumps(
            {"accepted": False, "errors": [error.as_json() for error in errors]},
            sort_keys=True,
        )
    )


def main() -> int:
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
