#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
ROOT: Final = Path(__file__).resolve().parents[1]
ENVELOPE_FIELDS: Final = ("schema_version", "event_type", "event_id", "source", "time", "trace_id", "session_id", "seq", "data")
TIMED_CUES: Final = frozenset({"vtuber.caption.command", "vtuber.expression.command", "vtuber.action.command"})
STREAM_STATES: Final = frozenset({"queued", "playing", "paused", "finished", "cancelled", "error"})
INVALID_FIXTURES: Final = tuple(sorted((ROOT / "schemas/fixtures/invalid").glob("*.json")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Orchestrator-owned protocol schemas and fixtures.")
    parser.add_argument("--expect-invalid", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> JsonValue:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_event(event: JsonValue, event_types: JsonObject) -> list[str]:
    if not isinstance(event, dict):
        return ["$: expected object"]
    errors = [f"$: missing required field {field}" for field in ENVELOPE_FIELDS if field not in event]
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or event_type not in event_types:
        return [*errors, "$.event_type: unknown event type"]
    data = event.get("data")
    if not isinstance(data, dict):
        return [*errors, "$.data: expected object"]
    definition = event_types[event_type]
    if not isinstance(definition, dict):
        return [*errors, "$.event_type: invalid event definition"]
    required = definition.get("required")
    if not isinstance(required, list) or not all(isinstance(field, str) for field in required):
        return [*errors, "$.event_type: invalid required fields"]
    errors.extend(f"$.data: missing required field {field}" for field in required if field not in data)
    if event_type in TIMED_CUES:
        errors.extend(validate_timed_cue(data))
    page = data.get("page")
    if event_type == "vtuber.scene.command" and (not isinstance(page, int) or page < 1):
        errors.append("$.data.page: expected integer at least 1")
    stream_start = data.get("start_rtp_timestamp")
    if event_type == "media.stream.command" and (not isinstance(stream_start, int) or stream_start < 0):
        errors.append("$.data.start_rtp_timestamp: expected non-negative integer")
    if event_type == "media.stream.state" and data.get("state") not in STREAM_STATES:
        errors.append("$.data.state: unknown stream state")
    return errors


def validate_timed_cue(data: JsonObject) -> list[str]:
    fields = ("start_at_ms", "end_at_ms", "start_rtp_timestamp", "end_rtp_timestamp")
    errors: list[str] = []
    for field in fields:
        value = data.get(field)
        if not isinstance(value, int) or value < 0:
            errors.append(f"$.data.{field}: expected non-negative integer")
    start_at = data.get("start_at_ms")
    end_at = data.get("end_at_ms")
    start_rtp = data.get("start_rtp_timestamp")
    end_rtp = data.get("end_rtp_timestamp")
    if isinstance(start_at, int) and isinstance(end_at, int) and end_at <= start_at:
        errors.append("$.data.end_at_ms: must be greater than start_at_ms")
    if isinstance(start_rtp, int) and isinstance(end_rtp, int) and end_rtp <= start_rtp:
        errors.append("$.data.end_rtp_timestamp: must be greater than start_rtp_timestamp")
    return errors


def validate_file(path: Path, event_types: JsonObject) -> list[str]:
    parsed = read_json(path)
    if isinstance(parsed, list):
        return [error for event in parsed for error in validate_event(event, event_types)]
    return validate_event(parsed, event_types)


def main() -> int:
    args = parse_args()
    schema = read_json(ROOT / "schemas/protocol/event-data.schema.json")
    if not isinstance(schema, dict):
        print("event-data schema does not define event types")
        return 1
    event_types = schema.get("event_types")
    if not isinstance(event_types, dict):
        print("event-data schema does not define event types")
        return 1
    if args.expect_invalid is not None:
        errors = validate_file(args.expect_invalid.resolve(), event_types)
        if errors:
            print(errors[0])
            return 0
        print("expected invalid fixture to fail")
        return 1
    valid_errors = validate_file(ROOT / "schemas/fixtures/valid/protocol-events.json", event_types)
    invalid_errors = [path for path in INVALID_FIXTURES if not validate_file(path, event_types)]
    if valid_errors or invalid_errors:
        print(*(valid_errors or [f"{path}: invalid fixture passed" for path in invalid_errors]), sep="\n")
        return 1
    print("protocol schema fixtures passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
