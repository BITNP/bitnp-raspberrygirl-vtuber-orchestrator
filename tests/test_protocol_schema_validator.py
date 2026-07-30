
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Final

from orchestrator.json_boundary import JsonValue, parse_json_value

ROOT: Final = Path(__file__).resolve().parents[1]

SCRIPT: Final = ROOT / "scripts" / "verify_protocol_schema.py"

INVALID_CUE: Final = ROOT / "schemas/fixtures/invalid/cue_end_before_start.json"

INVALID_LEGACY: Final = ROOT / "schemas/fixtures/invalid/legacy_asr_event.json"

INVALID_RTP_CODEC: Final = (
    ROOT / "schemas/fixtures/invalid/rtp_codec_wrong_payload_type.json"
)


def test_protocol_validator_uses_orchestrator_owned_schema_paths(
    tmp_path: Path,
) -> None:
    # Given: an invocation outside the Orchestrator checkout.

    # When: the canonical protocol validator runs with its local defaults.


    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: valid local fixtures pass without a parent repository.

    assert result.returncode == 0, result.stdout + result.stderr

    assert "protocol schema fixtures passed" in result.stdout


def test_canonical_protocol_mode_fields_are_not_required_or_fixture_data() -> None:
    # Given: the canonical event schema and valid protocol fixture.

    schema = parse_json_value(
        (ROOT / "schemas/protocol/event-data.schema.json").read_text(
            encoding="utf-8"
        )
    )
    events = _valid_events()

    # When: the mode-bearing event definitions and fixture records are inspected.

    # Then: runtime mode is absent from canonical protocol data.

    assert isinstance(schema, dict)
    event_types = schema["event_types"]
    assert isinstance(event_types, dict)
    for event_type in ("llm.request", "session.created"):
        definition = event_types[event_type]
        assert isinstance(definition, dict)
        required = definition["required"]
        assert isinstance(required, list)
        assert "mode" not in required

    for event in events:
        assert isinstance(event, dict)
        data = event["data"]
        assert isinstance(data, dict)
        assert "mode" not in data


def test_protocol_validator_rejects_invalid_local_cue_and_legacy_event(
    tmp_path: Path,
) -> None:
    # Given: malformed cue and legacy standalone-ASR fixtures in the checkout.

    # When: each is supplied through the independent validator CLI.


    cue_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--expect-invalid", str(INVALID_CUE)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    legacy_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--expect-invalid", str(INVALID_LEGACY)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: both malformed artifacts are rejected.

    assert cue_result.returncode == 0, cue_result.stdout + cue_result.stderr

    assert legacy_result.returncode == 0, legacy_result.stdout + legacy_result.stderr


def test_protocol_validator_rejects_noncanonical_rtp_codec() -> None:
    # Given: a canonical source registration with a non-PT96 payload type.

    # When: the independent schema validator checks its fixture.


    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--expect-invalid", str(INVALID_RTP_CODEC)],
        check=False,
        text=True,
        capture_output=True,
    )

    # Then: rejection identifies the fixed RTP codec invariant.

    assert result.returncode == 0, result.stdout + result.stderr

    assert '"path": "$[0].data.codec.payload_type"' in result.stdout


def test_protocol_validator_rejects_closed_envelope_semantic_and_version_violations(
    tmp_path: Path,
) -> None:
    # Given: canonical events with one closed-envelope or stream semantic violation.


    valid_events = parse_json_value(
        (ROOT / "schemas/fixtures/valid/protocol-events.json").read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(valid_events, list)

    cases = {
        "unknown_envelope_field": _with_unknown_envelope_field(valid_events),
        "duplicate_event_id": _with_duplicate_event_id(valid_events),
        "sequence_regression": _with_sequence_regression(valid_events),
        "unsupported_minor_version": _with_unsupported_minor_version(valid_events),
        "missing_turn_correlation": _without_turn_correlation(valid_events),
    }

    # When: each fixture is supplied to the actual protocol verification surface.

    results = {
        name: _expect_invalid_fixture(tmp_path, name, payload)
        for name, payload in cases.items()
    }

    # Then: each rejection is machine-readable and names the violated invariant.

    assert results["unknown_envelope_field"].returncode == 0

    assert '"code": "schema_validation"' in results["unknown_envelope_field"].stdout

    assert results["duplicate_event_id"].returncode == 0

    assert '"code": "duplicate_event_id"' in results["duplicate_event_id"].stdout

    assert results["sequence_regression"].returncode == 0

    assert '"code": "sequence_regression"' in results["sequence_regression"].stdout

    assert results["unsupported_minor_version"].returncode == 0

    assert (
        '"code": "unsupported_schema_version"'
        in results["unsupported_minor_version"].stdout
    )

    assert results["missing_turn_correlation"].returncode == 0

    assert '"code": "missing_correlation"' in results["missing_turn_correlation"].stdout


def test_protocol_validator_reports_each_missing_segment_correlation_once(
    tmp_path: Path,
) -> None:
    # Given: a segment-correlated flush event missing both required correlation IDs.


    events = _valid_events()

    event = _event_copy(events, 15)

    _ = event.pop("turn_id")

    _ = event.pop("segment_id")

    # When: the collection is validated through the real CLI.

    result = _expect_invalid_fixture(tmp_path, "missing_segment_correlation", [event])

    output = parse_json_value(result.stdout)

    # Then: each absent correlation emits exactly one deterministic rejection.

    assert result.returncode == 0

    assert isinstance(output, dict)

    errors = output["errors"]

    assert isinstance(errors, list)

    paths = [error["path"] for error in errors if isinstance(error, dict)]

    assert paths == ["$[0].turn_id", "$[0].segment_id"]


def _valid_events() -> list[JsonValue]:

    events = parse_json_value(
        (ROOT / "schemas/fixtures/valid/protocol-events.json").read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(events, list)

    return events


def _expect_invalid_fixture(
    tmp_path: Path, name: str, payload: list[JsonValue]
) -> subprocess.CompletedProcess[str]:

    fixture = tmp_path / f"{name}.json"

    _ = fixture.write_text(json.dumps(payload), encoding="utf-8")

    return subprocess.run(
        [sys.executable, str(SCRIPT), "--expect-invalid", str(fixture)],
        check=False,
        text=True,
        capture_output=True,
    )


def _with_unknown_envelope_field(events: list[JsonValue]) -> list[JsonValue]:

    event = _event_copy(events, 0)

    event["unexpected"] = "closed"

    return [event]


def _with_duplicate_event_id(events: list[JsonValue]) -> list[JsonValue]:

    first = _event_copy(events, 0)

    second = _event_copy(events, 1)

    second["event_id"] = first["event_id"]

    return [first, second]


def _with_sequence_regression(events: list[JsonValue]) -> list[JsonValue]:

    first = _event_copy(events, 0)

    second = _event_copy(events, 1)

    second["seq"] = first["seq"]

    return [first, second]


def _with_unsupported_minor_version(events: list[JsonValue]) -> list[JsonValue]:

    event = _event_copy(events, 0)

    event["schema_version"] = "1.2.0"

    return [event]


def _without_turn_correlation(events: list[JsonValue]) -> list[JsonValue]:

    event = _event_copy(events, 2)

    _ = event.pop("turn_id")

    return [event]


def _event_copy(events: list[JsonValue], index: int) -> dict[str, JsonValue]:

    value = events[index]

    assert isinstance(value, dict)

    return dict(value)
