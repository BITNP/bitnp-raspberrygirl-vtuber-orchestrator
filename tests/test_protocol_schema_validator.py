from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

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
    assert "$.data.codec.payload_type: expected 96" in result.stdout
