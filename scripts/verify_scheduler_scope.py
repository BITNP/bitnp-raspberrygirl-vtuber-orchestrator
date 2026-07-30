#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
PEER_LINK: Final = re.compile(r"\b(?:mic_to_sound|sound_to_mic)\b|\bmic\b\s*(?:->|-->|=>)\s*\bsound\b", re.IGNORECASE)
BIOMETRIC_AUTHORIZATION: Final = re.compile(r"(authorize.{0,40}voice|voice.{0,40}(authorize|payment|attendance))", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify scheduler scope guardrails.")
    _ = parser.add_argument("--root", type=Path, default=ROOT)
    _ = parser.add_argument("--forbid-peer-links", action="store_true")
    _ = parser.add_argument("--forbid-biometric-authorization", action="store_true")
    _ = parser.add_argument("--require-closed-command-validation", action="store_true")
    _ = parser.add_argument("--require-memory-provenance", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    source = root / "src" / "orchestrator"
    errors: list[str] = []
    if args.forbid_peer_links:
        errors.extend(scan(source, PEER_LINK, "peer link"))
    if args.forbid_biometric_authorization:
        errors.extend(scan(source, BIOMETRIC_AUTHORIZATION, "biometric authorization"))
    if args.require_closed_command_validation:
        errors.extend(required_tokens(source, ("SessionInteractionReducer", "ActionCapabilityRegistry", "TaskResultReducer"), "closed command validation"))
    if args.require_memory_provenance:
        errors.extend(required_tokens(source, ("MemoryProvenance", "MemoryProposal", "provenance"), "memory provenance"))
    write_evidence(root, errors)
    if errors:
        print(*errors, sep="\n")
        return 1
    print("scheduler scope accepted")
    return 0


def scan(source: Path, pattern: re.Pattern[str], label: str) -> list[str]:
    return [
        f"{label}: {path}:{number}"
        for path in source.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]


def required_tokens(source: Path, tokens: tuple[str, ...], label: str) -> list[str]:
    text = "\n".join(path.read_text(encoding="utf-8") for path in source.rglob("*.py"))
    return [f"{label}: missing {token}" for token in tokens if token not in text]


def write_evidence(root: Path, errors: list[str]) -> None:
    path = root.parent / ".omo" / "evidence" / "f4-scope.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps({"errors": errors, "passed": not errors}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    _ = main()
    raise SystemExit(_)
