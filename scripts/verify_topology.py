#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Final

PEERS: Final = frozenset({"mic", "comments", "sound", "frontend"})
PARTS: Final = {"bitnp-raspberrygirl-vtuber-mic": "mic", "bitnp-raspberrygirl-vtuber-comments": "comments", "bitnp-raspberrygirl-vtuber-orchestrator": "orchestrator", "bitnp-raspberrygirl-vtuber-sound": "sound", "bitnp-raspberrygirl-vtuber-frontend": "frontend", "mic": "mic", "comments": "comments", "sound": "sound", "frontend": "frontend"}
SUFFIXES: Final = frozenset({".py", ".gd", ".toml", ".json", ".md", ".yaml", ".yml", ".cfg", ".ini", ".godot", ".tscn"})
SKIP: Final = frozenset({".git", ".venv", ".pytest_cache", "__pycache__", ".godot", ".omo"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify hub-and-spoke service topology.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--fixture", type=Path)
    target.add_argument("--sibling-root", type=Path)
    parser.add_argument("--orchestrator-path", type=Path)
    parser.add_argument("--mic-path", type=Path)
    parser.add_argument("--comments-path", type=Path)
    parser.add_argument("--sound-path", type=Path)
    parser.add_argument("--frontend-path", type=Path)
    return parser.parse_args()


def paths(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.fixture is not None:
        return (args.fixture.resolve(),)
    if args.sibling_root is not None:
        root = args.sibling_root.resolve()
        return tuple(root / f"bitnp-raspberrygirl-vtuber-{name}" for name in ("orchestrator", "mic", "comments", "sound", "frontend"))
    supplied = (args.orchestrator_path, args.mic_path, args.comments_path, args.sound_path, args.frontend_path)
    if all(path is None for path in supplied):
        raise SystemExit("supply --fixture, --sibling-root, or every explicit --*-path")
    if any(path is None for path in supplied):
        raise SystemExit("explicit mode requires every --*-path")
    return tuple(path.resolve() for path in supplied if path is not None)


def source_for(path: Path) -> str | None:
    return next((PARTS[part] for part in reversed(path.parts) if part in PARTS), None)


def edges(root: Path, include_tests: bool) -> list[str]:
    found: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES or SKIP.intersection(path.parts):
            continue
        if not include_tests and "tests" in path.parts:
            continue
        source = source_for(path)
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            lowered = line.lower()
            if path.suffix == ".md" and re.search(r"\b(mic|comments|sound|frontend)\b\s*(?:->|-->|=>|to)\s*\b(mic|comments|sound|frontend)\b", lowered):
                match = re.search(r"\b(mic|comments|sound|frontend)\b\s*(?:->|-->|=>|to)\s*\b(mic|comments|sound|frontend)\b", lowered)
                if match is not None and match.group(1) != match.group(2):
                    found.append(f"{path}:{number}: forbidden peer edge {match.group(1)} -> {match.group(2)} via documented edge")
            if source in PEERS and source is not None:
                for peer in PEERS - {source}:
                    if re.search(rf"(?:{peer}_url|//{peer}(?::|/|\b)|import\s+{peer}\b|from\s+{peer}\b)", lowered):
                        found.append(f"{path}:{number}: forbidden peer edge {source} -> {peer} via {peer}_url")
    return found


def main() -> int:
    args = parse_args()
    found = [edge for root in paths(args) for edge in edges(root, args.fixture is not None)]
    if found:
        print(*found, sep="\n")
        return 1
    print("0 direct non-orchestrator communication edges found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
