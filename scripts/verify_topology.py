#!/usr/bin/env python3


from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Final

PEERS: Final = frozenset({"mic", "comments", "sound", "frontend"})

PARTS: Final = {
    "bitnp-raspberrygirl-vtuber-mic": "mic",
    "bitnp-raspberrygirl-vtuber-comments": "comments",
    "bitnp-raspberrygirl-vtuber-orchestrator": "orchestrator",
    "bitnp-raspberrygirl-vtuber-sound": "sound",
    "bitnp-raspberrygirl-vtuber-frontend": "frontend",
    "mic": "mic",
    "comments": "comments",
    "sound": "sound",
    "frontend": "frontend",
}

SUFFIXES: Final = frozenset(
    {
        ".py",
        ".gd",
        ".toml",
        ".json",
        ".md",
        ".yaml",
        ".yml",
        ".cfg",
        ".ini",
        ".godot",
        ".tscn",
    }
)

SKIP: Final = frozenset(
    {".git", ".venv", ".pytest_cache", "__pycache__", ".godot", ".omo"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify hub-and-spoke service topology."
    )

    target = parser.add_mutually_exclusive_group()

    _ = target.add_argument("--fixture", type=Path)

    _ = target.add_argument("--sibling-root", type=Path)

    _ = target.add_argument("--deployment-root", type=Path)

    _ = parser.add_argument("--orchestrator-path", type=Path)

    _ = parser.add_argument("--mic-path", type=Path)

    _ = parser.add_argument("--comments-path", type=Path)

    _ = parser.add_argument("--sound-path", type=Path)

    _ = parser.add_argument("--frontend-path", type=Path)

    return parser.parse_args()


def paths(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.fixture is not None:
        return (args.fixture.resolve(),)

    if args.sibling_root is not None:
        root = args.sibling_root.resolve()

        return tuple(
            root / f"bitnp-raspberrygirl-vtuber-{name}"
            for name in ("orchestrator", "mic", "comments", "sound", "frontend")
        )

    supplied = (
        args.orchestrator_path,
        args.mic_path,
        args.comments_path,
        args.sound_path,
        args.frontend_path,
    )

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
        if (
            not path.is_file()
            or path.suffix not in SUFFIXES
            or SKIP.intersection(path.parts)
        ):
            continue

        if not include_tests and "tests" in path.parts:
            continue

        source = source_for(path)

        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            lowered = line.lower()

            if path.suffix == ".md" and re.search(
                r"\b(mic|comments|sound|frontend)\b\s*(?:->|-->|=>|to)\s*\b(mic|comments|sound|frontend)\b",
                lowered,
            ):
                match = re.search(
                    r"\b(mic|comments|sound|frontend)\b\s*(?:->|-->|=>|to)\s*\b(mic|comments|sound|frontend)\b",
                    lowered,
                )

                if match is not None and match.group(1) != match.group(2):
                    found.append(
                        f"{path}:{number}: forbidden peer edge {match.group(1)} -> {match.group(2)} via documented edge"
                    )

            if source in PEERS and source is not None:
                for peer in PEERS - {source}:
                    if re.search(
                        rf"(?:{peer}_url|//{peer}(?::|/|\b)|import\s+{peer}\b|from\s+{peer}\b)",
                        lowered,
                    ):
                        found.append(
                            f"{path}:{number}: forbidden peer edge {source} -> {peer} via {peer}_url"
                        )

    return found


def main() -> int:
    args = parse_args()

    if args.deployment_root is not None:
        found = deployment_errors(args.deployment_root.resolve())

        if found:
            print(*found, sep="\n")

            return 1

        print("deployment topology accepted")

        return 0

    found = [
        edge for root in paths(args) for edge in edges(root, args.fixture is not None)
    ]

    if found:
        print(*found, sep="\n")

        return 1

    print("0 direct non-orchestrator communication edges found")

    return 0


def deployment_errors(root: Path) -> list[str]:
    orchestrator, mic, sound = _deployment_service_paths(root)

    environments = {
        "orchestrator": _read_environment(orchestrator / ".env.example"),
        "mic": _read_environment(mic / ".env.example"),
        "sound": _read_environment(sound / ".env.example"),
    }

    found = _environment_errors(environments)

    found.extend(_systemd_errors(orchestrator / "deploy" / "systemd"))

    return found


def _deployment_service_paths(root: Path) -> tuple[Path, Path, Path]:
    workspace_orchestrator = root / "bitnp-raspberrygirl-vtuber-orchestrator"

    if workspace_orchestrator.exists():
        return (
            workspace_orchestrator,
            root / "bitnp-raspberrygirl-vtuber-mic",
            root / "bitnp-raspberrygirl-vtuber-sound",
        )

    return root / "orchestrator", root / "mic", root / "sound"


def _read_environment(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    values: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue

        key, value = line.split("=", 1)

        values[key.strip()] = value.strip()

    return values


def _environment_errors(environments: dict[str, dict[str, str]]) -> list[str]:
    orchestrator = environments["orchestrator"]

    mic = environments["mic"]

    sound = environments["sound"]

    host = orchestrator.get("ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST", "")

    control_port = orchestrator.get(
        "ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT", ""
    )

    rtp_port = orchestrator.get("ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT", "")

    expected_url = _control_url(host, control_port)

    found: list[str] = []

    for service, values, forbidden_prefix in (
        ("mic", mic, "SOUND_"),
        ("sound", sound, "MIC_"),
    ):
        for key in values:
            if key.startswith(forbidden_prefix):
                found.append(f"{service}: direct peer endpoint in {key}")

    if mic.get("ORCHESTRATOR_WS_URL") != expected_url:
        found.append(
            "mic: control endpoint must be the advertised Orchestrator WSS URL"
        )

    if sound.get("ORCHESTRATOR_WS_URL") != expected_url:
        found.append(
            "sound: control endpoint must be the advertised Orchestrator WSS URL"
        )

    if (
        mic.get("ORCHESTRATOR_RTP_HOST") != host
        or mic.get("ORCHESTRATOR_RTP_PORT") != rtp_port
    ):
        found.append(
            "mic: RTP endpoint must be the advertised Orchestrator RTP endpoint"
        )

    mic_session_id = mic.get("BITNP_SESSION_ID", "")

    sound_session_id = sound.get("SOUND_SESSION_ID", "")

    if mic_session_id == "" or mic_session_id != sound_session_id:
        found.append("Mic and Sound session IDs differ")

    mic_stream_id = mic.get("BITNP_MIC_RTP_STREAM_ID", "")

    sound_stream_id = sound.get("SOUND_RTP_STREAM_ID", "")

    if mic_stream_id == "" or mic_stream_id != sound_stream_id:
        found.append("Mic and Sound stream IDs differ")

    return found


def _control_url(host: str, port: str) -> str:
    if host == "" or port == "":
        return ""

    authority = host if port == "443" else f"{host}:{port}"

    return f"wss://{authority}/control"


def _systemd_errors(systemd_root: Path) -> list[str]:
    if not systemd_root.is_dir():
        return []

    expected = {
        "orchestrator-transport.service": "/etc/bitnp/orchestrator-transport.env",
        "mic-stream.service": "/etc/bitnp/mic-stream.env",
        "sound-receive.service": "/etc/bitnp/sound-receive.env",
    }

    found: list[str] = []

    for unit, environment_file in expected.items():
        path = systemd_root / unit

        if not path.is_file():
            found.append(f"systemd: missing {unit}")

            continue

        values = _read_systemd_values(path)

        if values.get("EnvironmentFile") != environment_file:
            found.append(f"systemd: {unit} must load {environment_file}")

    return found


def _read_systemd_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue

        key, value = line.split("=", 1)

        values[key.strip()] = value.strip()

    return values


if __name__ == "__main__":
    raise SystemExit(main())
