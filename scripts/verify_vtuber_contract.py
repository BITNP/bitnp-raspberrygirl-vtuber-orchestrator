#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

JsonObject: TypeAlias = dict[str, JsonValue]

ROOT = Path(__file__).resolve().parents[1]
VALID_FIXTURE = ROOT / "schemas/fixtures/valid/protocol-events.json"
INVALID_FIXTURES = (
    ROOT / "schemas/fixtures/invalid/cue_end_before_start.json",
    ROOT / "schemas/fixtures/invalid/scene_page_zero.json",
)
FRONTEND_EVENT_TYPES = frozenset(
    {
        "vtuber.caption.command",
        "vtuber.action.command",
        "vtuber.scene.command",
        "presentation.load.command",
        "presentation.play.command",
        "presentation.navigate.command",
    }
)


@dataclass(frozen=True, slots=True)
class State:

    caption: str = ""

    action: str = "idle"

    scene: str = "stage_default"

    segments: frozenset[str] = frozenset()

    presentation_deck: str = ""

    presentation_page: int = 0

    presentation_playing: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the frontend contract without Godot."
    )

    _ = parser.add_argument("--frontend-path", type=Path, required=True)

    return parser.parse_args()


def as_object(value: JsonValue) -> JsonObject | None:
    return value if isinstance(value, dict) else None


def events(path: Path) -> list[JsonObject]:
    value: JsonValue = json.loads(path.read_text(encoding="utf-8"))

    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def apply(state: State, event: JsonObject) -> tuple[State, bool]:
    data = as_object(event.get("data"))

    event_type = event.get("event_type")

    if (
        event.get("source") != "orchestrator"
        or data is None
        or not isinstance(event_type, str)
    ):
        return state, False

    match event_type:
        case "session.created":
            return state, True

        case "vtuber.caption.command":
            text = data.get("text")

            return (
                (
                    State(
                        text,
                        state.action,
                        state.scene,
                        state.segments,
                    ),
                    True,
                )
                if isinstance(text, str) and text
                else (state, False)
            )

        case "vtuber.action.command":
            action = data.get("action")

            start_at_ms, end_at_ms = data.get("start_at_ms"), data.get("end_at_ms")

            return (
                (
                    State(state.caption, action, state.scene, state.segments),
                    True,
                )
                if isinstance(action, str)
                and action in {"act_cute", "emphasis", "hello"}
                and isinstance(start_at_ms, int)
                and isinstance(end_at_ms, int)
                and 0 <= start_at_ms < end_at_ms
                else (state, False)
            )

        case "vtuber.scene.command":
            scene, page = data.get("scene"), data.get("page")

            return (
                (
                    State(
                        state.caption,
                        state.action,
                        scene,
                        state.segments,
                        state.presentation_deck,
                        state.presentation_page,
                        state.presentation_playing,
                    ),
                    True,
                )
                if scene in {"stage_default", "lecture_slide_focus"}
                and isinstance(page, int)
                and page > 0
                else (state, False)
            )

        case "presentation.load.command":
            deck_id, page = data.get("deck_id"), data.get("page")

            return (
                (
                    State(
                        state.caption,
                        state.action,
                        state.scene,
                        state.segments,
                        deck_id,
                        page,
                        False,
                    ),
                    True,
                )
                if isinstance(deck_id, str)
                and deck_id
                and isinstance(page, int)
                and page > 0
                else (state, False)
            )

        case "presentation.play.command":
            deck_id, page = data.get("deck_id"), data.get("page")

            return (
                (
                    State(
                        state.caption,
                        state.action,
                        state.scene,
                        state.segments,
                        state.presentation_deck,
                        page,
                        True,
                    ),
                    True,
                )
                if deck_id == state.presentation_deck
                and isinstance(page, int)
                and page > 0
                else (state, False)
            )

        case "presentation.navigate.command":
            deck_id, page = data.get("deck_id"), data.get("page")

            return (
                (
                    State(
                        state.caption,
                        state.action,
                        state.scene,
                        state.segments,
                        state.presentation_deck,
                        page,
                        state.presentation_playing,
                    ),
                    True,
                )
                if deck_id == state.presentation_deck
                and isinstance(page, int)
                and page > 0
                else (state, False)
            )

        case _:
            return state, False


def main() -> int:
    frontend = parse_args().frontend_path.resolve()

    project = (frontend / "project.godot").read_text(encoding="utf-8")

    client = (frontend / "scripts/vtuber_control_client.gd").read_text(encoding="utf-8")

    if 'run/main_scene="res://main.tscn"' not in project:
        print("main.tscn is not the Frontend entry scene")

        return 1

    if (
        "run/orchestrator_ws_url" not in project
        or "application/run/orchestrator_ws_url" not in client
    ):
        print("Orchestrator WebSocket setting is missing")

        return 1

    if (
        "run/orchestrator_tls_ca_path" not in project
        or "application/run/orchestrator_tls_ca_path" not in client
    ):
        print("Orchestrator TLS CA setting is missing")

        return 1

    if any(
        setting in project
        for setting in (
            "asr_ws_url",
            "tts_ws_url",
            "comments_ws_url",
            "sound_ws_url",
            "mic_ws_url",
        )
    ):
        print("forbidden peer WebSocket setting")

        return 1

    if any(
        contract not in client
        for contract in (
            '"act_cute": Vector2(0.0, 0.0)',
            '"emphasis": Vector2(1.0, 0.0)',
            '"hello": Vector2(0.0, 1.0)',
            '"parameters/OneShot/request"',
            '"parameters/OneShot 2/request"',
        )
    ):
        print("Frontend AnimationTree action contract is missing")

        return 1

    state = State()

    for event in events(VALID_FIXTURE):
        if event.get("event_type") not in FRONTEND_EVENT_TYPES:
            continue

        state, accepted = apply(state, event)

        if not accepted:
            print("valid fixture rejected")

            return 1

    for fixture in INVALID_FIXTURES:
        invalid_events = events(fixture)
        if not invalid_events:
            value: JsonValue = json.loads(fixture.read_text(encoding="utf-8"))
            invalid_events = [value] if isinstance(value, dict) else []

        for event in invalid_events:
            _, accepted = apply(state, event)

            if accepted:
                print("invalid fixture accepted")

                return 1

    print("vtuber fallback contract passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
