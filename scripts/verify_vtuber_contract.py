#!/usr/bin/env python3

"""模块契约说明.

职责: 提供命令行脚本的参数处理、验证或运维流程。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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


@dataclass(frozen=True, slots=True)
class State:
    """类契约说明.

    职责: 保存 State 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: caption、action、scene、se
    gments、presentation_deck。
    """

    caption: str = ""

    action: str = "idle"

    scene: str = "stage_default"

    segments: frozenset[str] = frozenset()

    presentation_deck: str = ""

    presentation_page: int = 0

    presentation_playing: bool = False


def parse_args() -> argparse.Namespace:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `argparse.Namespace`。
    """
    parser = argparse.ArgumentParser(
        description="Verify the frontend contract without Godot."
    )

    parser.add_argument("--frontend-path", type=Path, required=True)

    return parser.parse_args()


def as_object(value: JsonValue) -> JsonObject | None:
    """函数契约说明.

    功能: 执行 as_object 的同步逻辑,并协调
    isinstance。
    参数: value: JsonValue。 必填。
    契约: 同步调用。 返回 `JsonObject | None`。
    """
    return value if isinstance(value, dict) else None


def events(path: Path) -> list[JsonObject]:
    """函数契约说明.

    功能: 执行 events 的同步逻辑,并协调 loads,
    read_text, isinstance。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `list[JsonObject]`。
    """
    value: JsonValue = json.loads(path.read_text(encoding="utf-8"))

    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def apply(state: State, event: JsonObject) -> tuple[State, bool]:
    """函数契约说明.

    功能: 执行 apply 的同步逻辑,并协调 as_object,
    get, isinstance, State。
    参数: state: State。 必填。 event:
    JsonObject。 必填。
    契约: 同步调用。 返回 `tuple[State, bool]`。
    """
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
            text, segment = data.get("text"), event.get("segment_id")

            return (
                (
                    State(
                        text,
                        state.action,
                        state.scene,
                        state.segments | {segment},
                    ),
                    True,
                )
                if isinstance(text, str)
                and text
                and isinstance(segment, str)
                and segment
                else (state, False)
            )

        case "vtuber.action.command":
            action, segment = data.get("action"), event.get("segment_id")

            return (
                (
                    State(state.caption, action, state.scene, state.segments),
                    True,
                )
                if isinstance(action, str)
                and action in {"idle", "breathe", "dance", "explain_point", "speak"}
                and segment in state.segments
                else (state, False)
            )

        case "vtuber.scene.command":
            scene = data.get("scene")

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
                if isinstance(scene, str) and scene
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
    """函数契约说明.

    功能: 执行命令行或服务入口流程并返回进程级结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `int`。
    """
    frontend = parse_args().frontend_path.resolve()

    project = (frontend / "project.godot").read_text(encoding="utf-8")

    client = (frontend / "scripts/vtuber_control_client.gd").read_text(encoding="utf-8")

    if (
        "run/orchestrator_ws_url" not in project
        or "application/run/orchestrator_ws_url" not in client
    ):
        print("Orchestrator WebSocket setting is missing")

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

    state = State()

    for event in events(frontend / "tests/fixtures/vtuber_control_commands.json"):
        state, accepted = apply(state, event)

        if not accepted:
            print("valid fixture rejected")

            return 1

    for event in events(frontend / "tests/fixtures/vtuber_control_invalid.json"):
        _, accepted = apply(state, event)

        if accepted:
            print("invalid fixture accepted")

            return 1

    print("vtuber fallback contract passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
