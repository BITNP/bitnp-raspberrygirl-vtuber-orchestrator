
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from orchestrator.lecturer_script import LectureScriptError, parse_lecture_script

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_lecture_script_returns_typed_steps_when_valid(tmp_path: Path) -> None:
    # Given: a user-authored lecture script with narration and frontend cues.


    script_path = tmp_path / "lecture.json"

    _ = script_path.write_text(
        json.dumps(
            {
                "title": "BitNet 入门演示",
                "voice": "raspberry-default",
                "steps": [
                    {
                        "id": "intro",
                        "narration": "大家好,今天我们用一页幻灯片认识 BitNet。",
                        "slide": {
                            "id": "slide-01",
                            "title": "什么是 BitNet",
                            "page": 1,
                        },
                        "expression": "smile",
                        "action": "explain_point",
                        "scene": "lecture_slide_focus",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # When: Orchestrator parses the script at the file boundary.

    lecture = parse_lecture_script(script_path)

    # Then: typed lecture data preserves the user-authored commands.

    assert lecture.title == "BitNet 入门演示"

    assert lecture.voice == "raspberry-default"

    assert lecture.steps[0].narration == "大家好,今天我们用一页幻灯片认识 BitNet。"

    assert lecture.steps[0].slide.id == "slide-01"

    assert lecture.steps[0].slide.page == 1

    assert lecture.steps[0].expression == "smile"

    assert lecture.steps[0].action == "explain_point"

    assert lecture.steps[0].scene == "lecture_slide_focus"


def test_parse_lecture_script_rejects_missing_narration(tmp_path: Path) -> None:
    # Given: a malformed script step without the required narration text.


    script_path = tmp_path / "bad-lecture.json"

    _ = script_path.write_text(
        json.dumps(
            {
                "title": "坏讲稿",
                "steps": [
                    {
                        "id": "intro",
                        "slide": {"id": "slide-01", "title": "缺少讲解词", "page": 1},
                        "expression": "smile",
                        "action": "explain_point",
                        "scene": "lecture_slide_focus",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # When / Then: parsing fails before any downstream protocol command exists.

    with pytest.raises(LectureScriptError, match=r"steps\[0\].narration"):
        _ = parse_lecture_script(script_path)


def test_parse_lecture_script_rejects_missing_expression(tmp_path: Path) -> None:
    # Given: a step without the required frontend expression cue.


    script_path = tmp_path / "missing-expression-lecture.json"

    _ = script_path.write_text(
        json.dumps(
            {
                "title": "缺表情讲稿",
                "steps": [
                    {
                        "id": "intro",
                        "narration": "这一步缺少表情。",
                        "slide": {"id": "slide-01", "title": "表情", "page": 1},
                        "action": "explain_point",
                        "scene": "lecture_slide_focus",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # When / Then: the parser rejects incomplete frontend control at the boundary.

    with pytest.raises(LectureScriptError, match=r"steps\[0\].expression"):
        _ = parse_lecture_script(script_path)


def test_lecture_script_rejects_zero_slide_page_at_boundary(tmp_path: Path) -> None:
    # Given: a structurally complete lecture step with an invalid zero page.


    script_path = tmp_path / "zero-page-lecture.json"

    _ = script_path.write_text(
        json.dumps(
            {
                "title": "坏页码讲稿",
                "steps": [
                    {
                        "id": "intro",
                        "narration": "这一页不应被接受。",
                        "slide": {"id": "slide-00", "title": "零页", "page": 0},
                        "expression": "smile",
                        "action": "explain_point",
                        "scene": "lecture_slide_focus",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # When / Then: page validation rejects the script before pipeline commands exist.

    with pytest.raises(LectureScriptError, match=r"steps\[0\].slide.page"):
        _ = parse_lecture_script(script_path)
