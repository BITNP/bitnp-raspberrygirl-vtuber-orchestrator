"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from orchestrator.ids import SegmentId, TurnId
from orchestrator.pipeline_contracts import (
    VtuberActionCommand,
    VtuberCaptionCommand,
    VtuberSceneCommand,
)


def test_vtuber_commands_accept_named_lecturer_actions_and_scene() -> None:
    # Given: a lecturer step that asks Godot to show a slide and play a named action.

    """函数契约说明.

    功能: 验证 vtuber commands accept named
    lecturer actions and scene
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    turn_id = TurnId("turn-demo-0001")

    segment_id = SegmentId("seg-demo-0001")

    # When: Orchestrator builds vtuber protocol commands for the step.

    caption = VtuberCaptionCommand(turn_id, segment_id, "什么是 BitNet")

    action = VtuberActionCommand(turn_id, segment_id, "explain_point")

    scene = VtuberSceneCommand(
        turn_id=turn_id,
        segment_id=segment_id,
        scene="lecture_slide_focus",
        slide_id="slide-01",
        slide_title="什么是 BitNet",
    )

    # Then: the protocol carries names for Godot to map later.

    assert caption.event_type == "vtuber.caption.command"

    assert action.event_type == "vtuber.action.command"

    assert action.action == "explain_point"

    assert scene.event_type == "vtuber.scene.command"

    assert scene.scene == "lecture_slide_focus"

    assert scene.slide_id == "slide-01"
