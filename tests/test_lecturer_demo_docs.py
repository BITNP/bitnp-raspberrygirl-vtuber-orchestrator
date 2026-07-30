"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]

DOC: Final = ROOT / "docs" / "lecturer_demo_zh.md"

DEMO_COMMAND: Final = (
    "python scripts/run_lecturer_demo.py --script "
    "samples/lecturer/bitnet_intro_zh.json --evidence "
    ".omo/evidence/lecturer-demo.json"
)

PEER_ARROW_RE: Final = re.compile(
    r"\b(?:mic|asr|comments|tts|sound|vtuber)\b\s*(?:->|-->|=>|to)\s*\b(?:mic|asr|comments|tts|sound|vtuber)\b",
    re.IGNORECASE,
)

LEGACY_EVENT_RE: Final = re.compile(
    r"\b(?:tts\.request|tts\.chunk|tts\.done|sound\.play\.command|sound\.play\.state)\b",
)


def test_chinese_demo_docs_cover_user_and_developer_surfaces() -> None:
    # Given: the Chinese lecturer demo documentation.

    """函数契约说明.

    功能: 验证 chinese demo docs cover user
    and developer surfaces 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    docs = DOC.read_text(encoding="utf-8")

    required_terms = (
        "讲稿演示 Demo",
        "快速开始",
        "讲稿格式",
        "通信协议",
        "调试日志",
        DEMO_COMMAND,
        "narration",
        "slide",
        "action",
        "scene",
        "Orchestrator",
        "OpenAI-compatible TTS 提供方",
        "不部署独立的 TTS 服务",
        "media.stream.command",
        "media.stream.state",
        "RTP",
        "vtuber.caption.command",
        "vtuber.expression.command",
        "vtuber.action.command",
        "vtuber.scene.command",
    )

    # When / Then: all required topics are documented without peer arrows.

    for term in required_terms:
        assert term in docs

    assert PEER_ARROW_RE.search(docs) is None

    assert LEGACY_EVENT_RE.search(docs) is None
