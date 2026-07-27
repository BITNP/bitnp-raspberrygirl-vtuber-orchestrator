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
    docs = DOC.read_text(encoding="utf-8")
    required_terms = (
        "讲解模式 Demo",
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
