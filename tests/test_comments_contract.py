"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from orchestrator.modes import AudienceInput, AudienceSource, ModePolicy

COMMENT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "bilibili_comments.jsonl"
)

COMMENT_FIELD_PATTERN: Final = re.compile(
    r'"(?P<key>platform|source|user|text|timestamp)"\s*:\s*"(?P<value>[^"]*)"',
)


@dataclass(frozen=True, slots=True)
class FixtureComment:
    """类契约说明.

    职责: 保存 FixtureComment
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    platform、source、user、text、timestamp。
    """

    platform: str

    source: str

    user: str

    text: str

    timestamp: str


def test_orchestrator_contract_accepts_replay_comment_as_turn_candidate() -> None:
    # Given: a comments replay fixture with the contract fields Orchestrator needs.

    """函数契约说明.

    功能: 验证 orchestrator contract accepts
    replay comment as turn candidate
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    fixture_comment = _load_fixture_comment(COMMENT_FIXTURE)

    # When: Orchestrator receives the normalized comment payload.

    audience_input = AudienceInput(
        source=AudienceSource.COMMENT,
        text=fixture_comment.text,
        received_at_ms=_timestamp_ms(fixture_comment.timestamp),
    )

    candidate = ModePolicy.virtual_streamer(topic="bitnet").select_answer_candidate(
        (audience_input,),
    )

    # Then: the comment has the required contract fields and becomes a turn candidate.

    assert fixture_comment.platform == "bilibili"

    assert fixture_comment.source == "danmaku"

    assert fixture_comment.user == "alice"

    assert fixture_comment.text == "What is BitNet quantization?"

    assert fixture_comment.timestamp == "2026-07-08T00:00:01Z"

    assert candidate is not None

    assert candidate.input == audience_input

    assert candidate.reason == "virtual_streamer_comment_priority"


def _timestamp_ms(raw_timestamp: str) -> int:
    """函数契约说明.

    功能: 执行 _timestamp_ms 的同步逻辑,并协调
    fromisoformat, int, timestamp。
    参数: raw_timestamp: str。 必填。
    契约: 同步调用。 返回 `int`。
    """

    parsed = datetime.fromisoformat(raw_timestamp)

    return int(parsed.timestamp() * 1000)


def _load_fixture_comment(path: Path) -> FixtureComment:
    """函数契约说明.

    功能: 执行 _load_fixture_comment
    的同步逻辑,并协调 FixtureComment, group,
    finditer, splitlines。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `FixtureComment`。
    """

    fields = {
        match.group("key"): match.group("value")
        for match in COMMENT_FIELD_PATTERN.finditer(path.read_text().splitlines()[0])
    }

    return FixtureComment(
        platform=fields["platform"],
        source=fields["source"],
        user=fields["user"],
        text=fields["text"],
        timestamp=fields["timestamp"],
    )
