"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import re
from pathlib import Path
from typing import Final

from orchestrator.llm import MockLLMAdapter
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.pipeline import (
    AnswerPolicy,
    OrchestratorTurnPipeline,
    PipelineAdapters,
)
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    CommentAudienceEvent,
    PipelineConfig,
    TurnResult,
)
from orchestrator.retrieval import RetrievalFixtureProvider

from .e2e_replay_harness import ModuleEdge, ReplayHarness, ScenarioSummary

COMMENT_FIXTURE: Final = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bilibili_comments.jsonl"
)

COMMENT_FIELD_PATTERN: Final = re.compile(
    r'"(?P<key>platform|source|user|text|timestamp)"\s*:\s*"(?P<value>[^"]*)"',
)


def all_adaptive_scenarios() -> tuple[ScenarioSummary, ...]:
    """函数契约说明.

    功能: 执行 all_adaptive_scenarios 的同步逻辑,并协调
    _presentation_question,
    _interrupted_voice_question,
    _comment_question,
    _onsite_voice_question。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `tuple[ScenarioSummary,
    ...]`。
    """

    return (
        _presentation_question(),
        _interrupted_voice_question(),
        _comment_question(),
        _onsite_voice_question(),
    )


def negative_peer_harness() -> ReplayHarness:
    """函数契约说明.

    功能: 执行 negative_peer_harness
    的同步逻辑,并协调 _harness, submit,
    finish_turn, inject_edge。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `ReplayHarness`。
    """

    harness = _harness("negative_peer", AdaptiveAgentPolicy())

    harness.submit(ASRAudienceEvent("question", 1, "asr-neg", 1))

    _ = harness.finish_turn()

    harness.inject_edge(ModuleEdge("asr", "sound", "forbidden.peer"))

    return harness


def negative_stale_harness() -> tuple[ReplayHarness, TurnResult]:
    """函数契约说明.

    功能: 执行 negative_stale_harness
    的同步逻辑,并协调 _harness, submit,
    start_next_turn。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `tuple[ReplayHarness,
    TurnResult]`。
    """

    harness = _harness("negative_stale", AdaptiveAgentPolicy())

    harness.submit(ASRAudienceEvent("first", 1, "asr-1", 1))

    first = harness.start_next_turn()

    harness.submit(ASRAudienceEvent("second", 2, "asr-2", 2))

    return harness, first


def _presentation_question() -> ScenarioSummary:
    """函数契约说明.

    功能: 执行 _presentation_question
    的同步逻辑,并协调 _harness, submit,
    finish_turn, assert_no_peer_edges。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `ScenarioSummary`。
    """

    harness = _harness("presentation_question", AdaptiveAgentPolicy())

    harness.submit(ASRAudienceEvent("What is the takeaway?", 15_000, "asr-qa", 1))

    _ = harness.finish_turn()

    harness.assert_no_peer_edges()

    return harness.summary()


def _interrupted_voice_question() -> ScenarioSummary:
    """函数契约说明.

    功能: 执行 _interrupted_voice_question
    的同步逻辑,并协调 _harness, submit,
    start_next_turn,
    reject_stale_synthesis。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `ScenarioSummary`。
    """

    harness = _harness("interrupted_voice_question", AdaptiveAgentPolicy())

    harness.submit(ASRAudienceEvent("Please repeat", 1_000, "asr-int-1", 1))

    first = harness.start_next_turn()

    harness.submit(ASRAudienceEvent("Actually, define BitNet", 1_100, "asr-int-2", 2))

    harness.reject_stale_synthesis(first)

    _ = harness.finish_turn()

    harness.assert_no_peer_edges()

    return harness.summary()


def _comment_question() -> ScenarioSummary:
    """函数契约说明.

    功能: 执行 _comment_question
    的同步逻辑,并协调 _harness, submit,
    finish_turn, assert_no_peer_edges。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `ScenarioSummary`。
    """

    harness = _harness("comment_question", AdaptiveAgentPolicy())

    harness.submit(_fixture_comment())

    _ = harness.finish_turn()

    harness.assert_no_peer_edges()

    return harness.summary()


def _onsite_voice_question() -> ScenarioSummary:
    """函数契约说明.

    功能: 执行 _onsite_voice_question 的同步逻辑,并协调
    _harness, submit, finish_turn,
    assert_no_peer_edges。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `ScenarioSummary`。
    """

    harness = _harness("onsite_voice_question", AdaptiveAgentPolicy())

    harness.submit(ASRAudienceEvent("How does the demo work?", 2_000, "asr-onsite", 1))

    _ = harness.finish_turn()

    harness.assert_no_peer_edges()

    return harness.summary()


def _harness(name: str, mode_policy: AnswerPolicy) -> ReplayHarness:
    """函数契约说明.

    功能: 执行 _harness 的同步逻辑,并协调
    ReplayHarness,
    OrchestratorTurnPipeline,
    PipelineAdapters, PipelineConfig。
    参数: name: str。 必填。 mode_policy:
    AnswerPolicy。 必填。
    契约: 同步调用。 返回 `ReplayHarness`。
    """

    return ReplayHarness(
        name=name,
        pipeline=OrchestratorTurnPipeline(
            adapters=PipelineAdapters(
                mode_policy,
                MockLLMAdapter(("Deterministic ", "answer")),
                RetrievalFixtureProvider(()),
            ),
            config=PipelineConfig(4, "turn", "seg"),
        ),
    )


def _fixture_comment() -> CommentAudienceEvent:
    """函数契约说明.

    功能: 执行 _fixture_comment 的同步逻辑,并协调
    CommentAudienceEvent, splitlines,
    group, finditer。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `CommentAudienceEvent`。
    """

    text = COMMENT_FIXTURE.read_text(encoding="utf-8").splitlines()[0]

    fields = {
        match.group("key"): match.group("value")
        for match in COMMENT_FIELD_PATTERN.finditer(text)
    }

    return CommentAudienceEvent(
        fields["platform"],
        fields["source"],
        fields["user"],
        fields["text"],
        fields["timestamp"],
    )
