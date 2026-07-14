import re
from pathlib import Path
from typing import Final

from orchestrator.llm import MockLLMAdapter
from orchestrator.modes import (
    LecturerState,
    ModePolicy,
    QaWindow,
    ScriptStep,
    SlideStep,
)
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

ROOT: Final = Path(__file__).resolve().parents[3]
COMMENT_FIXTURE: Final = (
    ROOT / "comments" / "tests" / "fixtures" / "bilibili_comments.jsonl"
)
COMMENT_FIELD_PATTERN: Final = re.compile(
    r'"(?P<key>platform|source|user|text|timestamp)"\s*:\s*"(?P<value>[^"]*)"',
)


def all_mode_scenarios() -> tuple[ScenarioSummary, ...]:
    return (
        _lecturer_scheduled_qa(),
        _lecturer_interruption(),
        _virtual_streamer_comment_qa(),
        _onsite_asr_qa(),
    )


def negative_peer_harness() -> ReplayHarness:
    harness = _harness("negative_peer", ModePolicy.onsite_explainer())
    harness.submit(ASRAudienceEvent("question", 1, "asr-neg", 1))
    _ = harness.finish_turn()
    harness.inject_edge(ModuleEdge("asr", "tts", "forbidden.peer"))
    return harness


def negative_stale_harness() -> tuple[ReplayHarness, TurnResult]:
    harness = _harness("negative_stale", ModePolicy.onsite_explainer())
    harness.submit(ASRAudienceEvent("first", 1, "asr-1", 1))
    first = harness.start_next_turn()
    harness.submit(ASRAudienceEvent("second", 2, "asr-2", 2))
    return harness, first


def _lecturer_scheduled_qa() -> ScenarioSummary:
    harness = _harness(
        "lecturer_scheduled_qa",
        ModePolicy.lecturer(
            LecturerState(
                script_step=ScriptStep(6),
                slide_step=SlideStep(12),
                immediate_interruption_enabled=False,
                qa_window=QaWindow(10_000, 20_000),
            ),
        ),
    )
    harness.submit(ASRAudienceEvent("What is the takeaway?", 15_000, "asr-qa", 1))
    _ = harness.finish_turn()
    harness.assert_no_peer_edges()
    return harness.summary()


def _lecturer_interruption() -> ScenarioSummary:
    harness = _harness(
        "lecturer_interruption",
        ModePolicy.lecturer(
            LecturerState(
                script_step=ScriptStep(2),
                slide_step=SlideStep(5),
                immediate_interruption_enabled=True,
                qa_window=None,
            ),
        ),
    )
    harness.submit(ASRAudienceEvent("Please repeat", 1_000, "asr-int-1", 1))
    first = harness.start_next_turn()
    harness.submit(ASRAudienceEvent("Actually, define BitNet", 1_100, "asr-int-2", 2))
    harness.reject_stale_tts(first)
    _ = harness.finish_turn()
    harness.assert_no_peer_edges()
    return harness.summary()


def _virtual_streamer_comment_qa() -> ScenarioSummary:
    harness = _harness(
        "virtual_streamer_comment_qa",
        ModePolicy.virtual_streamer(topic="bitnet"),
    )
    harness.submit(_fixture_comment())
    _ = harness.finish_turn()
    harness.assert_no_peer_edges()
    return harness.summary()


def _onsite_asr_qa() -> ScenarioSummary:
    harness = _harness("onsite_asr_qa", ModePolicy.onsite_explainer())
    harness.submit(ASRAudienceEvent("How does the demo work?", 2_000, "asr-onsite", 1))
    _ = harness.finish_turn()
    harness.assert_no_peer_edges()
    return harness.summary()


def _harness(name: str, mode_policy: AnswerPolicy) -> ReplayHarness:
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
