from orchestrator.llm import MockLLMAdapter
from orchestrator.modes import ModePolicy
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    PipelineConfig,
    TTSDoneEvent,
)
from orchestrator.retrieval import RetrievalFixtureProvider


def test_interruption_cancels_previous_segment_and_rejects_stale_tts() -> None:
    # Given: an onsite explainer pipeline with one active turn.
    pipeline = OrchestratorTurnPipeline(
        adapters=PipelineAdapters(
            mode_policy=ModePolicy.onsite_explainer(),
            llm=MockLLMAdapter(answer_chunks=("First answer",)),
            retrieval=RetrievalFixtureProvider(refs=()),
        ),
        config=PipelineConfig(
            queue_capacity=2,
            turn_id_prefix="turn",
            segment_id_prefix="seg",
        ),
    )
    assert pipeline.accept_audience_input(
        ASRAudienceEvent(text="first", received_at_ms=10, segment_id="asr-1", seq=1),
    )
    first = pipeline.process_next_turn()
    assert first is not None

    # When: a second user input arrives before the previous segment finishes.
    assert pipeline.accept_audience_input(
        ASRAudienceEvent(text="second", received_at_ms=20, segment_id="asr-2", seq=2),
    )
    stale = pipeline.accept_tts_event(
        TTSDoneEvent(turn_id=first.turn_id, segment_id=first.segment_id),
    )
    second = pipeline.process_next_turn()

    # Then: previous downstream work is cancelled and stale events are ignored.
    assert stale is None
    assert [command.segment_id for command in pipeline.cancel_commands] == [
        "seg-0001",
    ] * 3
    assert [command.target for command in pipeline.cancel_commands] == [
        "tts",
        "sound",
        "vtuber",
    ]
    assert second is not None
    assert second.turn_id == "turn-0002"
    assert second.segment_id == "seg-0002"
