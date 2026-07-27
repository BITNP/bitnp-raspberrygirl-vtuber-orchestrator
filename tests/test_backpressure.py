from orchestrator.llm import FallbackLLMAdapter, MockLLMAdapter, TimeoutLLMAdapter
from orchestrator.modes import ModePolicy
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import ASRAudienceEvent, PipelineConfig
from orchestrator.retrieval import RetrievalFixtureProvider


def test_bounded_queue_rejects_overflow_without_sleeping() -> None:
    # Given: a pipeline with capacity for one pending audience input.
    pipeline = OrchestratorTurnPipeline(
        adapters=PipelineAdapters(
            mode_policy=ModePolicy.onsite_explainer(),
            llm=MockLLMAdapter(answer_chunks=("ok",)),
            retrieval=RetrievalFixtureProvider(refs=()),
        ),
        config=PipelineConfig(
            queue_capacity=1,
            turn_id_prefix="turn",
            segment_id_prefix="seg",
        ),
    )

    # When: two inputs are submitted before the queue is drained.
    first = pipeline.accept_audience_input(
        ASRAudienceEvent(text="first", received_at_ms=1, segment_id="asr-1", seq=1),
    )
    second = pipeline.accept_audience_input(
        ASRAudienceEvent(text="second", received_at_ms=2, segment_id="asr-2", seq=2),
    )

    # Then: backpressure rejects the overflow deterministically.
    assert first is True
    assert second is False
    assert pipeline.rejections == ("queue_full",)


def test_llm_timeout_emits_fallback_answer_without_wall_clock() -> None:
    # Given: the provider times out and the adapter has deterministic fallback text.
    pipeline = OrchestratorTurnPipeline(
        adapters=PipelineAdapters(
            mode_policy=ModePolicy.onsite_explainer(),
            llm=FallbackLLMAdapter(
                primary=TimeoutLLMAdapter(timeout_reason="deadline"),
                fallback_text="Fallback answer.",
            ),
            retrieval=RetrievalFixtureProvider(refs=()),
        ),
        config=PipelineConfig(
            queue_capacity=1,
            turn_id_prefix="turn",
            segment_id_prefix="seg",
        ),
    )
    assert pipeline.accept_audience_input(
        ASRAudienceEvent(text="where", received_at_ms=1, segment_id="asr-1", seq=1),
    )

    # When: the turn is processed.
    turn = pipeline.process_next_turn()

    # Then: fallback text is retained and timeout cancels pending media work.
    assert turn is not None
    assert turn.answer_text == "Fallback answer."
    assert turn.used_fallback is True
    assert [command.target for command in pipeline.cancel_commands] == ["media_stream"]
