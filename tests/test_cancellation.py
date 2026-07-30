"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from orchestrator.llm import MockLLMAdapter
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    AudioMetadata,
    MockSynthesisResult,
    PipelineConfig,
)
from orchestrator.retrieval import RetrievalFixtureProvider


def test_interruption_cancels_previous_segment_and_rejects_stale_synthesis() -> None:
    # Given: an onsite explainer pipeline with one active turn.

    """函数契约说明.

    功能: 验证 interruption cancels previous
    segment and rejects stale synthesis
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    pipeline = OrchestratorTurnPipeline(
        adapters=PipelineAdapters(
            mode_policy=AdaptiveAgentPolicy(),
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

    stale = pipeline.complete_synthesis(
        MockSynthesisResult(
            turn_id=first.turn_id,
            segment_id=first.segment_id,
            audio=AudioMetadata(24_000, 1, "pcm_s16le", 120, 5_760),
            expression="smile",
            action="speak",
            scene="onsite",
            slide_page=1,
        ),
        rtp_stream_start_ms=10,
    )

    second = pipeline.process_next_turn()

    # Then: previous downstream work is cancelled and stale events are ignored.

    assert stale is None

    assert [command.segment_id for command in pipeline.cancel_commands] == [
        "seg-0001",
    ] * 2

    assert [command.target for command in pipeline.cancel_commands] == [
        "media_stream",
        "frontend",
    ]

    assert second is not None

    assert second.turn_id == "turn-0002"

    assert second.segment_id == "seg-0002"


def test_cancelled_segment_emits_no_stale_media_or_frontend_cues() -> None:
    # Given: a segment cancelled by a newer ASR request before synthesis completes.

    """函数契约说明.

    功能: 验证 cancelled segment emits no
    stale media or frontend cues
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    pipeline = OrchestratorTurnPipeline(
        adapters=PipelineAdapters(
            mode_policy=AdaptiveAgentPolicy(),
            llm=MockLLMAdapter(answer_chunks=("First answer",)),
            retrieval=RetrievalFixtureProvider(refs=()),
        ),
        config=PipelineConfig(2, "turn", "seg"),
    )

    assert pipeline.accept_audience_input(
        ASRAudienceEvent(text="first", received_at_ms=10, segment_id="asr-1", seq=1),
    )

    first = pipeline.process_next_turn()

    assert first is not None

    assert pipeline.accept_audience_input(
        ASRAudienceEvent(text="second", received_at_ms=20, segment_id="asr-2", seq=2),
    )

    # When: the cancelled segment's completed audio arrives late.

    stale = pipeline.complete_synthesis(
        MockSynthesisResult(
            turn_id=first.turn_id,
            segment_id=first.segment_id,
            audio=None,
            expression="smile",
            action="wave",
            scene="lecture_slide_focus",
            slide_page=1,
        ),
        rtp_stream_start_ms=10_000,
    )

    # Then: no stale RTP media, caption, expression, action, or scene event escapes.

    assert stale is None


def test_interruption_preserves_monotonic_turn_identifiers() -> None:
    # Given: an onsite pipeline with a completed first turn.

    """函数契约说明.

    功能: 验证 interruption preserves
    monotonic turn identifiers
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    pipeline = OrchestratorTurnPipeline(
        adapters=PipelineAdapters(
            mode_policy=AdaptiveAgentPolicy(),
            llm=MockLLMAdapter(answer_chunks=("First answer",)),
            retrieval=RetrievalFixtureProvider(refs=()),
        ),
        config=PipelineConfig(2, "turn", "seg"),
    )

    assert pipeline.accept_audience_input(
        ASRAudienceEvent(text="first", received_at_ms=10, segment_id="asr-1", seq=1),
    )

    first = pipeline.process_next_turn()

    # When: a newer input interrupts the first turn and opens its replacement.

    assert first is not None

    assert pipeline.accept_audience_input(
        ASRAudienceEvent(text="second", received_at_ms=20, segment_id="asr-2", seq=2),
    )

    second = pipeline.process_next_turn()

    # Then: cancellation does not reuse the prior turn or segment identity.

    assert second is not None

    assert (first.turn_id, first.segment_id) == ("turn-0001", "seg-0001")

    assert (second.turn_id, second.segment_id) == ("turn-0002", "seg-0002")
