from orchestrator.llm import MockLLMAdapter
from orchestrator.modes import ModePolicy
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    AudioMetadata,
    CommentAudienceEvent,
    MockSynthesisResult,
    PipelineConfig,
)
from orchestrator.retrieval import RetrievalFixtureProvider


def test_comment_turn_emits_media_stream_and_frontend_cues_after_synthesis() -> None:
    # Given: a virtual-streamer policy and normalized comment input.
    pipeline = OrchestratorTurnPipeline(
        adapters=PipelineAdapters(
            mode_policy=ModePolicy.virtual_streamer(topic="bitnet"),
            llm=MockLLMAdapter(answer_chunks=("Hello ", "Alice")),
            retrieval=RetrievalFixtureProvider(refs=()),
        ),
        config=PipelineConfig(
            queue_capacity=2,
            turn_id_prefix="turn",
            segment_id_prefix="seg",
        ),
    )

    # When: Orchestrator completes its local synthesis for the answered turn.
    accepted = pipeline.accept_audience_input(
        CommentAudienceEvent(
            platform="bilibili",
            source="danmaku",
            user="alice",
            text="Say hi",
            timestamp="2026-07-08T00:00:01Z",
        ),
    )
    turn = pipeline.process_next_turn()
    assert turn is not None
    cues = pipeline.complete_synthesis(
        MockSynthesisResult(
            turn_id=turn.turn_id,
            segment_id=turn.segment_id,
            audio=AudioMetadata(
                sample_rate=24_000,
                channels=1,
                codec="pcm_s16le",
                duration_ms=120,
                byte_length=5_760,
            ),
            expression="smile",
            action="speak",
            scene="streamer_main",
            slide_page=1,
        ),
        rtp_stream_start_ms=2_000,
        stream_id="rtp-comment-0001",
    )

    # Then: a single Orchestrator-owned synthesis emits canonical controls.
    assert accepted is True
    assert turn.answer_text == "Hello Alice"
    assert cues is not None
    assert cues.media is not None
    assert cues.media.event_type == "media.stream.command"
    assert cues.media.stream_id == "rtp-comment-0001"
    assert cues.media.start_at_ms == 2_000
    assert cues.caption.text == "Hello Alice"
    assert cues.caption.start_at_ms == 2_000
    assert cues.expression.event_type == "vtuber.expression.command"
    assert cues.action.event_type == "vtuber.action.command"
    assert cues.scene.event_type == "vtuber.scene.command"


def test_mock_synthesis_emits_rtp_relative_frontend_cues() -> None:
    # Given: one mock synthesis result and a known RTP stream start.
    pipeline = OrchestratorTurnPipeline(
        adapters=PipelineAdapters(
            mode_policy=ModePolicy.onsite_explainer(),
            llm=MockLLMAdapter(answer_chunks=("Hello Alice",)),
            retrieval=RetrievalFixtureProvider(refs=()),
        ),
        config=PipelineConfig(
            queue_capacity=2,
            turn_id_prefix="turn",
            segment_id_prefix="seg",
        ),
    )
    assert pipeline.accept_audience_input(
        ASRAudienceEvent(
            text="Say hi",
            received_at_ms=1_000,
            segment_id="asr-local-0001",
            seq=1,
        ),
    )
    turn = pipeline.process_next_turn()
    assert turn is not None
    # When: the pipeline maps the completed mock audio to frontend-control cues.
    cues = pipeline.complete_synthesis(
        MockSynthesisResult(
            turn_id=turn.turn_id,
            segment_id=turn.segment_id,
            audio=AudioMetadata(24_000, 1, "pcm_s16le", 120, 5_760),
            expression="smile",
            action="wave",
            scene="lecture_slide_focus",
            slide_page=3,
        ),
        rtp_stream_start_ms=10_000,
    )

    # Then: all frontend cues share deterministic stream-relative timing.
    assert cues is not None
    assert cues.caption.start_at_ms == 10_000
    assert cues.expression.expression == "smile"
    assert cues.expression.start_at_ms == 10_000
    assert cues.action.action == "wave"
    assert cues.action.start_at_ms == 10_000
    assert cues.scene.scene == "lecture_slide_focus"
    assert cues.scene.slide_page == 3
    assert cues.scene.start_at_ms == 10_000
