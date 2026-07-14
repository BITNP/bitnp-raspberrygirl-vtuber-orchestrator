from orchestrator.llm import MockLLMAdapter
from orchestrator.modes import ModePolicy
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import (
    AudioMetadata,
    CommentAudienceEvent,
    PipelineConfig,
    SoundPlayCommand,
    TTSChunkEvent,
    TTSDoneEvent,
    VtuberSegmentCommands,
)
from orchestrator.retrieval import RetrievalFixtureProvider


def test_comment_turn_routes_llm_to_tts_sound_and_vtuber_commands() -> None:
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

    # When: Orchestrator accepts the comment and routes downstream observations.
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
    chunk = TTSChunkEvent(
        turn_id=turn.turn_id,
        segment_id=turn.segment_id,
        chunk_id="chunk-001",
        audio=AudioMetadata(
            sample_rate=24_000,
            channels=1,
            codec="pcm_s16le",
            duration_ms=120,
            byte_length=5_760,
        ),
        uri="segment://seg-0001/chunk-001",
    )
    sound = pipeline.accept_tts_event(chunk)
    vtuber = pipeline.accept_tts_event(
        TTSDoneEvent(turn_id=turn.turn_id, segment_id=turn.segment_id),
    )

    # Then: the turn emits typed commands with stable turn and segment IDs.
    assert accepted is True
    assert turn.tts_command.text == "Hello Alice"
    assert turn.tts_command.turn_id == "turn-0001"
    assert turn.tts_command.segment_id == "seg-0001"
    assert isinstance(sound, SoundPlayCommand)
    assert sound.command_id == "sound-seg-0001-chunk-001"
    assert sound.segment_id == "seg-0001"
    assert isinstance(vtuber, VtuberSegmentCommands)
    assert vtuber.caption.text == "Hello Alice"
    assert vtuber.action.action == "speak"
