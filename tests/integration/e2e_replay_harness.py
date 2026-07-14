from dataclasses import dataclass
from typing import Final, Literal, final, override

from orchestrator.pipeline import OrchestratorTurnPipeline
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    AudioMetadata,
    CommentAudienceEvent,
    SoundPlayCommand,
    TTSChunkEvent,
    TTSDoneEvent,
    TurnResult,
    VtuberSegmentCommands,
)

AUDIENCE_REJECTED: Final = "audience input was rejected"
EXPECTED_TURN: Final = "expected replay turn"
STALE_ACCEPTED: Final = "stale segment was accepted"
PEER_EDGE_RECORDED: Final = "peer communication edge recorded"
CHUNK_REJECTED: Final = "fresh TTS chunk was rejected"
CHUNK_TO_VTUBER: Final = "TTS chunk routed to vtuber"
DONE_REJECTED: Final = "fresh TTS completion was rejected"
DONE_TO_SOUND: Final = "TTS completion routed to sound"

type ServiceName = Literal["comments", "asr", "tts", "orchestrator", "sound", "vtuber"]
type TargetName = Literal["orchestrator", "tts", "sound", "vtuber"]
type ReplayEvent = CommentAudienceEvent | ASRAudienceEvent


@dataclass(frozen=True, slots=True)
class ReplayError(Exception):
    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ModuleEdge:
    source: ServiceName
    target: TargetName
    event_type: str


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    label: str
    event_type: str
    turn_id: str
    segment_id: str
    latency_ms: int


@dataclass(frozen=True, slots=True)
class ReplayTurnOutput:
    turn: TurnResult
    sound: SoundPlayCommand
    vtuber: VtuberSegmentCommands


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    name: str
    timeline: tuple[TimelineEvent, ...]
    edges: tuple[ModuleEdge, ...]
    turn_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]


@final
class ReplayHarness:
    """Mutable deterministic E2E driver for the Orchestrator pipeline."""

    def __init__(self, *, name: str, pipeline: OrchestratorTurnPipeline) -> None:
        self._name: str = name
        self._pipeline: OrchestratorTurnPipeline = pipeline
        self._timeline: list[TimelineEvent] = []
        self._edges: list[ModuleEdge] = []
        self._turn_ids: list[str] = []
        self._segment_ids: list[str] = []

    def submit(self, event: ReplayEvent) -> None:
        self._edges.append(
            ModuleEdge(_source_for(event), "orchestrator", "audience.input"),
        )
        self._timeline.append(TimelineEvent("audience", "audience.input", "-", "-", 0))
        if not self._pipeline.accept_audience_input(event):
            raise ReplayError(AUDIENCE_REJECTED)

    def start_next_turn(self) -> TurnResult:
        turn = self._pipeline.process_next_turn()
        if turn is None:
            raise ReplayError(EXPECTED_TURN)
        self._turn_ids.append(str(turn.turn_id))
        self._segment_ids.append(str(turn.segment_id))
        self._edges.append(
            ModuleEdge("orchestrator", "tts", turn.tts_command.event_type),
        )
        self._timeline.append(
            TimelineEvent(
                "tts_request",
                turn.tts_command.event_type,
                turn.turn_id,
                turn.segment_id,
                0,
            ),
        )
        return turn

    def finish_turn(self) -> ReplayTurnOutput:
        turn = self.start_next_turn()
        return ReplayTurnOutput(
            turn=turn,
            sound=self._sound_for(turn),
            vtuber=self._vtuber_for(turn),
        )

    def reject_stale_tts(self, turn: TurnResult) -> None:
        stale = self._pipeline.accept_tts_event(
            TTSDoneEvent(turn_id=turn.turn_id, segment_id=turn.segment_id),
        )
        if stale is not None:
            raise ReplayError(STALE_ACCEPTED)
        self._timeline.append(
            TimelineEvent(
                "stale_rejected",
                "tts.done",
                turn.turn_id,
                turn.segment_id,
                0,
            ),
        )

    def require_vtuber_commands(self, turn: TurnResult) -> VtuberSegmentCommands:
        return self._vtuber_for(turn)

    def assert_no_peer_edges(self) -> None:
        peer_edges = tuple(edge for edge in self._edges if is_peer_edge(edge))
        if len(peer_edges) > 0:
            raise ReplayError(PEER_EDGE_RECORDED)

    def summary(self) -> ScenarioSummary:
        return ScenarioSummary(
            self._name,
            tuple(self._timeline),
            tuple(self._edges),
            tuple(self._turn_ids),
            tuple(self._segment_ids),
        )

    def inject_edge(self, edge: ModuleEdge) -> None:
        self._edges.append(edge)

    def _sound_for(self, turn: TurnResult) -> SoundPlayCommand:
        chunk = TTSChunkEvent(
            turn.turn_id,
            turn.segment_id,
            "chunk-001",
            AudioMetadata(24_000, 1, "pcm_s16le", 120, 5_760),
            f"segment://{turn.segment_id}/chunk-001",
        )
        routed = self._pipeline.accept_tts_event(chunk)
        match routed:
            case SoundPlayCommand() as sound:
                self._edges.append(ModuleEdge("tts", "orchestrator", chunk.event_type))
                self._edges.append(
                    ModuleEdge("orchestrator", "sound", sound.event_type),
                )
                self._timeline.append(
                    TimelineEvent(
                        "sound_play",
                        sound.event_type,
                        sound.turn_id,
                        sound.segment_id,
                        0,
                    ),
                )
                return sound
            case None:
                raise ReplayError(CHUNK_REJECTED)
            case VtuberSegmentCommands():
                raise ReplayError(CHUNK_TO_VTUBER)

    def _vtuber_for(self, turn: TurnResult) -> VtuberSegmentCommands:
        done = TTSDoneEvent(turn_id=turn.turn_id, segment_id=turn.segment_id)
        routed = self._pipeline.accept_tts_event(done)
        match routed:
            case VtuberSegmentCommands() as vtuber:
                self._edges.append(ModuleEdge("tts", "orchestrator", done.event_type))
                self._edges.append(
                    ModuleEdge("orchestrator", "vtuber", vtuber.caption.event_type),
                )
                self._edges.append(
                    ModuleEdge("orchestrator", "vtuber", vtuber.action.event_type),
                )
                self._timeline.append(
                    TimelineEvent(
                        "vtuber_caption",
                        vtuber.caption.event_type,
                        vtuber.caption.turn_id,
                        vtuber.caption.segment_id,
                        0,
                    ),
                )
                self._timeline.append(
                    TimelineEvent(
                        "vtuber_action",
                        vtuber.action.event_type,
                        vtuber.action.turn_id,
                        vtuber.action.segment_id,
                        0,
                    ),
                )
                return vtuber
            case None:
                raise ReplayError(DONE_REJECTED)
            case SoundPlayCommand():
                raise ReplayError(DONE_TO_SOUND)


def event_types(summary: ScenarioSummary, event_type: str) -> tuple[str, ...]:
    return tuple(
        event.event_type for event in summary.timeline if event.event_type == event_type
    )


def is_peer_edge(edge: ModuleEdge) -> bool:
    return edge.source != "orchestrator" and edge.target != "orchestrator"


def _source_for(event: ReplayEvent) -> ServiceName:
    match event:
        case CommentAudienceEvent():
            return "comments"
        case ASRAudienceEvent():
            return "asr"
