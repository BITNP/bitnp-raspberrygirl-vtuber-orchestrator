
from dataclasses import dataclass
from typing import Final, Literal, final, override

from orchestrator.pipeline import OrchestratorTurnPipeline
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    AudioMetadata,
    CommentAudienceEvent,
    MediaStreamState,
    MockSynthesisResult,
    SynthesisCueResult,
    TurnResult,
)

AUDIENCE_REJECTED: Final = "audience input was rejected"

EXPECTED_TURN: Final = "expected replay turn"

STALE_ACCEPTED: Final = "stale segment was accepted"

PEER_EDGE_RECORDED: Final = "peer communication edge recorded"

SYNTHESIS_REJECTED: Final = "fresh synthesis result was rejected"


type ServiceName = Literal["comments", "asr", "orchestrator", "sound", "frontend"]

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

    target: ServiceName

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

    cues: SynthesisCueResult

    state: MediaStreamState


@dataclass(frozen=True, slots=True)
class ScenarioSummary:

    name: str

    timeline: tuple[TimelineEvent, ...]

    edges: tuple[ModuleEdge, ...]

    turn_ids: tuple[str, ...]

    segment_ids: tuple[str, ...]


@final
class ReplayHarness:

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

        return turn

    def finish_turn(self) -> ReplayTurnOutput:

        turn = self.start_next_turn()

        cues = self._complete(turn)

        self._record_cues(cues)

        media = cues.media

        if media is None:
            raise ReplayError(SYNTHESIS_REJECTED)

        state = MediaStreamState(
            turn.turn_id,
            turn.segment_id,
            media.stream_id,
            "finished",
            media.audio.duration_ms,
        )

        self._edges.append(ModuleEdge("sound", "orchestrator", state.event_type))

        self._timeline.append(
            TimelineEvent(
                "media_state",
                state.event_type,
                turn.turn_id,
                turn.segment_id,
                0,
            ),
        )

        return ReplayTurnOutput(turn, cues, state)

    def reject_stale_synthesis(self, turn: TurnResult) -> None:

        stale = self._pipeline.complete_synthesis(
            _synthesis(turn),
            rtp_stream_start_ms=0,
            stream_id=f"rtp-{turn.segment_id}",
        )

        if stale is not None:
            raise ReplayError(STALE_ACCEPTED)

        self._timeline.append(
            TimelineEvent(
                "stale_rejected",
                "media.stream.command",
                turn.turn_id,
                turn.segment_id,
                0,
            ),
        )

    def require_synthesis_cues(self, turn: TurnResult) -> SynthesisCueResult:

        return self._complete(turn)

    def assert_no_peer_edges(self) -> None:

        if any(is_peer_edge(edge) for edge in self._edges):
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

    def _complete(self, turn: TurnResult) -> SynthesisCueResult:

        cues = self._pipeline.complete_synthesis(
            _synthesis(turn),
            rtp_stream_start_ms=0,
            stream_id=f"rtp-{turn.segment_id}",
        )

        if cues is None:
            raise ReplayError(SYNTHESIS_REJECTED)

        return cues

    def _record_cues(self, cues: SynthesisCueResult) -> None:

        media = cues.media

        if media is None:
            raise ReplayError(SYNTHESIS_REJECTED)

        self._edges.append(ModuleEdge("orchestrator", "sound", media.event_type))

        self._timeline.append(
            TimelineEvent(
                "media_command",
                media.event_type,
                media.turn_id,
                media.segment_id,
                0,
            ),
        )

        for cue in (cues.caption, cues.expression, cues.action, cues.scene):
            self._edges.append(ModuleEdge("orchestrator", "frontend", cue.event_type))

            self._timeline.append(
                TimelineEvent(
                    "frontend_cue",
                    cue.event_type,
                    cue.turn_id,
                    cue.segment_id,
                    0,
                ),
            )


def event_types(summary: ScenarioSummary, event_type: str) -> tuple[str, ...]:

    return tuple(
        event.event_type for event in summary.timeline if event.event_type == event_type
    )


def is_peer_edge(edge: ModuleEdge) -> bool:

    return edge.source != "orchestrator" and edge.target != "orchestrator"


def _synthesis(turn: TurnResult) -> MockSynthesisResult:

    return MockSynthesisResult(
        turn.turn_id,
        turn.segment_id,
        AudioMetadata(24_000, 1, "pcm_s16le", 120, 5_760),
        "smile",
        "speak",
        "presentation",
        1,
    )


def _source_for(event: ReplayEvent) -> ServiceName:

    match event:
        case CommentAudienceEvent():
            return "comments"

        case ASRAudienceEvent():
            return "asr"
