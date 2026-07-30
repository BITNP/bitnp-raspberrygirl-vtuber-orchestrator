"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 ReplayError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。 方法: __str__。
    """

    reason: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """

        return self.reason


@dataclass(frozen=True, slots=True)
class ModuleEdge:
    """类契约说明.

    职责: 保存 ModuleEdge
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: source、target、event_type。
    """

    source: ServiceName

    target: ServiceName

    event_type: str


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    """类契约说明.

    职责: 保存 TimelineEvent
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: label、event_type、turn_id、seg
    ment_id、latency_ms。
    """

    label: str

    event_type: str

    turn_id: str

    segment_id: str

    latency_ms: int


@dataclass(frozen=True, slots=True)
class ReplayTurnOutput:
    """类契约说明.

    职责: 保存 ReplayTurnOutput
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn、cues、state。
    """

    turn: TurnResult

    cues: SynthesisCueResult

    state: MediaStreamState


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    """类契约说明.

    职责: 保存 ScenarioSummary
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: name、timeline、edges、turn_ids
    、segment_ids。
    """

    name: str

    timeline: tuple[TimelineEvent, ...]

    edges: tuple[ModuleEdge, ...]

    turn_ids: tuple[str, ...]

    segment_ids: tuple[str, ...]


@final
class ReplayHarness:
    """类契约说明.

    职责: 定义 ReplayHarness 的状态、行为和对外协作边界。
    契约: 方法: __init__、submit、start_next_t
    urn、finish_turn、reject_stale_synthes
    is、require_synthesis_cues。
    """

    def __init__(self, *, name: str, pipeline: OrchestratorTurnPipeline) -> None:
        """函数契约说明.

        功能: 初始化 ReplayHarness
        的字段并建立实例不变式。
        参数: self 表示当前实例。 name: str。 必填。
        pipeline:
        OrchestratorTurnPipeline。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self._name: str = name

        self._pipeline: OrchestratorTurnPipeline = pipeline

        self._timeline: list[TimelineEvent] = []

        self._edges: list[ModuleEdge] = []

        self._turn_ids: list[str] = []

        self._segment_ids: list[str] = []

    def submit(self, event: ReplayEvent) -> None:
        """函数契约说明.

        功能: 执行 submit 的同步逻辑,并协调 append,
        ModuleEdge, TimelineEvent,
        accept_audience_input。
        参数: self 表示当前实例。 event:
        ReplayEvent。 必填。
        契约: 同步调用。 返回 `None`。 可能抛出
        ReplayError。
        """

        self._edges.append(
            ModuleEdge(_source_for(event), "orchestrator", "audience.input"),
        )

        self._timeline.append(TimelineEvent("audience", "audience.input", "-", "-", 0))

        if not self._pipeline.accept_audience_input(event):
            raise ReplayError(AUDIENCE_REJECTED)

    def start_next_turn(self) -> TurnResult:
        """函数契约说明.

        功能: 执行 start_next_turn 的同步逻辑,并协调
        process_next_turn, append,
        ReplayError, str。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `TurnResult`。 可能抛出
        ReplayError。
        """

        turn = self._pipeline.process_next_turn()

        if turn is None:
            raise ReplayError(EXPECTED_TURN)

        self._turn_ids.append(str(turn.turn_id))

        self._segment_ids.append(str(turn.segment_id))

        return turn

    def finish_turn(self) -> ReplayTurnOutput:
        """函数契约说明.

        功能: 执行 finish_turn 的同步逻辑,并协调
        start_next_turn, _complete,
        _record_cues, MediaStreamState。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `ReplayTurnOutput`。
        可能抛出 ReplayError。
        """

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
        """函数契约说明.

        功能: 执行 reject_stale_synthesis
        的同步逻辑,并协调 complete_synthesis,
        append, _synthesis, ReplayError。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。
        契约: 同步调用。 返回 `None`。 可能抛出
        ReplayError。
        """

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
        """函数契约说明.

        功能: 执行 require_synthesis_cues
        的同步逻辑,并协调 _complete。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。
        契约: 同步调用。 返回
        `SynthesisCueResult`。
        """

        return self._complete(turn)

    def assert_no_peer_edges(self) -> None:
        """函数契约说明.

        功能: 执行 assert_no_peer_edges
        的同步逻辑,并协调 any, ReplayError,
        is_peer_edge。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。 可能抛出
        ReplayError。
        """

        if any(is_peer_edge(edge) for edge in self._edges):
            raise ReplayError(PEER_EDGE_RECORDED)

    def summary(self) -> ScenarioSummary:
        """函数契约说明.

        功能: 执行 summary 的同步逻辑,并协调
        ScenarioSummary, tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `ScenarioSummary`。
        """

        return ScenarioSummary(
            self._name,
            tuple(self._timeline),
            tuple(self._edges),
            tuple(self._turn_ids),
            tuple(self._segment_ids),
        )

    def inject_edge(self, edge: ModuleEdge) -> None:
        """函数契约说明.

        功能: 执行 inject_edge 的同步逻辑,并协调
        append。
        参数: self 表示当前实例。 edge:
        ModuleEdge。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self._edges.append(edge)

    def _complete(self, turn: TurnResult) -> SynthesisCueResult:
        """函数契约说明.

        功能: 执行 _complete 的同步逻辑,并协调
        complete_synthesis, _synthesis,
        ReplayError。
        参数: self 表示当前实例。 turn:
        TurnResult。 必填。
        契约: 同步调用。 返回
        `SynthesisCueResult`。 可能抛出
        ReplayError。
        """

        cues = self._pipeline.complete_synthesis(
            _synthesis(turn),
            rtp_stream_start_ms=0,
            stream_id=f"rtp-{turn.segment_id}",
        )

        if cues is None:
            raise ReplayError(SYNTHESIS_REJECTED)

        return cues

    def _record_cues(self, cues: SynthesisCueResult) -> None:
        """函数契约说明.

        功能: 执行 _record_cues 的同步逻辑,并协调
        append, ReplayError, ModuleEdge,
        TimelineEvent。
        参数: self 表示当前实例。 cues:
        SynthesisCueResult。 必填。
        契约: 同步调用。 返回 `None`。 可能抛出
        ReplayError。
        """

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
    """函数契约说明.

    功能: 执行 event_types 的同步逻辑,并协调 tuple。
    参数: summary: ScenarioSummary。 必填。
    event_type: str。 必填。
    契约: 同步调用。 返回 `tuple[str, ...]`。
    """

    return tuple(
        event.event_type for event in summary.timeline if event.event_type == event_type
    )


def is_peer_edge(edge: ModuleEdge) -> bool:
    """函数契约说明.

    功能: 执行 is_peer_edge 的同步逻辑,并维持签名契约。
    参数: edge: ModuleEdge。 必填。
    契约: 同步调用。 返回 `bool`。
    """

    return edge.source != "orchestrator" and edge.target != "orchestrator"


def _synthesis(turn: TurnResult) -> MockSynthesisResult:
    """函数契约说明.

    功能: 执行 _synthesis 的同步逻辑,并协调
    MockSynthesisResult, AudioMetadata。
    参数: turn: TurnResult。 必填。
    契约: 同步调用。 返回 `MockSynthesisResult`。
    """

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
    """函数契约说明.

    功能: 执行 _source_for 的同步逻辑,并维持签名契约。
    参数: event: ReplayEvent。 必填。
    契约: 同步调用。 返回 `ServiceName`。
    """

    match event:
        case CommentAudienceEvent():
            return "comments"

        case ASRAudienceEvent():
            return "asr"
