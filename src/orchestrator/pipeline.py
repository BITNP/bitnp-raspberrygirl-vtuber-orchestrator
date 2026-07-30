"""模块契约说明.

职责: 提供 orchestrator.pipeline
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from orchestrator.ids import SegmentId, TurnId
from orchestrator.llm import (
    CancellationToken,
    LLMAdapter,
    LLMChunk,
    LLMError,
    LLMFinal,
    build_llm_request,
)
from orchestrator.modes import AnswerCandidate, AudienceInput, AudienceSource
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    AudienceEvent,
    CancelCommand,
    CommentAudienceEvent,
    MediaStreamCommand,
    MockSynthesisResult,
    PipelineConfig,
    SynthesisCueResult,
    TurnResult,
    VtuberActionCommand,
    VtuberCaptionCommand,
    VtuberExpressionCommand,
    VtuberSceneCommand,
)
from orchestrator.retrieval import RetrievalProvider


class AnswerPolicy(Protocol):
    """类契约说明.

    职责: 声明 AnswerPolicy
    协议接口,约束实现方必须提供的行为。
    契约: 方法: select_answer_candidate。
    """

    def select_answer_candidate(
        self,
        audience_inputs: tuple[AudienceInput, ...],
    ) -> AnswerCandidate | None:
        """函数契约说明.

        功能: 执行 select_answer_candidate
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        audience_inputs:
        tuple[AudienceInput, ...]。 必填。
        契约: 同步调用。 返回 `AnswerCandidate |
        None`。
        """
        ...


@dataclass(frozen=True, slots=True)
class PipelineAdapters:
    """类契约说明.

    职责: 保存 PipelineAdapters
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: mode_policy、llm、retrieval。
    """

    mode_policy: AnswerPolicy

    llm: LLMAdapter

    retrieval: RetrievalProvider


class OrchestratorTurnPipeline:
    """类契约说明.

    职责: 定义 OrchestratorTurnPipeline
    的状态、行为和对外协作边界。
    契约: 方法: __init__、rejections、cancel_c
    ommands、accept_audience_input、proces
    s_next_turn、complete_synthesis。
    """

    def __init__(
        self,
        *,
        adapters: PipelineAdapters,
        config: PipelineConfig,
    ) -> None:
        """函数契约说明.

        功能: 初始化 OrchestratorTurnPipeline
        的字段并建立实例不变式。
        参数: self 表示当前实例。 adapters:
        PipelineAdapters。 必填。 config:
        PipelineConfig。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._mode_policy: AnswerPolicy = adapters.mode_policy

        self._llm: LLMAdapter = adapters.llm

        self._retrieval: RetrievalProvider = adapters.retrieval

        self._queue_capacity: int = config.queue_capacity

        self._turn_id_prefix: str = config.turn_id_prefix

        self._segment_id_prefix: str = config.segment_id_prefix

        self._queue: deque[AudienceEvent] = deque()

        self._turn_seq: int = 0

        self._active: _ActiveTurn | None = None

        self._stale_segments: set[SegmentId] = set()

        self._rejections: list[str] = []

        self._cancel_commands: list[CancelCommand] = []

    @property
    def rejections(self) -> tuple[str, ...]:
        """函数契约说明.

        功能: 执行 rejections 的同步逻辑,并协调
        tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[str, ...]`。
        """
        return tuple(self._rejections)

    @property
    def cancel_commands(self) -> tuple[CancelCommand, ...]:
        """函数契约说明.

        功能: 执行 cancel_commands 的同步逻辑,并协调
        tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `tuple[CancelCommand, ...]`。
        """
        return tuple(self._cancel_commands)

    def accept_audience_input(self, event: AudienceEvent) -> bool:
        """函数契约说明.

        功能: 执行 accept_audience_input
        的同步逻辑,并协调 append,
        _cancel_active, len。
        参数: self 表示当前实例。 event:
        AudienceEvent。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        if self._active is not None:
            self._cancel_active(reason="user_interrupt")

        if len(self._queue) >= self._queue_capacity:
            self._rejections.append("queue_full")

            return False

        self._queue.append(event)

        return True

    def process_next_turn(
        self, cancellation: CancellationToken | None = None
    ) -> TurnResult | None:
        """函数契约说明.

        功能: 执行 process_next_turn
        的同步逻辑,并协调 _to_audience_input,
        select_answer_candidate, TurnId,
        SegmentId。
        参数: self 表示当前实例。 cancellation:
        CancellationToken | None。 可省略。
        契约: 同步调用。 返回 `TurnResult |
        None`。
        """
        if len(self._queue) == 0:
            return None

        audience_input = _to_audience_input(self._queue.popleft())

        candidate = self._mode_policy.select_answer_candidate((audience_input,))

        if candidate is None:
            return None

        self._turn_seq += 1

        turn_id = TurnId(f"{self._turn_id_prefix}-{self._turn_seq:04d}")

        segment_id = SegmentId(f"{self._segment_id_prefix}-{self._turn_seq:04d}")

        token = CancellationToken() if cancellation is None else cancellation

        text_parts: list[str] = []

        final: LLMFinal | None = None

        request = build_llm_request(
            candidate,
            retrieval=self._retrieval.retrieve(candidate),
        )

        for llm_event in self._llm.stream(
            request,
            cancellation=token,
        ):
            match llm_event:
                case LLMChunk(text=text):
                    text_parts.append(text)

                case LLMError() as error:
                    if error.cancel_pending_media:
                        self._cancel_commands.append(
                            _cancel(
                                _CancelIntent(
                                    turn_id,
                                    segment_id,
                                    "media_stream",
                                    "llm_timeout",
                                ),
                            ),
                        )

                case LLMFinal() as llm_final:
                    final = llm_final

        answer_text = final.text if final is not None else "".join(text_parts)

        self._active = _ActiveTurn(
            turn_id=turn_id,
            segment_id=segment_id,
            text=answer_text,
            cancellation=token,
        )

        return TurnResult(
            turn_id=turn_id,
            segment_id=segment_id,
            answer_text=answer_text,
            used_fallback=final.used_fallback if final is not None else False,
        )

    def complete_synthesis(
        self,
        synthesis: MockSynthesisResult,
        *,
        rtp_stream_start_ms: int,
        stream_id: str = "rtp-local",
    ) -> SynthesisCueResult | None:
        """函数契约说明.

        功能: 执行 complete_synthesis
        的同步逻辑,并协调 SynthesisCueResult,
        MediaStreamCommand,
        VtuberCaptionCommand,
        VtuberExpressionCommand。
        参数: self 表示当前实例。 synthesis:
        MockSynthesisResult。 必填。
        rtp_stream_start_ms: int。 必填。
        stream_id: str。 可省略。
        契约: 同步调用。 返回 `SynthesisCueResult
        | None`。
        """
        active = self._active

        if active is None or synthesis.segment_id in self._stale_segments:
            return None

        if (
            synthesis.turn_id != active.turn_id
            or synthesis.segment_id != active.segment_id
        ):
            return None

        if synthesis.audio is None:
            return None

        offset_ms = synthesis.offset_samples * 1_000 // synthesis.audio.sample_rate

        start_at_ms = rtp_stream_start_ms + offset_ms

        return SynthesisCueResult(
            media=MediaStreamCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                stream_id=stream_id,
                audio=synthesis.audio,
                start_at_ms=start_at_ms,
            ),
            caption=VtuberCaptionCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                text=active.text,
                start_at_ms=start_at_ms,
            ),
            expression=VtuberExpressionCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                expression=synthesis.expression,
                start_at_ms=start_at_ms,
            ),
            action=VtuberActionCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                action=synthesis.action,
                start_at_ms=start_at_ms,
            ),
            scene=VtuberSceneCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                scene=synthesis.scene,
                slide_id="",
                slide_title="",
                slide_page=synthesis.slide_page,
                start_at_ms=start_at_ms,
            ),
        )

    def _cancel_active(self, *, reason: str) -> None:
        """函数契约说明.

        功能: 执行 _cancel_active 的同步逻辑,并协调
        cancel, add, extend, _cancel。
        参数: self 表示当前实例。 reason: str。
        必填。
        契约: 同步调用。 返回 `None`。
        """
        active = self._active

        if active is None:
            return

        _ = active.cancellation.cancel(reason=reason)

        self._stale_segments.add(active.segment_id)

        self._cancel_commands.extend(
            (
                _cancel(
                    _CancelIntent(active.turn_id, active.segment_id, target, reason),
                )
                for target in ("media_stream", "frontend")
            ),
        )

        self._active = None


@dataclass(frozen=True, slots=True)
class _ActiveTurn:
    """类契约说明.

    职责: 保存 _ActiveTurn
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、text、canc
    ellation。
    """

    turn_id: TurnId

    segment_id: SegmentId

    text: str

    cancellation: CancellationToken


@dataclass(frozen=True, slots=True)
class _CancelIntent:
    """类契约说明.

    职责: 保存 _CancelIntent
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    turn_id、segment_id、target、reason。
    """

    turn_id: TurnId

    segment_id: SegmentId

    target: Literal["media_stream", "frontend"]

    reason: str


def _to_audience_input(event: AudienceEvent) -> AudienceInput:
    """函数契约说明.

    功能: 将输入转换为目标表示。
    参数: event: AudienceEvent。 必填。
    契约: 同步调用。 返回 `AudienceInput`。
    """
    match event:
        case CommentAudienceEvent(text=text, timestamp=timestamp):
            return AudienceInput(
                source=AudienceSource.COMMENT,
                text=text,
                received_at_ms=_timestamp_ms(timestamp),
            )

        case ASRAudienceEvent(text=text, received_at_ms=received_at_ms):
            return AudienceInput(
                source=AudienceSource.ASR,
                text=text,
                received_at_ms=received_at_ms,
            )


def _timestamp_ms(raw_timestamp: str) -> int:
    """函数契约说明.

    功能: 执行 _timestamp_ms 的同步逻辑,并协调
    fromisoformat, int, timestamp。
    参数: raw_timestamp: str。 必填。
    契约: 同步调用。 返回 `int`。
    """
    parsed = datetime.fromisoformat(raw_timestamp)

    return int(parsed.timestamp() * 1000)


def _cancel(intent: _CancelIntent) -> CancelCommand:
    """函数契约说明.

    功能: 执行 _cancel 的同步逻辑,并协调
    CancelCommand。
    参数: intent: _CancelIntent。 必填。
    契约: 同步调用。 返回 `CancelCommand`。
    """
    match intent.target:
        case "media_stream" | "frontend":
            return CancelCommand(
                turn_id=intent.turn_id,
                segment_id=intent.segment_id,
                target=intent.target,
                reason=intent.reason,
            )
