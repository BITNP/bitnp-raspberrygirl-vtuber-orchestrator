"""模块契约说明.

职责: 提供 orchestrator.semantic_barge_in
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol, final, override

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.llm import CancellationToken
from orchestrator.provider_streaming import ProviderResponseError
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    FlushAdmission,
    FlushClock,
    FlushRequestId,
    FlushSender,
    GeneratedSsrc,
    SegmentId,
    StreamFlush,
    StreamKey,
    TurnId,
)

if TYPE_CHECKING:
    from orchestrator.observability import OnsiteObservability, OnsiteStage


CLASSIFIER_TIMEOUT_MS: Final = 400

_MAX_ANSWER_EXCERPT_CHARS: Final = 512


@dataclass(frozen=True, slots=True)
class EndpointedTranscript:
    """类契约说明.

    职责: 保存 EndpointedTranscript
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    stream、text、turn_id、segment_id。
    """

    stream: StreamKey

    text: str

    turn_id: TurnId

    segment_id: SegmentId


@dataclass(frozen=True, slots=True)
class ActiveAnswer:
    """类契约说明.

    职责: 保存 ActiveAnswer
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stream、turn_id、segment_id、ca
    ncellation_epoch、answer_excerpt、targ
    et_generated_ssrc。
    """

    stream: StreamKey

    turn_id: TurnId

    segment_id: SegmentId

    cancellation_epoch: CancellationEpoch

    answer_excerpt: str

    target_generated_ssrc: GeneratedSsrc

    cancellation: CancellationToken = field(default_factory=CancellationToken)


@dataclass(frozen=True, slots=True)
class BargeInClassifierRequest:
    """类契约说明.

    职责: 保存 BargeInClassifierRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: transcript、active_turn_id、ac
    tive_segment_id、active_answer_excerp
    t、timeout_ms。
    """

    transcript: str

    active_turn_id: TurnId

    active_segment_id: SegmentId

    active_answer_excerpt: str

    timeout_ms: Literal[400] = CLASSIFIER_TIMEOUT_MS


class BargeInClassifier(Protocol):
    """类契约说明.

    职责: 声明 BargeInClassifier
    协议接口,约束实现方必须提供的行为。
    契约: 方法: classify。
    """

    def classify(self, request: BargeInClassifierRequest) -> str:
        """函数契约说明.

        功能: 执行 classify 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 request:
        BargeInClassifierRequest。 必填。
        契约: 同步调用。 返回 `str`。
        """
        ...


@dataclass(frozen=True, slots=True)
class BargeInClassifierFailure:
    """类契约说明.

    职责: 保存 BargeInClassifierFailure
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    stream、turn_id、segment_id、reason。
    """

    stream: StreamKey

    turn_id: TurnId

    segment_id: SegmentId

    reason: Literal["timeout", "malformed", "unavailable"]


@dataclass(frozen=True, slots=True)
class BargeInCancellation:
    """类契约说明.

    职责: 保存 BargeInCancellation
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、targets。
    """

    turn_id: TurnId

    segment_id: SegmentId

    targets: tuple[Literal["llm", "tts", "rtp"], ...] = ("llm", "tts", "rtp")


@dataclass(frozen=True, slots=True)
class _Decision:
    """类契约说明.

    职责: 保存 _Decision
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: utterance、active、value。
    """

    utterance: EndpointedTranscript

    active: ActiveAnswer | None

    value: Literal["interrupt", "continue"]


@final
class SemanticBargeInGate:
    """类契约说明.

    职责: 定义 SemanticBargeInGate
    的状态、行为和对外协作边界。
    契约: 方法: __init__、set_observability、c
    ancellation_epoch、failures、cancellat
    ions、activate。
    """

    def __init__(
        self, *, classifier: BargeInClassifier, clock: FlushClock, sender: FlushSender
    ) -> None:
        """函数契约说明.

        功能: 初始化 SemanticBargeInGate
        的字段并建立实例不变式。
        参数: self 表示当前实例。 classifier:
        BargeInClassifier。 必填。 clock:
        FlushClock。 必填。 sender:
        FlushSender。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._classifier = classifier

        self._flush_admission = FlushAdmission(clock=clock, sender=sender)

        self._active: ActiveAnswer | None = None

        self._queued: EndpointedTranscript | None = None

        self._replacement: EndpointedTranscript | None = None

        self._replacement_flush: StreamFlush | None = None

        self._epoch = CancellationEpoch(0)

        self._flush_sequence = 0

        self._failures: list[BargeInClassifierFailure] = []

        self._cancellations: list[BargeInCancellation] = []

        self._observability: OnsiteObservability | None = None

    def set_observability(self, observability: OnsiteObservability) -> None:
        """函数契约说明.

        功能: 执行 set_observability
        的同步逻辑,并产出 _observability。
        参数: self 表示当前实例。 observability:
        OnsiteObservability。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._observability = observability

    @property
    def cancellation_epoch(self) -> CancellationEpoch:
        """函数契约说明.

        功能: 执行 cancellation_epoch
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `CancellationEpoch`。
        """
        return self._epoch

    @property
    def failures(self) -> tuple[BargeInClassifierFailure, ...]:
        """函数契约说明.

        功能: 执行 failures 的同步逻辑,并协调 tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `tuple[BargeInClassifierFailure,
        ...]`。
        """
        return tuple(self._failures)

    @property
    def cancellations(self) -> tuple[BargeInCancellation, ...]:
        """函数契约说明.

        功能: 执行 cancellations 的同步逻辑,并协调
        tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `tuple[BargeInCancellation,
        ...]`。
        """
        return tuple(self._cancellations)

    def activate(self, active: ActiveAnswer) -> None:
        """函数契约说明.

        功能: 执行 activate 的同步逻辑,并产出
        _active, _epoch。
        参数: self 表示当前实例。 active:
        ActiveAnswer。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._active = active

        self._epoch = active.cancellation_epoch

    def handle(self, utterance: EndpointedTranscript) -> None:
        """函数契约说明.

        功能: 处理输入事件、请求或状态转换。
        参数: self 表示当前实例。 utterance:
        EndpointedTranscript。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self.apply(self.classify(utterance))

    def classify(self, utterance: EndpointedTranscript) -> _Decision:
        """函数契约说明.

        功能: 执行 classify 的同步逻辑,并协调
        BargeInClassifierRequest,
        _record, _Decision,
        _parse_decision。
        参数: self 表示当前实例。 utterance:
        EndpointedTranscript。 必填。
        契约: 同步调用。 返回 `_Decision`。
        """
        active = self._active

        if active is None or active.stream != utterance.stream:
            return _Decision(utterance=utterance, active=None, value="continue")

        request = BargeInClassifierRequest(
            transcript=utterance.text,
            active_turn_id=active.turn_id,
            active_segment_id=active.segment_id,
            active_answer_excerpt=active.answer_excerpt[-_MAX_ANSWER_EXCERPT_CHARS:],
        )

        try:
            value = _parse_decision(self._classifier.classify(request))

        except TimeoutError:
            self._record_failure(active, "timeout")

            value = "continue"

        except (JsonBoundaryError, BargeInResponseError):
            self._record_failure(active, "malformed")

            value = "continue"

        except (OSError, ProviderResponseError):
            self._record_failure(active, "unavailable")

            value = "continue"

        self._record("classifier_decision", active)

        return _Decision(utterance=utterance, active=active, value=value)

    def apply(self, decision: _Decision) -> None:
        """函数契约说明.

        功能: 执行 apply 的同步逻辑,并协调
        _interrupt。
        参数: self 表示当前实例。 decision:
        _Decision。 必填。
        契约: 同步调用。 返回 `None`。
        """
        if self._active != decision.active:
            return

        match decision.value:
            case "continue":
                self._queued = decision.utterance

            case "interrupt":
                if decision.active is not None:
                    self._interrupt(decision)

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> None:
        """函数契约说明.

        功能: 执行 acknowledge 的同步逻辑,并协调
        acknowledge。
        参数: self 表示当前实例。
        acknowledgement:
        FlushAcknowledgement。 必填。
        契约: 同步调用。 返回 `None`。
        """
        _ = self._flush_admission.acknowledge(acknowledgement)

    def pop_queued_utterance(self) -> EndpointedTranscript | None:
        """函数契约说明.

        功能: 执行 pop_queued_utterance
        的同步逻辑,并产出 queued, _queued。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `EndpointedTranscript | None`。
        """
        queued = self._queued

        self._queued = None

        return queued

    def pop_admitted_replacement(self) -> EndpointedTranscript | None:
        """函数契约说明.

        功能: 执行 pop_admitted_replacement
        的同步逻辑,并协调 admitted。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `EndpointedTranscript | None`。
        """
        replacement = self._replacement

        replacement_flush = self._replacement_flush

        if (
            replacement is None
            or replacement_flush is None
            or not self._flush_admission.admitted(replacement_flush)
        ):
            return None

        self._replacement = None

        self._replacement_flush = None

        return replacement

    def _interrupt(self, decision: _Decision) -> None:
        """函数契约说明.

        功能: 执行 _interrupt 的同步逻辑,并协调
        CancellationEpoch, cancel,
        append, _record。
        参数: self 表示当前实例。 decision:
        _Decision。 必填。
        契约: 同步调用。 返回 `None`。
        """
        active = decision.active

        if active is None:
            return

        self._epoch = CancellationEpoch(int(active.cancellation_epoch) + 1)

        _ = active.cancellation.cancel(reason="semantic_interrupt")

        self._cancellations.append(
            BargeInCancellation(turn_id=active.turn_id, segment_id=active.segment_id)
        )

        self._record("cancellation", active)

        self._replacement = decision.utterance

        self._flush_sequence += 1

        flush = StreamFlush(
            stream=active.stream,
            turn_id=active.turn_id,
            segment_id=active.segment_id,
            cancellation_epoch=self._epoch,
            request_id=FlushRequestId(
                f"{active.stream.session_id}:{active.stream.stream_id}:flush:{self._flush_sequence}"
            ),
            target_generated_ssrc=active.target_generated_ssrc,
        )

        self._replacement_flush = flush

        self._flush_admission.begin(flush)

        self._record("flush", active)

        self._active = None

    def _record_failure(
        self,
        active: ActiveAnswer,
        reason: Literal["timeout", "malformed", "unavailable"],
    ) -> None:
        """函数契约说明.

        功能: 执行 _record_failure 的同步逻辑,并协调
        append, _record,
        BargeInClassifierFailure。
        参数: self 表示当前实例。 active:
        ActiveAnswer。 必填。 reason:
        Literal['timeout', 'malformed',
        'unavailable']。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._failures.append(
            BargeInClassifierFailure(
                stream=active.stream,
                turn_id=active.turn_id,
                segment_id=active.segment_id,
                reason=reason,
            )
        )

        self._record("classifier_failure", active)

    def _record(self, stage: OnsiteStage, active: ActiveAnswer) -> None:
        """函数契约说明.

        功能: 执行 _record 的同步逻辑,并协调
        correlation, str, record。
        参数: self 表示当前实例。 stage:
        OnsiteStage。 必填。 active:
        ActiveAnswer。 必填。
        契约: 同步调用。 返回 `None`。
        """
        observability = self._observability

        if observability is not None:
            correlation = observability.correlation(
                active.stream,
                str(active.turn_id),
                str(active.segment_id),
                active.cancellation_epoch,
            )

            if correlation is not None:
                observability.record(stage, correlation)


@dataclass(frozen=True, slots=True)
class BargeInResponseError(ValueError):
    """类契约说明.

    职责: 保存 BargeInResponseError
    不可变数据结构,用类型标注表达字段契约。
    契约: 方法: __str__。
    """

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return "semantic barge-in classifier returned malformed output"


def _parse_decision(response: str) -> Literal["interrupt", "continue"]:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: response: str。 必填。
    契约: 同步调用。 返回 `Literal['interrupt',
    'continue']`。
    """
    payload = parse_json_value(response)

    if not isinstance(payload, dict) or set(payload) != {"decision"}:
        raise BargeInResponseError

    value = payload["decision"]

    match value:
        case "interrupt":
            return "interrupt"

        case "continue":
            return "continue"

        case _:
            raise BargeInResponseError
