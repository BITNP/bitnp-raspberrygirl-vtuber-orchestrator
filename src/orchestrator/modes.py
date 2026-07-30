"""模块契约说明.

职责: 提供 orchestrator.modes
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType, override

ScriptStep = NewType("ScriptStep", int)

SlideStep = NewType("SlideStep", int)


@unique
class OrchestratorMode(StrEnum):
    """类契约说明.

    职责: 定义 OrchestratorMode
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    LECTURER = "lecturer"

    VIRTUAL_STREAMER = "virtual_streamer"

    ONSITE_EXPLAINER = "onsite_explainer"


@unique
class AudienceSource(StrEnum):
    """类契约说明.

    职责: 定义 AudienceSource 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    ASR = "asr"

    COMMENT = "comment"


@dataclass(frozen=True, slots=True)
class UnknownModeError(Exception):
    """类契约说明.

    职责: 保存 UnknownModeError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: raw_mode。 方法: __str__。
    """

    raw_mode: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"unknown orchestrator mode: {self.raw_mode}"


@dataclass(frozen=True, slots=True)
class AudienceInput:
    """类契约说明.

    职责: 保存 AudienceInput
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: source、text、received_at_ms。
    """

    source: AudienceSource

    text: str

    received_at_ms: int


@dataclass(frozen=True, slots=True)
class QaWindow:
    """类契约说明.

    职责: 保存 QaWindow 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: start_ms、end_ms。 方法:
    contains。
    """

    start_ms: int

    end_ms: int

    def contains(self, received_at_ms: int) -> bool:
        """函数契约说明.

        功能: 执行 contains 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 received_at_ms:
        int。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        return self.start_ms <= received_at_ms <= self.end_ms


@dataclass(frozen=True, slots=True)
class LecturerState:
    """类契约说明.

    职责: 保存 LecturerState
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: script_step、slide_step、immed
    iate_interruption_enabled、qa_window。
    """

    script_step: ScriptStep

    slide_step: SlideStep

    immediate_interruption_enabled: bool

    qa_window: QaWindow | None


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    """类契约说明.

    职责: 保存 AnswerCandidate
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: mode、input、reason、script_ste
    p、slide_step、topic。
    """

    mode: OrchestratorMode

    input: AudienceInput

    reason: str

    script_step: ScriptStep | None = None

    slide_step: SlideStep | None = None

    topic: str | None = None


@dataclass(frozen=True, slots=True)
class LecturerModePolicy:
    """类契约说明.

    职责: 保存 LecturerModePolicy
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: state。 方法: select_answer_can
    didate、_candidate、_qa_window_contain
    s。
    """

    state: LecturerState

    def select_answer_candidate(
        self,
        audience_inputs: Sequence[AudienceInput],
    ) -> AnswerCandidate | None:
        """函数契约说明.

        功能: 执行 select_answer_candidate
        的同步逻辑,并协调 _oldest_input,
        _qa_window_contains, _candidate。
        参数: self 表示当前实例。
        audience_inputs:
        Sequence[AudienceInput]。 必填。
        契约: 同步调用。 返回 `AnswerCandidate |
        None`。
        """
        audience_input = _oldest_input(audience_inputs)

        if audience_input is None:
            return None

        if self.state.immediate_interruption_enabled:
            return self._candidate(
                audience_input,
                reason="lecturer_immediate_interruption",
            )

        if self._qa_window_contains(audience_input.received_at_ms):
            return self._candidate(audience_input, reason="lecturer_scheduled_qa")

        return None

    def _candidate(
        self,
        audience_input: AudienceInput,
        *,
        reason: str,
    ) -> AnswerCandidate:
        """函数契约说明.

        功能: 执行 _candidate 的同步逻辑,并协调
        AnswerCandidate。
        参数: self 表示当前实例。 audience_input:
        AudienceInput。 必填。 reason: str。
        必填。
        契约: 同步调用。 返回 `AnswerCandidate`。
        """
        return AnswerCandidate(
            mode=OrchestratorMode.LECTURER,
            input=audience_input,
            reason=reason,
            script_step=self.state.script_step,
            slide_step=self.state.slide_step,
        )

    def _qa_window_contains(self, received_at_ms: int) -> bool:
        """函数契约说明.

        功能: 执行 _qa_window_contains
        的同步逻辑,并协调 contains。
        参数: self 表示当前实例。 received_at_ms:
        int。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        qa_window = self.state.qa_window

        if qa_window is None:
            return False

        return qa_window.contains(received_at_ms)


@dataclass(frozen=True, slots=True)
class VirtualStreamerModePolicy:
    """类契约说明.

    职责: 保存 VirtualStreamerModePolicy
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: topic。 方法:
    select_answer_candidate。
    """

    topic: str

    def select_answer_candidate(
        self,
        audience_inputs: Sequence[AudienceInput],
    ) -> AnswerCandidate | None:
        """函数契约说明.

        功能: 执行 select_answer_candidate
        的同步逻辑,并协调 _oldest_source,
        AnswerCandidate, _oldest_input。
        参数: self 表示当前实例。
        audience_inputs:
        Sequence[AudienceInput]。 必填。
        契约: 同步调用。 返回 `AnswerCandidate |
        None`。
        """
        audience_input = _oldest_source(audience_inputs, AudienceSource.COMMENT)

        if audience_input is None:
            audience_input = _oldest_input(audience_inputs)

        if audience_input is None:
            return None

        return AnswerCandidate(
            mode=OrchestratorMode.VIRTUAL_STREAMER,
            input=audience_input,
            reason="virtual_streamer_comment_priority",
            topic=self.topic,
        )


@dataclass(frozen=True, slots=True)
class OnsiteExplainerModePolicy:
    """类契约说明.

    职责: 保存 OnsiteExplainerModePolicy
    不可变数据结构,用类型标注表达字段契约。
    契约: 方法: select_answer_candidate。
    """

    def select_answer_candidate(
        self,
        audience_inputs: Sequence[AudienceInput],
    ) -> AnswerCandidate | None:
        """函数契约说明.

        功能: 执行 select_answer_candidate
        的同步逻辑,并协调 _oldest_source,
        AnswerCandidate, _oldest_input。
        参数: self 表示当前实例。
        audience_inputs:
        Sequence[AudienceInput]。 必填。
        契约: 同步调用。 返回 `AnswerCandidate |
        None`。
        """
        audience_input = _oldest_source(audience_inputs, AudienceSource.ASR)

        if audience_input is None:
            audience_input = _oldest_input(audience_inputs)

        if audience_input is None:
            return None

        return AnswerCandidate(
            mode=OrchestratorMode.ONSITE_EXPLAINER,
            input=audience_input,
            reason="onsite_explainer_asr_priority",
        )


class ModePolicy:
    """类契约说明.

    职责: 定义 ModePolicy 的状态、行为和对外协作边界。
    契约: 方法: lecturer、virtual_streamer、on
    site_explainer。
    """

    @staticmethod
    def lecturer(state: LecturerState) -> LecturerModePolicy:
        """函数契约说明.

        功能: 执行 lecturer 的同步逻辑,并协调
        LecturerModePolicy。
        参数: state: LecturerState。 必填。
        契约: 同步调用。 返回
        `LecturerModePolicy`。
        """
        return LecturerModePolicy(state=state)

    @staticmethod
    def virtual_streamer(*, topic: str) -> VirtualStreamerModePolicy:
        """函数契约说明.

        功能: 执行 virtual_streamer
        的同步逻辑,并协调
        VirtualStreamerModePolicy。
        参数: topic: str。 必填。
        契约: 同步调用。 返回
        `VirtualStreamerModePolicy`。
        """
        return VirtualStreamerModePolicy(topic=topic)

    @staticmethod
    def onsite_explainer() -> OnsiteExplainerModePolicy:
        """函数契约说明.

        功能: 执行 onsite_explainer
        的同步逻辑,并协调
        OnsiteExplainerModePolicy。
        参数: 无显式业务参数。
        契约: 同步调用。 返回
        `OnsiteExplainerModePolicy`。
        """
        return OnsiteExplainerModePolicy()


def parse_orchestrator_mode(raw_mode: str) -> OrchestratorMode:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: raw_mode: str。 必填。
    契约: 同步调用。 返回 `OrchestratorMode`。
    可能抛出 UnknownModeError。
    """
    try:
        mode = OrchestratorMode(raw_mode)

    except ValueError as error:
        raise UnknownModeError(raw_mode=raw_mode) from error

    return mode


def _oldest_input(audience_inputs: Sequence[AudienceInput]) -> AudienceInput | None:
    """函数契约说明.

    功能: 执行 _oldest_input 的同步逻辑,并协调 min,
    len。
    参数: audience_inputs:
    Sequence[AudienceInput]。 必填。
    契约: 同步调用。 返回 `AudienceInput | None`。
    """
    if len(audience_inputs) == 0:
        return None

    return min(
        audience_inputs,
        key=lambda audience_input: audience_input.received_at_ms,
    )


def _oldest_source(
    audience_inputs: Sequence[AudienceInput],
    source: AudienceSource,
) -> AudienceInput | None:
    """函数契约说明.

    功能: 执行 _oldest_source 的同步逻辑,并协调
    tuple, _oldest_input。
    参数: audience_inputs:
    Sequence[AudienceInput]。 必填。 source:
    AudienceSource。 必填。
    契约: 同步调用。 返回 `AudienceInput | None`。
    """
    matching_inputs = tuple(
        audience_input
        for audience_input in audience_inputs
        if audience_input.source is source
    )

    return _oldest_input(matching_inputs)
