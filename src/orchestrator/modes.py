"""模块契约说明.

职责: 提供模式无关的受众输入选择策略。
契约: 模块只暴露受众来源、输入、候选和自适应策略。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique


@unique
class AudienceSource(StrEnum):
    """类契约说明.

    职责: 定义受众输入来源。
    契约: 枚举值表示边界输入来源,不表示产品模式。
    """

    ASR = "asr"
    COMMENT = "comment"


@dataclass(frozen=True, slots=True)
class AudienceInput:
    """类契约说明.

    职责: 保存规范化受众输入。
    契约: 字段: source、text、received_at_ms。
    """

    source: AudienceSource
    text: str
    received_at_ms: int


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    """类契约说明.

    职责: 保存可提交给回答流程的受众输入候选。
    契约: 字段: input。
    """

    input: AudienceInput


class AdaptiveAgentPolicy:
    """类契约说明.

    职责: 定义单一自适应智能体的输入选择策略。
    契约: 方法: select_answer_candidate。
    """

    def select_answer_candidate(
        self,
        audience_inputs: Sequence[AudienceInput],
    ) -> AnswerCandidate | None:
        """函数契约说明.

        功能: 选择最早到达的受众输入作为回答候选。
        参数: self 表示当前实例。 audience_inputs:
        Sequence[AudienceInput]。 必填。
        契约: 同步调用。 返回 `AnswerCandidate | None`。
        """
        audience_input = _oldest_input(audience_inputs)

        if audience_input is None:
            return None

        return AnswerCandidate(input=audience_input)


def _oldest_input(audience_inputs: Sequence[AudienceInput]) -> AudienceInput | None:
    if len(audience_inputs) == 0:
        return None

    return min(
        audience_inputs,
        key=lambda audience_input: audience_input.received_at_ms,
    )
