"""模块契约说明.

职责: 提供 orchestrator.asr_semantic_gate
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Protocol

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value

_INSTRUCTION: Final = (
    "判断这条已完成的语音输入是否应开启有意义的对话轮次。"
    '仅返回 JSON 对象 {"decision":"accept"} 或 {"decision":"discard"}。'
)


@dataclass(frozen=True, slots=True)
class AsrGateRequest:
    """类契约说明.

    职责: 保存 AsrGateRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: transcript、instruction。
    """

    transcript: str

    instruction: str = _INSTRUCTION


class AsrGateProvider(Protocol):
    """类契约说明.

    职责: 声明 AsrGateProvider
    协议接口,约束实现方必须提供的行为。
    契约: 方法: __call__。
    """

    def __call__(self, request: AsrGateRequest) -> str:
        """函数契约说明.

        功能: 执行 __call__ 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 request:
        AsrGateRequest。 必填。
        契约: 同步调用。 返回 `str`。
        """
        ...


@unique
class AsrGateDecision(StrEnum):
    """类契约说明.

    职责: 定义 AsrGateDecision
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    ACCEPT = "accept"

    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class AsrSemanticGate:
    """类契约说明.

    职责: 保存 AsrSemanticGate
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: provider。 方法: evaluate。
    """

    provider: AsrGateProvider

    def evaluate(self, transcript: str) -> AsrGateDecision:
        """函数契约说明.

        功能: 执行 evaluate 的同步逻辑,并协调
        parse_json_value, provider,
        isinstance, set。
        参数: self 表示当前实例。 transcript:
        str。 必填。
        契约: 同步调用。 返回 `AsrGateDecision`。
        """
        try:
            value = parse_json_value(self.provider(AsrGateRequest(transcript)))

        except (JsonBoundaryError, OSError, TimeoutError):
            return AsrGateDecision.DISCARD

        if not isinstance(value, dict) or set(value) != {"decision"}:
            return AsrGateDecision.DISCARD

        decision = value["decision"]

        match decision:
            case "accept":
                return AsrGateDecision.ACCEPT

            case "discard":
                return AsrGateDecision.DISCARD

            case _:
                return AsrGateDecision.DISCARD
