"""模块契约说明.

职责: 提供 orchestrator.runtime_contracts
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass

from orchestrator.ids import TurnId
from orchestrator.sessions import EventCorrelation, SessionSnapshot
from orchestrator.task_reducer import TaskResult


@dataclass(frozen=True, slots=True)
class RuntimeDispatch:
    """类契约说明.

    职责: 保存 RuntimeDispatch
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: correlation、turn_id。
    """

    correlation: EventCorrelation

    turn_id: TurnId


@dataclass(frozen=True, slots=True)
class RuntimeRejection:
    """类契约说明.

    职责: 保存 RuntimeRejection
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: correlation、reason。
    """

    correlation: EventCorrelation

    reason: str


@dataclass(frozen=True, slots=True)
class RuntimeObservables:
    """类契约说明.

    职责: 保存 RuntimeObservables
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: snapshot、dispatches、task_com
    mits、generated_rtp、sound_transitions
    、rejections。
    """

    snapshot: SessionSnapshot

    dispatches: tuple[RuntimeDispatch, ...]

    task_commits: tuple[TaskResult, ...]

    generated_rtp: tuple[bytes, ...]

    sound_transitions: tuple[str, ...]

    rejections: tuple[RuntimeRejection, ...]


@dataclass(frozen=True, slots=True)
class RuntimeOutcome:
    """类契约说明.

    职责: 保存 RuntimeOutcome
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    accepted、correlation、turn_id。
    """

    accepted: bool

    correlation: EventCorrelation

    turn_id: TurnId | None = None
