"""模块契约说明.

职责: 提供 orchestrator.task_executor
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Final

from orchestrator.task_registry import (
    TaskKind,
    TaskRegistry,
    TaskRequest,
    TaskState,
)

_LANE_PRIORITY: Final = (
    TaskKind.REFLEX,
    TaskKind.INTERACTIVE,
    TaskKind.DELIBERATIVE,
    TaskKind.MAINTENANCE,
)


@dataclass(slots=True)
class TaskLaneExecutor:
    """类契约说明.

    职责: 保存 TaskLaneExecutor
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: registry、max_pending_per_lan
    e、_pending。 方法: enqueue、next。
    """

    registry: TaskRegistry

    max_pending_per_lane: int

    _pending: dict[TaskKind, deque[TaskRequest]] = field(
        default_factory=lambda: {kind: deque() for kind in TaskKind}
    )

    def enqueue(self, request: TaskRequest) -> bool:
        """函数契约说明.

        功能: 执行 enqueue 的同步逻辑,并协调 task,
        len, append。
        参数: self 表示当前实例。 request:
        TaskRequest。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        lane = self._pending[request.kind]

        if len(lane) >= self.max_pending_per_lane:
            return False

        record = self.registry.task(request.task_id)

        if record is None:
            return False

        match record.state:
            case TaskState.PENDING:
                lane.append(request)

                return True

            case (
                TaskState.CANCELLED
                | TaskState.SUPERSEDED
                | TaskState.TIMED_OUT
                | TaskState.COMPLETED
            ):
                return False

    def next(self, *, now_ms: int) -> TaskRequest | None:
        """函数契约说明.

        功能: 执行 next 的同步逻辑,并协调 popleft,
        task, timeout。
        参数: self 表示当前实例。 now_ms: int。
        必填。
        契约: 同步调用。 返回 `TaskRequest |
        None`。
        """
        for kind in _LANE_PRIORITY:
            lane = self._pending[kind]

            while lane:
                request = lane.popleft()

                record = self.registry.task(request.task_id)

                if record is None:
                    continue

                match record.state:
                    case TaskState.PENDING:
                        if now_ms > request.deadline_ms:
                            _ = self.registry.timeout(request.task_id)

                            continue

                        return request

                    case (
                        TaskState.CANCELLED
                        | TaskState.SUPERSEDED
                        | TaskState.TIMED_OUT
                        | TaskState.COMPLETED
                    ):
                        continue

        return None
