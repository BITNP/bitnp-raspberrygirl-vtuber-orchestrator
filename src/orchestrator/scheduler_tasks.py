"""模块契约说明.

职责: 提供 orchestrator.scheduler_tasks
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from collections.abc import Callable
from dataclasses import dataclass

from orchestrator.sessions import SessionScheduler, SessionSnapshot
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.task_reducer import TaskReductionResult, TaskResult, TaskResultReducer
from orchestrator.task_registry import (
    SchedulerTaskConfig,
    TaskRegistrationRejected,
    TaskRegistrationRejection,
    TaskRegistrationResult,
    TaskRegistry,
    TaskRequest,
)


@dataclass(frozen=True, slots=True)
class SchedulerTaskFacade:
    """类契约说明.

    职责: 保存 SchedulerTaskFacade
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: scheduler、registry、reducer、i
    nitial_data_snapshot、data_snapshot_p
    rovider。 方法: create、data_snapshot、sc
    hedule、reduce。
    """

    scheduler: SessionScheduler

    registry: TaskRegistry

    reducer: TaskResultReducer

    initial_data_snapshot: TaskStateSnapshot

    data_snapshot_provider: Callable[[], TaskStateSnapshot] | None = None

    @classmethod
    def create(
        cls,
        scheduler: SessionScheduler,
        config: SchedulerTaskConfig,
        data_snapshot: TaskStateSnapshot | None = None,
        data_snapshot_provider: Callable[[], TaskStateSnapshot] | None = None,
    ) -> "SchedulerTaskFacade":
        """函数契约说明.

        功能: 执行 create 的同步逻辑,并协调
        TaskRegistry, cls,
        TaskResultReducer, initial。
        参数: cls 表示当前类。 scheduler:
        SessionScheduler。 必填。 config:
        SchedulerTaskConfig。 必填。
        data_snapshot: TaskStateSnapshot
        | None。 可省略。
        data_snapshot_provider:
        Callable[[], TaskStateSnapshot]
        | None。 可省略。
        契约: 同步调用。 返回
        `'SchedulerTaskFacade'`。
        """
        registry = TaskRegistry(session_id=scheduler.snapshot.session_id, config=config)

        return cls(
            scheduler,
            registry,
            TaskResultReducer(registry),
            data_snapshot or TaskStateSnapshot.initial(),
            data_snapshot_provider,
        )

    @property
    def data_snapshot(self) -> TaskStateSnapshot:
        """函数契约说明.

        功能: 执行 data_snapshot 的同步逻辑,并协调
        provider。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `TaskStateSnapshot`。
        """
        provider = self.data_snapshot_provider

        if provider is None:
            return self.initial_data_snapshot

        return provider()

    def schedule(self, request: TaskRequest) -> TaskRegistrationResult:
        """函数契约说明.

        功能: 执行 schedule 的同步逻辑,并协调
        _scheduling_rejection, register,
        TaskRegistrationRejected。
        参数: self 表示当前实例。 request:
        TaskRequest。 必填。
        契约: 同步调用。 返回
        `TaskRegistrationResult`。
        """
        rejection = _scheduling_rejection(request, self.scheduler.snapshot)

        if rejection is not None:
            return TaskRegistrationRejected(rejection)

        if request.data_snapshot != self.data_snapshot:
            return TaskRegistrationRejected(TaskRegistrationRejection.STALE_SNAPSHOT)

        return self.registry.register(request)

    def reduce(self, result: TaskResult, *, now_ms: int) -> TaskReductionResult:
        """函数契约说明.

        功能: 执行 reduce 的同步逻辑,并协调 reduce。
        参数: self 表示当前实例。 result:
        TaskResult。 必填。 now_ms: int。 必填。
        契约: 同步调用。 返回
        `TaskReductionResult`。
        """
        return self.reducer.reduce(
            result,
            snapshot=self.scheduler.snapshot,
            data_snapshot=self.data_snapshot,
            now_ms=now_ms,
        )


def _scheduling_rejection(
    request: TaskRequest, snapshot: SessionSnapshot
) -> TaskRegistrationRejection | None:
    """函数契约说明.

    功能: 执行 _scheduling_rejection
    的同步逻辑,并维持签名契约。
    参数: request: TaskRequest。 必填。
    snapshot: SessionSnapshot。 必填。
    契约: 同步调用。 返回
    `TaskRegistrationRejection | None`。
    """
    if request.session_id != snapshot.session_id:
        return TaskRegistrationRejection.SESSION_MISMATCH

    if request.turn_id != snapshot.active_turn_id:
        return TaskRegistrationRejection.ACTIVE_TURN_MISMATCH

    if request.snapshot_revision != snapshot.revision:
        return TaskRegistrationRejection.STALE_SNAPSHOT

    return None
