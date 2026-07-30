"""Scheduler-facing task admission facade."""

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
    """Bind task registration and effect admission to one session scheduler."""

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
        """Create scheduler-owned task admission state for one session."""
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
        """Return the current scheduler-owned version vector for data dependencies."""
        provider = self.data_snapshot_provider
        if provider is None:
            return self.initial_data_snapshot
        return provider()

    def schedule(self, request: TaskRequest) -> TaskRegistrationResult:
        """Register work only against the scheduler's current turn snapshot."""
        rejection = _scheduling_rejection(request, self.scheduler.snapshot)
        if rejection is not None:
            return TaskRegistrationRejected(rejection)
        if request.data_snapshot != self.data_snapshot:
            return TaskRegistrationRejected(TaskRegistrationRejection.STALE_SNAPSHOT)
        return self.registry.register(request)

    def reduce(self, result: TaskResult, *, now_ms: int) -> TaskReductionResult:
        """Admit effects only through live scheduler reducer validation."""
        return self.reducer.reduce(
            result,
            snapshot=self.scheduler.snapshot,
            data_snapshot=self.data_snapshot,
            now_ms=now_ms,
        )


def _scheduling_rejection(
    request: TaskRequest, snapshot: SessionSnapshot
) -> TaskRegistrationRejection | None:
    if request.session_id != snapshot.session_id:
        return TaskRegistrationRejection.SESSION_MISMATCH
    if request.turn_id != snapshot.active_turn_id:
        return TaskRegistrationRejection.ACTIVE_TURN_MISMATCH
    if request.snapshot_revision != snapshot.revision:
        return TaskRegistrationRejection.STALE_SNAPSHOT
    return None
