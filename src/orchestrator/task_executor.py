"""Bounded priority selection for scheduler-authorized task records."""

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
    """Retain bounded pending work and select it in scheduler lane priority order."""

    registry: TaskRegistry
    max_pending_per_lane: int
    _pending: dict[TaskKind, deque[TaskRequest]] = field(
        default_factory=lambda: {kind: deque() for kind in TaskKind}
    )

    def enqueue(self, request: TaskRequest) -> bool:
        """Retain an already-admitted pending task while its lane has capacity."""
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
        """Return one current task, expiring stale queued work before execution."""
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
