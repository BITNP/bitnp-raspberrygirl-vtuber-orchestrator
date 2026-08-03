
from collections import deque
from dataclasses import dataclass, field
from typing import Final

from orchestrator.task_registry import (
    TaskId,
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

    registry: TaskRegistry

    max_pending_per_lane: int

    _pending: dict[TaskKind, deque[TaskRequest]] = field(
        default_factory=lambda: {kind: deque() for kind in TaskKind}
    )

    def enqueue(self, request: TaskRequest) -> bool:
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
                | TaskState.RUNNING
            ):
                return False

    def claim(self, task_id: TaskId) -> TaskRequest | None:
        """Claim one queued task for a dedicated worker path."""
        for lane in self._pending.values():
            for request in lane:
                if request.task_id == task_id:
                    lane.remove(request)
                    return request if self.registry.claim(task_id) is not None else None
        return None

    def next(self, *, now_ms: int) -> TaskRequest | None:
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

                        if self.registry.claim(request.task_id) is not None:
                            return request
                        continue

                    case (
                        TaskState.CANCELLED
                        | TaskState.SUPERSEDED
                        | TaskState.TIMED_OUT
                        | TaskState.COMPLETED
                        | TaskState.RUNNING
                    ):
                        continue

        return None
