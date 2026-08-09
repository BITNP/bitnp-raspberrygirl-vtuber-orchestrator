import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from orchestrator.brain_contracts import BrainStateSnapshot
from orchestrator.ids import SessionId, TraceId
from orchestrator.intent_router import IntentRouter
from orchestrator.interactions import CommentProposal
from orchestrator.response_contracts import BrainDecision, ResponseProposal
from orchestrator.response_coordinator import AsyncResponseCoordinator
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence
from orchestrator.task_registry import (
    SchedulerTaskConfig,
    TaskId,
    TaskKind,
    TaskRecord,
    TaskState,
)


@dataclass(frozen=True)
class _Brain:
    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        available_operations: tuple[dict[str, object], ...],
        observation: str | None = None,
    ) -> ResponseProposal:
        _ = snapshot, available_operations, observation
        return ResponseProposal(BrainDecision.ACCEPT, "我记住你的研究方向了。", None)


@dataclass(frozen=True)
class _Tools:
    async def execute(self, request: object, snapshot: BrainStateSnapshot) -> None:
        _ = request, snapshot


@dataclass(frozen=True)
class _Extractor:
    raw: str

    async def extract(self, *, user_text: str, reply_text: str) -> str:
        assert user_text == "我想研究通用人工智能并且现在要找导师"
        assert reply_text == "我记住你的研究方向了。"
        return self.raw


def test_memory_maintenance_succeeds_only_after_candidate_is_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    asyncio.run(
        _memory_maintenance_scenario(
            json.dumps(
                {
                    "decision": "remember",
                    "key": "research_goal",
                    "value": "研究通用人工智能并寻找导师",
                    "confidence": 95,
                },
                ensure_ascii=False,
            ),
            expected_state=TaskState.SUCCEEDED,
            expected_reason=None,
            expected_revision=1,
        )
    )


def test_memory_maintenance_exposes_policy_rejection_as_task_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    asyncio.run(
        _memory_maintenance_scenario(
            json.dumps(
                {
                    "decision": "remember",
                    "key": "research_goal",
                    "value": "可能对人工智能感兴趣",
                    "confidence": 70,
                },
                ensure_ascii=False,
            ),
            expected_state=TaskState.FAILED,
            expected_reason="memory_candidate_unsupported_assertion",
            expected_revision=0,
        )
    )


async def _memory_maintenance_scenario(
    raw: str,
    *,
    expected_state: TaskState,
    expected_reason: str | None,
    expected_revision: int,
) -> None:
    session_id = SessionId("session-memory-maintenance")
    runtime = SessionRuntime.create(
        session_id=session_id,
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        async_response_coordinator=AsyncResponseCoordinator(
            _Brain(), IntentRouter(()), _Tools()
        ),
        memory_candidate_extractor=_Extractor(raw),
    )
    correlation = EventCorrelation(
        TraceId("trace-memory-maintenance"), session_id, EventSequence(1)
    )

    outcome = await runtime.receive_comment_async(
        CommentProposal("我想研究通用人工智能并且现在要找导师", correlation)
    )
    assert outcome.accepted
    assert outcome.turn_id is not None
    task_id = TaskId(f"memory-extract-{outcome.turn_id}")
    record = await _wait_for_terminal_task(runtime, task_id)

    assert record.state is expected_state
    assert record.cancellation_reason == expected_reason
    memory = runtime.interaction_ingress.data.memory.snapshot
    assert memory.revision == expected_revision
    if expected_revision == 0:
        assert memory.entries == ()
    else:
        assert len(memory.entries) == 1
        assert memory.entries[0].key == "research_goal"
        assert memory.entries[0].value == "研究通用人工智能并寻找导师"


async def _wait_for_terminal_task(
    runtime: SessionRuntime, task_id: TaskId
) -> TaskRecord:
    async with asyncio.timeout(1.0):
        while True:
            record = runtime.task_registry.task(task_id)
            if record is not None and record.state in {
                TaskState.SUCCEEDED,
                TaskState.FAILED,
            }:
                return record
            await asyncio.sleep(0)
