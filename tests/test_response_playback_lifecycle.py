"""A started playback must reach a verified end or be released by its owner.

An open playback window makes every later ASR final look like speech that
arrived during the agent's own audio, so a window that is never closed silently
mutes the whole session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orchestrator.brain_contracts import AudienceInput, AudienceSource
from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.intent_router import IntentRouter
from orchestrator.interactions import CommentProposal
from orchestrator.response_contracts import BrainDecision, ResponseProposal
from orchestrator.response_coordinator import AsyncResponseCoordinator
from orchestrator.runtime_contracts import RuntimeOutcome
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from orchestrator.brain_contracts import BrainStateSnapshot

_SESSION_ID = "session-playback"


@dataclass
class _AcceptingBrain:
    speech: str = "这是树莓娘的回答。"

    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        available_operations: tuple[dict[str, object], ...],
        observation: str | None = None,
    ) -> ResponseProposal:
        _ = snapshot, available_operations, observation
        return ResponseProposal(BrainDecision.ACCEPT, self.speech, None)


class _Tools:
    async def execute(self, request: object, snapshot: BrainStateSnapshot) -> None:
        _ = request, snapshot


def _runtime(*, response_task_timeout_ms: int = 30_000) -> SessionRuntime:
    runtime = SessionRuntime.create(
        session_id=SessionId(_SESSION_ID),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 2),
        response_task_timeout_ms=response_task_timeout_ms,
    )
    runtime.async_response_coordinator = AsyncResponseCoordinator(
        _AcceptingBrain(), IntentRouter(()), _Tools()
    )
    return runtime


def _correlation(sequence: int) -> EventCorrelation:
    return EventCorrelation(
        TraceId(f"trace-{sequence}"), SessionId(_SESSION_ID), EventSequence(sequence)
    )


def _audience_input(sequence: int, source: AudienceSource) -> AudienceInput:
    return AudienceInput(
        _SESSION_ID,
        f"trace-{sequence}",
        sequence,
        source,
        sequence,
        f"input-{sequence}",
    )


def _admit(
    correlation: EventCorrelation,
) -> Callable[
    [ResponseProposal, BrainStateSnapshot], Coroutine[None, None, RuntimeOutcome]
]:
    async def admission(
        _proposal: ResponseProposal, _snapshot: BrainStateSnapshot
    ) -> RuntimeOutcome:
        return RuntimeOutcome(accepted=True, correlation=correlation)

    return admission


async def _accepted_turn(runtime: SessionRuntime, sequence: int) -> TurnId:
    outcome = await runtime.receive_comment_async(
        CommentProposal(f"介绍一下网协-{sequence}", _correlation(sequence))
    )
    assert outcome.accepted
    assert outcome.turn_id is not None
    return outcome.turn_id


def test_provider_failure_after_the_first_frame_releases_playback() -> None:
    """A broken stream must not leave every later utterance looking like barge-in."""

    async def scenario() -> None:
        runtime = _runtime()
        clock = [1_000]
        runtime.clock = lambda: clock[0]
        turn_id = await _accepted_turn(runtime, 1)

        first_frames: list[bool] = []

        async def synthesize(_text: str, first_frame: Callable[[], bool]) -> bool:
            first_frames.append(first_frame())
            message = "provider stream broke"
            raise OSError(message)

        assert not await runtime.run_agent_tts_for_turn(
            turn_id, synthesize, _correlation(1)
        )
        assert first_frames == [True]

        clock[0] += 5_000
        coordinator = runtime.async_response_coordinator
        assert coordinator is not None
        outcome = await runtime._brain_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
            coordinator,
            _audience_input(2, AudienceSource.ASR),
            _correlation(2),
            _admit(_correlation(2)),
        )

        assert not runtime._was_playing_at(  # pyright: ignore[reportPrivateUsage]
            clock[0] - 1_000
        )
        assert runtime.response_turn_state.phase != "playing"
        assert outcome.accepted

    asyncio.run(scenario())


def test_started_playback_outlives_the_response_deadline() -> None:
    """Paced media outlives its turn deadline; provider startup does not."""

    async def scenario() -> None:
        runtime = _runtime(response_task_timeout_ms=20)
        clock = [1_000]
        runtime.clock = lambda: clock[0]
        turn_id = await _accepted_turn(runtime, 1)
        first_frames: list[bool] = []

        async def synthesize(_text: str, first_frame: Callable[[], bool]) -> bool:
            first_frames.append(first_frame())
            await asyncio.sleep(0.2)
            return True

        assert await runtime.run_agent_tts_for_turn(
            turn_id, synthesize, _correlation(1)
        )
        assert first_frames == [True]
        assert runtime.response_turn_state.phase == "playing"

    asyncio.run(scenario())
