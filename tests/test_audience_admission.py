import asyncio
from dataclasses import dataclass, replace

from orchestrator.brain_contracts import (
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
)
from orchestrator.ids import SessionId, TraceId
from orchestrator.intent_router import IntentRouter
from orchestrator.response_contracts import BrainDecision, ResponseProposal
from orchestrator.response_coordinator import AsyncResponseCoordinator
from orchestrator.runtime_contracts import RuntimeOutcome
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import (
    EventCorrelation,
    EventSequence,
    SchedulerEvent,
    StartTurn,
)
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind


@dataclass
class _Brain:
    proposal: ResponseProposal
    started: asyncio.Event | None = None
    release: asyncio.Event | None = None

    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        available_operations: tuple[dict[str, object], ...],
        observation: str | None = None,
    ) -> ResponseProposal:
        _ = snapshot, available_operations, observation
        if self.started is not None:
            _ = self.started.set()
        if self.release is not None:
            _ = await self.release.wait()
        return self.proposal


class _Tools:
    async def execute(self, request: object, snapshot: BrainStateSnapshot) -> None:
        _ = request, snapshot


@dataclass
class _OrderedBrain:
    first_started: asyncio.Event
    release_first: asyncio.Event
    calls: list[int]

    async def respond(
        self,
        snapshot: BrainStateSnapshot,
        *,
        available_operations: tuple[dict[str, object], ...],
        observation: str | None = None,
    ) -> ResponseProposal:
        _ = available_operations, observation
        self.calls.append(snapshot.input.sequence)
        if len(self.calls) == 1:
            self.first_started.set()
            _ = await self.release_first.wait()
        return ResponseProposal(BrainDecision.ACCEPT, "回答", None)


def _runtime(brain: _Brain) -> SessionRuntime:
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 2),
    )
    runtime.async_response_coordinator = AsyncResponseCoordinator(
        brain, IntentRouter(()), _Tools()
    )
    return runtime


def _correlation(sequence: int) -> EventCorrelation:
    return EventCorrelation(
        TraceId(f"trace-{sequence}"), SessionId("session-1"), EventSequence(sequence)
    )


def _input(
    sequence: int, source: AudienceSource = AudienceSource.COMMENT
) -> AudienceInput:
    return AudienceInput(
        "session-1",
        f"trace-{sequence}",
        sequence,
        source,
        sequence,
        f"input-{sequence}",
    )


def test_discard_candidate_does_not_create_turn_or_advance_epoch() -> None:
    async def scenario() -> None:
        runtime = _runtime(_Brain(ResponseProposal(BrainDecision.DISCARD, "", None)))
        revision = runtime.scheduler.snapshot.revision
        epoch = runtime.cancellation_epoch
        coordinator = runtime.async_response_coordinator
        assert coordinator is not None
        outcome = await runtime._brain_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
            coordinator,
            _input(1),
            _correlation(1),
            lambda _proposal, _snapshot: asyncio.sleep(
                0,
                result=RuntimeOutcome(
                    accepted=True, correlation=_correlation(1)
                ),
            ),
        )
        assert not outcome.accepted
        assert runtime.scheduler.snapshot.revision == revision
        assert runtime.cancellation_epoch == epoch

    asyncio.run(scenario())


def test_candidate_revalidates_scheduler_revision_before_admission() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        runtime = _runtime(
            _Brain(
                ResponseProposal(BrainDecision.ACCEPT, "回答", None), started, release
            )
        )
        coordinator = runtime.async_response_coordinator
        assert coordinator is not None
        task = asyncio.create_task(
            runtime._brain_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
                coordinator,
                _input(1),
                _correlation(1),
                lambda _proposal, _snapshot: asyncio.sleep(
                    0,
                    result=RuntimeOutcome(
                        accepted=True, correlation=_correlation(1)
                    ),
                ),
            )
        )
        _ = await started.wait()
        snapshot = runtime.scheduler.snapshot
        _ = runtime.scheduler.apply(
            StartTurn(snapshot.revision, SchedulerEvent("test", _correlation(99)))
        )
        _ = release.set()
        assert not (await task).accepted

    asyncio.run(scenario())


def test_duplicate_candidate_is_rejected_while_first_is_in_flight() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        runtime = _runtime(
            _Brain(
                ResponseProposal(BrainDecision.ACCEPT, "回答", None), started, release
            )
        )
        coordinator = runtime.async_response_coordinator
        assert coordinator is not None
        first = asyncio.create_task(
            runtime._brain_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
                coordinator,
                _input(1),
                _correlation(1),
                lambda _proposal, _snapshot: asyncio.sleep(
                    0,
                    result=RuntimeOutcome(
                        accepted=True, correlation=_correlation(1)
                    ),
                ),
            )
        )
        _ = await started.wait()
        duplicate = await runtime._brain_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
            coordinator,
            _input(1),
            _correlation(1),
            lambda _proposal, _snapshot: asyncio.sleep(
                0,
                result=RuntimeOutcome(
                    accepted=True, correlation=_correlation(1)
                ),
            ),
        )
        assert not duplicate.accepted
        _ = release.set()
        assert (await first).accepted

    asyncio.run(scenario())


def test_candidates_are_serialized_before_brain_with_voice_priority() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        brain = _OrderedBrain(started, release, [])
        runtime = SessionRuntime.create(
            session_id=SessionId("session-1"),
            turn_id_prefix="turn",
            task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 2),
        )
        coordinator = AsyncResponseCoordinator(brain, IntentRouter(()), _Tools())
        runtime.async_response_coordinator = coordinator

        def submit(
            sequence: int, source: AudienceSource
        ) -> asyncio.Task[RuntimeOutcome]:
            return asyncio.create_task(
                runtime._brain_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
                    coordinator,
                    _input(sequence, source),
                    _correlation(sequence),
                    lambda _proposal, _snapshot: asyncio.sleep(
                        0,
                        result=RuntimeOutcome(
                            accepted=True, correlation=_correlation(sequence)
                        ),
                    ),
                )
            )

        first = submit(1, AudienceSource.COMMENT)
        _ = await started.wait()
        second_comment = submit(2, AudienceSource.COMMENT)
        await asyncio.sleep(0)
        voice = submit(3, AudienceSource.ASR)
        await asyncio.sleep(0)

        assert brain.calls == [1]
        release.set()
        outcomes = await asyncio.gather(first, second_comment, voice)

        assert all(outcome.accepted for outcome in outcomes)
        assert brain.calls == [1, 3, 2]

    asyncio.run(scenario())


def test_playback_policy_is_frozen_at_enqueue_and_cannot_be_overridden() -> None:
    async def scenario() -> None:
        runtime = _runtime(
            _Brain(ResponseProposal(BrainDecision.ACCEPT, "错误接受", None))
        )
        runtime.clock = lambda: 5_000
        runtime._playback_intervals.append(  # pyright: ignore[reportPrivateUsage]
            (3_500, 4_500)
        )
        coordinator = runtime.async_response_coordinator
        assert coordinator is not None

        outcome = await runtime._brain_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
            coordinator,
            replace(_input(1, AudienceSource.ASR), text="我在听请继续讲"),
            _correlation(1),
            lambda _proposal, _snapshot: asyncio.sleep(
                0,
                result=RuntimeOutcome(accepted=True, correlation=_correlation(1)),
            ),
        )

        assert not outcome.accepted
        assert runtime.observables.rejections[-1].reason == (
            "brain_playback_policy_violated"
        )

    asyncio.run(scenario())


def test_explicit_interruption_can_be_accepted_during_playback_policy() -> None:
    async def scenario() -> None:
        runtime = _runtime(_Brain(ResponseProposal(BrainDecision.ACCEPT, "好的", None)))
        runtime.clock = lambda: 5_000
        runtime._playback_intervals.append(  # pyright: ignore[reportPrivateUsage]
            (3_500, 4_500)
        )
        coordinator = runtime.async_response_coordinator
        assert coordinator is not None

        outcome = await runtime._brain_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
            coordinator,
            replace(_input(1, AudienceSource.ASR), text="停一下"),
            _correlation(1),
            lambda _proposal, _snapshot: asyncio.sleep(
                0,
                result=RuntimeOutcome(accepted=True, correlation=_correlation(1)),
            ),
        )

        assert outcome.accepted

    asyncio.run(scenario())
