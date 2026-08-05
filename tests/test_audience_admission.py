import asyncio

from orchestrator.brain_contracts import AudienceInput, AudienceSource, GateDecision
from orchestrator.ids import SessionId, TraceId
from orchestrator.runtime_contracts import RuntimeOutcome
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind


class _AcceptGate:
    async def evaluate(
        self,
        audience_input: AudienceInput,
        *,
        active_summary: str,
        recent_turn_context: tuple[str, ...] = (),
    ) -> GateDecision:
        _ = audience_input, active_summary, recent_turn_context
        await asyncio.sleep(0)
        return GateDecision.ACCEPT


class _FailOnceGate:
    def __init__(self) -> None:
        self.failed: bool = False

    async def evaluate(
        self,
        audience_input: AudienceInput,
        *,
        active_summary: str,
        recent_turn_context: tuple[str, ...] = (),
    ) -> GateDecision:
        _ = audience_input, active_summary, recent_turn_context
        if not self.failed:
            self.failed = True
            raise RuntimeError
        return GateDecision.ACCEPT


def _runtime() -> SessionRuntime:
    return SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset({TaskKind.INTERACTIVE}), 2),
    )


def _correlation(sequence: int) -> EventCorrelation:
    return EventCorrelation(
        TraceId(f"trace-{sequence}"),
        SessionId("session-1"),
        EventSequence(sequence),
    )


def _input(sequence: int, source: AudienceSource) -> AudienceInput:
    return AudienceInput(
        session_id="session-1",
        trace_id=f"trace-{sequence}",
        sequence=sequence,
        source=source,
        received_at_ms=sequence,
        text=f"input-{sequence}",
    )


def test_voice_is_admitted_before_queued_comments() -> None:
    async def scenario() -> None:
        runtime = _runtime()
        gate = _AcceptGate()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        order: list[str] = []

        async def first() -> RuntimeOutcome:
            order.append("comment-1")
            _ = first_started.set()
            _ = await release_first.wait()
            return RuntimeOutcome(accepted=True, correlation=_correlation(1))

        async def record(name: str, sequence: int) -> RuntimeOutcome:
            order.append(name)
            return RuntimeOutcome(
                accepted=True, correlation=_correlation(sequence)
            )

        first_task = asyncio.create_task(
            runtime._gate_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
                gate,
                _input(1, AudienceSource.COMMENT),
                _correlation(1),
                first,
            )
        )
        _ = await first_started.wait()
        comment_task = asyncio.create_task(
            runtime._gate_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
                gate,
                _input(2, AudienceSource.COMMENT),
                _correlation(2),
                lambda: record("comment-2", 2),
            )
        )
        voice_task = asyncio.create_task(
            runtime._gate_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
                gate,
                _input(3, AudienceSource.ASR),
                _correlation(3),
                lambda: record("voice-3", 3),
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        _ = release_first.set()
        _ = await asyncio.gather(first_task, comment_task, voice_task)

        assert order == ["comment-1", "voice-3", "comment-2"]

    asyncio.run(scenario())


def test_gate_exception_fails_closed_and_later_input_survives() -> None:
    async def scenario() -> None:
        runtime = _runtime()
        gate = _FailOnceGate()

        failed = await runtime._gate_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
            gate,
            _input(1, AudienceSource.COMMENT),
            _correlation(1),
            lambda: asyncio.sleep(
                0,
                result=RuntimeOutcome(
                    accepted=True, correlation=_correlation(1)
                ),
            ),
        )
        accepted = await runtime._gate_and_enqueue_audience(  # pyright: ignore[reportPrivateUsage]
            gate,
            _input(2, AudienceSource.COMMENT),
            _correlation(2),
            lambda: asyncio.sleep(
                0,
                result=RuntimeOutcome(
                    accepted=True, correlation=_correlation(2)
                ),
            ),
        )

        assert failed.accepted is False
        assert accepted.accepted is True

    asyncio.run(scenario())
