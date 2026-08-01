from orchestrator.agent_state import (
    AgentStateReducer,
    GateOutcome,
    StateEffect,
    TurnPhase,
)


def test_interrupt_cancels_reasoning_but_keeps_sound_until_new_audio_is_ready() -> None:
    reducer = AgentStateReducer()
    assert reducer.gate(GateOutcome.ACCEPT).effects == (StateEffect.START_REASONING,)
    assert reducer.reasoning_complete(0, has_text=True).effects == (
        StateEffect.START_TTS,
    )
    assert reducer.audio_ready(0).effects == (StateEffect.EMIT_AUDIO,)

    interrupt = reducer.gate(GateOutcome.INTERRUPT)
    assert interrupt.state.phase is TurnPhase.REASONING
    assert interrupt.effects == (
        StateEffect.CANCEL_DELIBERATIVE,
        StateEffect.START_REASONING,
    )

    _ = reducer.reasoning_complete(1, has_text=True)
    ready = reducer.audio_ready(1)
    assert ready.state.phase is TurnPhase.CUTOVER_PENDING
    assert ready.effects == (StateEffect.FLUSH_SOUND,)
    assert reducer.flush_acknowledged(1).effects == (StateEffect.EMIT_AUDIO,)


def test_stale_callbacks_cannot_emit_audio_after_interrupt() -> None:
    reducer = AgentStateReducer()
    _ = reducer.gate(GateOutcome.ACCEPT)
    _ = reducer.reasoning_complete(0, has_text=True)
    _ = reducer.audio_ready(0)
    _ = reducer.gate(GateOutcome.INTERRUPT)

    assert reducer.reasoning_complete(0, has_text=True).effects == ()
    assert reducer.audio_ready(0).effects == ()


def test_failed_pre_audio_turn_has_no_recovery_effect() -> None:
    reducer = AgentStateReducer()
    _ = reducer.gate(GateOutcome.ACCEPT)

    failed = reducer.failed(0, audio_started=False)
    assert failed.state.phase is TurnPhase.FAILED
    assert failed.effects == ()
