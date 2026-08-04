from orchestrator.agent_state import (
    GateOutcome,
    StateEffect,
    TurnCoordinator,
    TurnPhase,
)


def test_interrupt_cancels_reasoning_but_keeps_sound_until_new_audio_is_ready() -> None:
    reducer = TurnCoordinator()
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
    reducer = TurnCoordinator()
    _ = reducer.gate(GateOutcome.ACCEPT)
    _ = reducer.reasoning_complete(0, has_text=True)
    _ = reducer.audio_ready(0)
    _ = reducer.gate(GateOutcome.INTERRUPT)

    assert reducer.reasoning_complete(0, has_text=True).effects == ()
    assert reducer.audio_ready(0).effects == ()


def test_failed_pre_audio_turn_has_no_recovery_effect() -> None:
    reducer = TurnCoordinator()
    _ = reducer.gate(GateOutcome.ACCEPT)

    failed = reducer.failed(0, audio_started=False)
    assert failed.state.phase is TurnPhase.FAILED
    assert failed.effects == ()


def test_runtime_turn_lifecycle_rejects_stale_provider_callbacks() -> None:
    coordinator = TurnCoordinator()

    queued = coordinator.enqueue(turn_id="turn-1", epoch=7)
    assert queued.state.phase is TurnPhase.QUEUED
    reasoning = coordinator.start_reasoning(turn_id="turn-1", epoch=7)
    assert reasoning.state.phase is TurnPhase.REASONING
    waiting = coordinator.wait_for_tool(turn_id="turn-1", epoch=7)
    assert waiting.state.phase is TurnPhase.WAITING_TOOL
    resumed = coordinator.resume_reasoning(turn_id="turn-1", epoch=7)
    assert resumed.state.phase is TurnPhase.REASONING
    synthesizing = coordinator.start_synthesizing(turn_id="turn-1", epoch=7)
    assert synthesizing.state.phase is TurnPhase.SYNTHESIZING

    stale = coordinator.playback_started(turn_id="turn-1", epoch=6)
    assert stale.state.phase is TurnPhase.SYNTHESIZING
    assert stale.effects == ()

    playing = coordinator.playback_started(turn_id="turn-1", epoch=7)
    assert playing.state.phase is TurnPhase.PLAYING
    completed = coordinator.playback_finished(turn_id="turn-1", epoch=7)
    assert completed.state.phase is TurnPhase.COMPLETED


def test_runtime_replacement_requires_cutover_before_playback() -> None:
    coordinator = TurnCoordinator()
    _ = coordinator.enqueue(turn_id="turn-2", epoch=8, replacement=True)
    _ = coordinator.start_reasoning(turn_id="turn-2", epoch=8)
    _ = coordinator.start_synthesizing(turn_id="turn-2", epoch=8)

    pending = coordinator.await_cutover(turn_id="turn-2", epoch=8)
    assert pending.state.phase is TurnPhase.CUTOVER_PENDING
    assert pending.effects == (StateEffect.FLUSH_SOUND,)
    playing = coordinator.playback_started(turn_id="turn-2", epoch=8)
    assert playing.state.phase is TurnPhase.PLAYING


def test_rejected_replacement_restores_the_retained_playback_turn() -> None:
    coordinator = TurnCoordinator()
    _ = coordinator.enqueue(turn_id="turn-old", epoch=7)
    _ = coordinator.start_reasoning(turn_id="turn-old", epoch=7)
    _ = coordinator.start_synthesizing(turn_id="turn-old", epoch=7)
    _ = coordinator.playback_started(turn_id="turn-old", epoch=7)
    _ = coordinator.enqueue(turn_id="turn-new", epoch=8, replacement=True)
    _ = coordinator.start_reasoning(turn_id="turn-new", epoch=8)
    _ = coordinator.start_synthesizing(turn_id="turn-new", epoch=8)
    _ = coordinator.await_cutover(turn_id="turn-new", epoch=8)

    restored = coordinator.restore_retained_playback(turn_id="turn-new", epoch=8)

    assert restored.state.phase is TurnPhase.PLAYING
    assert restored.state.turn_id == "turn-old"
    assert restored.state.retained_playback_turn_id is None
