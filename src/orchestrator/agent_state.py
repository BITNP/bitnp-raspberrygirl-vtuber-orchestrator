"""Reducer-owned state for a single onsite conversational stream.

Provider callbacks are untrusted: callers must apply their result through this
reducer before emitting an effect.  The state is intentionally transport-free
so cancellation and media cut-over races can be tested deterministically.
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import final


@unique
class TurnPhase(StrEnum):
    IDLE = "idle"
    GATE_PENDING = "gate_pending"
    REASONING = "reasoning"
    TTS_PENDING = "tts_pending"
    AUDIO_READY = "audio_ready"
    CUTOVER_PENDING = "cutover_pending"
    PLAYING = "playing"
    FAILED = "failed"


@unique
class GateOutcome(StrEnum):
    DISCARD = "discard"
    ACCEPT = "accept"
    INTERRUPT = "interrupt"


@unique
class StateEffect(StrEnum):
    NONE = "none"
    START_REASONING = "start_reasoning"
    CANCEL_DELIBERATIVE = "cancel_deliberative"
    START_TTS = "start_tts"
    FLUSH_SOUND = "flush_sound"
    EMIT_AUDIO = "emit_audio"
    FINISH_AUDIO = "finish_audio"
    SHOW_FAILURE = "show_failure"


@dataclass(frozen=True, slots=True)
class AgentState:
    epoch: int = 0
    phase: TurnPhase = TurnPhase.IDLE
    pending_interrupt: bool = False


@dataclass(frozen=True, slots=True)
class StateTransition:
    state: AgentState
    effects: tuple[StateEffect, ...] = ()


@final
class AgentStateReducer:
    """Enforces delayed Sound interruption and rejects stale provider output."""

    def __init__(self) -> None:
        self._state = AgentState()

    @property
    def state(self) -> AgentState:
        return self._state

    def gate(self, outcome: GateOutcome) -> StateTransition:
        state = self._state
        if outcome is GateOutcome.DISCARD:
            return StateTransition(state)
        if outcome is GateOutcome.ACCEPT and state.phase is TurnPhase.IDLE:
            return self._set(
                AgentState(state.epoch, TurnPhase.REASONING),
                StateEffect.START_REASONING,
            )
        if outcome is GateOutcome.INTERRUPT and state.phase is TurnPhase.PLAYING:
            # Audio keeps playing.  Only non-media work is cancelled here.
            return self._set(
                AgentState(
                    state.epoch + 1,
                    TurnPhase.REASONING,
                    pending_interrupt=True,
                ),
                StateEffect.CANCEL_DELIBERATIVE,
                StateEffect.START_REASONING,
            )
        return StateTransition(state)

    def reasoning_complete(self, epoch: int, *, has_text: bool) -> StateTransition:
        if epoch != self._state.epoch or self._state.phase is not TurnPhase.REASONING:
            return StateTransition(self._state)
        if not has_text:
            return self._set(
                AgentState(epoch, TurnPhase.FAILED), StateEffect.SHOW_FAILURE
            )
        return self._set(
            AgentState(epoch, TurnPhase.TTS_PENDING, self._state.pending_interrupt),
            StateEffect.START_TTS,
        )

    def audio_ready(self, epoch: int) -> StateTransition:
        state = self._state
        if epoch != state.epoch or state.phase is not TurnPhase.TTS_PENDING:
            return StateTransition(state)
        if state.pending_interrupt:
            return self._set(
                AgentState(
                    epoch, TurnPhase.CUTOVER_PENDING, pending_interrupt=True
                ),
                StateEffect.FLUSH_SOUND,
            )
        return self._set(AgentState(epoch, TurnPhase.PLAYING), StateEffect.EMIT_AUDIO)

    def flush_acknowledged(self, epoch: int) -> StateTransition:
        state = self._state
        if epoch != state.epoch or state.phase is not TurnPhase.CUTOVER_PENDING:
            return StateTransition(state)
        return self._set(AgentState(epoch, TurnPhase.PLAYING), StateEffect.EMIT_AUDIO)

    def audio_finished(self, epoch: int) -> StateTransition:
        if epoch != self._state.epoch or self._state.phase is not TurnPhase.PLAYING:
            return StateTransition(self._state)
        return self._set(AgentState(epoch, TurnPhase.IDLE), StateEffect.FINISH_AUDIO)

    def failed(self, epoch: int, *, audio_started: bool) -> StateTransition:
        if epoch != self._state.epoch:
            return StateTransition(self._state)
        effects = (
            (StateEffect.FINISH_AUDIO, StateEffect.SHOW_FAILURE)
            if audio_started
            else (StateEffect.SHOW_FAILURE,)
        )
        return self._set(AgentState(epoch, TurnPhase.FAILED), *effects)

    def _set(self, state: AgentState, *effects: StateEffect) -> StateTransition:
        self._state = state
        return StateTransition(state, effects)
