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
    QUEUED = "queued"
    REASONING = "reasoning"
    WAITING_TOOL = "waiting_tool"
    SYNTHESIZING = "synthesizing"
    CUTOVER_PENDING = "cutover_pending"
    PLAYING = "playing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
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


@dataclass(frozen=True, slots=True)
class AgentState:
    epoch: int = 0
    phase: TurnPhase = TurnPhase.IDLE
    pending_interrupt: bool = False
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class StateTransition:
    state: AgentState
    effects: tuple[StateEffect, ...] = ()


@final
class TurnCoordinator:
    """Own the turn's transport-free reasoning, cutover, and playback states.

    Provider adapters return observations to this coordinator; they do not
    choose media effects themselves.  The legacy name remains an alias below
    while SessionRuntime's response path is migrated onto this same state
    machine.
    """

    def __init__(self) -> None:
        self._state = AgentState()

    @property
    def state(self) -> AgentState:
        return self._state

    def enqueue(
        self, *, turn_id: str, epoch: int, replacement: bool = False
    ) -> StateTransition:
        """Admit one scheduler-owned turn before any provider work begins.

        ``SessionScheduler`` remains the revision authority and passes its
        already-admitted id here.  In particular this method never creates a
        turn or advances an epoch itself.  A replacement may begin while the
        previous lease is still playing; the media fence retains that lease,
        while this coordinator tracks the new turn's work.
        """
        if epoch < self._state.epoch:
            return StateTransition(self._state)
        return self._set(
            AgentState(
                epoch=epoch,
                phase=TurnPhase.QUEUED,
                pending_interrupt=replacement,
                turn_id=turn_id,
            )
        )

    def start_reasoning(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id, epoch, {TurnPhase.QUEUED}, TurnPhase.REASONING,
            StateEffect.START_REASONING,
        )

    def wait_for_tool(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id, epoch, {TurnPhase.REASONING}, TurnPhase.WAITING_TOOL,
        )

    def resume_reasoning(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id, epoch, {TurnPhase.WAITING_TOOL}, TurnPhase.REASONING,
        )

    def start_synthesizing(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id, epoch, {TurnPhase.REASONING}, TurnPhase.SYNTHESIZING,
            StateEffect.START_TTS,
        )

    def await_cutover(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id, epoch, {TurnPhase.SYNTHESIZING}, TurnPhase.CUTOVER_PENDING,
            StateEffect.FLUSH_SOUND,
        )

    def playback_started(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id,
            epoch,
            {TurnPhase.SYNTHESIZING, TurnPhase.CUTOVER_PENDING},
            TurnPhase.PLAYING,
            StateEffect.EMIT_AUDIO,
        )

    def playback_finished(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id, epoch, {TurnPhase.PLAYING}, TurnPhase.COMPLETED,
            StateEffect.FINISH_AUDIO,
        )

    def cancel(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id,
            epoch,
            {
                TurnPhase.QUEUED,
                TurnPhase.REASONING,
                TurnPhase.WAITING_TOOL,
                TurnPhase.SYNTHESIZING,
                TurnPhase.CUTOVER_PENDING,
            },
            TurnPhase.CANCELLED,
            StateEffect.CANCEL_DELIBERATIVE,
        )

    def fail(self, *, turn_id: str, epoch: int) -> StateTransition:
        return self._advance(
            turn_id,
            epoch,
            {
                TurnPhase.QUEUED,
                TurnPhase.REASONING,
                TurnPhase.WAITING_TOOL,
                TurnPhase.SYNTHESIZING,
                TurnPhase.CUTOVER_PENDING,
            },
            TurnPhase.FAILED,
        )

    def gate(self, outcome: GateOutcome) -> StateTransition:
        state = self._state
        if outcome is GateOutcome.DISCARD:
            return StateTransition(state)
        if outcome is GateOutcome.ACCEPT and state.phase in {
            TurnPhase.IDLE,
            TurnPhase.FAILED,
        }:
            return self._set(
                AgentState(state.epoch, TurnPhase.REASONING, turn_id=state.turn_id),
                StateEffect.START_REASONING,
            )
        if outcome is GateOutcome.ACCEPT and state.phase in {
            TurnPhase.REASONING,
            TurnPhase.SYNTHESIZING,
        }:
            return self._set(
                AgentState(state.epoch + 1, TurnPhase.REASONING, turn_id=state.turn_id),
                StateEffect.CANCEL_DELIBERATIVE,
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
            return self._set(AgentState(epoch, TurnPhase.FAILED))
        return self._set(
            AgentState(
                epoch,
                TurnPhase.SYNTHESIZING,
                self._state.pending_interrupt,
                self._state.turn_id,
            ),
            StateEffect.START_TTS,
        )

    def audio_ready(self, epoch: int) -> StateTransition:
        state = self._state
        if epoch != state.epoch or state.phase is not TurnPhase.SYNTHESIZING:
            return StateTransition(state)
        if state.pending_interrupt:
            return self._set(
                AgentState(
                    epoch,
                    TurnPhase.CUTOVER_PENDING,
                    pending_interrupt=True,
                    turn_id=state.turn_id,
                ),
                StateEffect.FLUSH_SOUND,
            )
        return self._set(
            AgentState(epoch, TurnPhase.PLAYING, turn_id=state.turn_id),
            StateEffect.EMIT_AUDIO,
        )

    def flush_acknowledged(self, epoch: int) -> StateTransition:
        state = self._state
        if epoch != state.epoch or state.phase is not TurnPhase.CUTOVER_PENDING:
            return StateTransition(state)
        return self._set(
            AgentState(epoch, TurnPhase.PLAYING, turn_id=state.turn_id),
            StateEffect.EMIT_AUDIO,
        )

    def audio_finished(self, epoch: int) -> StateTransition:
        if epoch != self._state.epoch or self._state.phase is not TurnPhase.PLAYING:
            return StateTransition(self._state)
        return self._set(
            AgentState(epoch, TurnPhase.COMPLETED, turn_id=self._state.turn_id),
            StateEffect.FINISH_AUDIO,
        )

    def failed(self, epoch: int, *, audio_started: bool) -> StateTransition:
        if epoch != self._state.epoch:
            return StateTransition(self._state)
        # Provider failure is terminal for its turn.  The runtime records the
        # structured error and emits no recovery speech or frontend status.
        _ = audio_started
        return self._set(
            AgentState(epoch, TurnPhase.FAILED, turn_id=self._state.turn_id)
        )

    def _advance(
        self,
        turn_id: str,
        epoch: int,
        allowed: set[TurnPhase],
        phase: TurnPhase,
        *effects: StateEffect,
    ) -> StateTransition:
        state = self._state
        if (
            state.turn_id != turn_id
            or state.epoch != epoch
            or state.phase not in allowed
        ):
            return StateTransition(state)
        return self._set(
            AgentState(epoch, phase, state.pending_interrupt, turn_id), *effects
        )

    def _set(self, state: AgentState, *effects: StateEffect) -> StateTransition:
        self._state = state
        return StateTransition(state, effects)


# Transitional import surface for existing callers.  New code must depend on
# TurnCoordinator so the coordinator is visibly the turn authority.
AgentStateReducer = TurnCoordinator
