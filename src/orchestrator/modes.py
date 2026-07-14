"""Orchestrator-owned mode policies and audience-input selection."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType, override

ScriptStep = NewType("ScriptStep", int)
SlideStep = NewType("SlideStep", int)


@unique
class OrchestratorMode(StrEnum):
    """Closed set of Orchestrator modes."""

    LECTURER = "lecturer"
    VIRTUAL_STREAMER = "virtual_streamer"
    ONSITE_EXPLAINER = "onsite_explainer"


@unique
class AudienceSource(StrEnum):
    """Audience input source normalized by Orchestrator policy."""

    ASR = "asr"
    COMMENT = "comment"


@dataclass(frozen=True, slots=True)
class UnknownModeError(Exception):
    """Raised when an external mode value is not supported."""

    raw_mode: str

    @override
    def __str__(self) -> str:
        return f"unknown orchestrator mode: {self.raw_mode}"


@dataclass(frozen=True, slots=True)
class AudienceInput:
    """Normalized audience input considered by a mode policy."""

    source: AudienceSource
    text: str
    received_at_ms: int


@dataclass(frozen=True, slots=True)
class QaWindow:
    """Inclusive scheduled lecturer Q&A interval."""

    start_ms: int
    end_ms: int

    def contains(self, received_at_ms: int) -> bool:
        """Return whether an input timestamp falls inside this Q&A window."""
        return self.start_ms <= received_at_ms <= self.end_ms


@dataclass(frozen=True, slots=True)
class LecturerState:
    """Lecturer script, slide, interruption, and Q&A state."""

    script_step: ScriptStep
    slide_step: SlideStep
    immediate_interruption_enabled: bool
    qa_window: QaWindow | None


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    """Audience input selected for the next Orchestrator answer turn."""

    mode: OrchestratorMode
    input: AudienceInput
    reason: str
    script_step: ScriptStep | None = None
    slide_step: SlideStep | None = None
    topic: str | None = None


@dataclass(frozen=True, slots=True)
class LecturerModePolicy:
    """Lecturer policy for scripted slides and controlled interruption."""

    state: LecturerState

    def select_answer_candidate(
        self,
        audience_inputs: Sequence[AudienceInput],
    ) -> AnswerCandidate | None:
        """Select an audience question if interruption or Q&A policy allows it."""
        audience_input = _oldest_input(audience_inputs)
        if audience_input is None:
            return None
        if self.state.immediate_interruption_enabled:
            return self._candidate(
                audience_input,
                reason="lecturer_immediate_interruption",
            )
        if self._qa_window_contains(audience_input.received_at_ms):
            return self._candidate(audience_input, reason="lecturer_scheduled_qa")
        return None

    def _candidate(
        self,
        audience_input: AudienceInput,
        *,
        reason: str,
    ) -> AnswerCandidate:
        return AnswerCandidate(
            mode=OrchestratorMode.LECTURER,
            input=audience_input,
            reason=reason,
            script_step=self.state.script_step,
            slide_step=self.state.slide_step,
        )

    def _qa_window_contains(self, received_at_ms: int) -> bool:
        qa_window = self.state.qa_window
        if qa_window is None:
            return False
        return qa_window.contains(received_at_ms)


@dataclass(frozen=True, slots=True)
class VirtualStreamerModePolicy:
    """Virtual streamer policy that prefers live comment input."""

    topic: str

    def select_answer_candidate(
        self,
        audience_inputs: Sequence[AudienceInput],
    ) -> AnswerCandidate | None:
        """Select comments before other audience input sources."""
        audience_input = _oldest_source(audience_inputs, AudienceSource.COMMENT)
        if audience_input is None:
            audience_input = _oldest_input(audience_inputs)
        if audience_input is None:
            return None
        return AnswerCandidate(
            mode=OrchestratorMode.VIRTUAL_STREAMER,
            input=audience_input,
            reason="virtual_streamer_comment_priority",
            topic=self.topic,
        )


@dataclass(frozen=True, slots=True)
class OnsiteExplainerModePolicy:
    """Onsite explainer policy that prefers ASR voice input."""

    def select_answer_candidate(
        self,
        audience_inputs: Sequence[AudienceInput],
    ) -> AnswerCandidate | None:
        """Select ASR input before other audience input sources."""
        audience_input = _oldest_source(audience_inputs, AudienceSource.ASR)
        if audience_input is None:
            audience_input = _oldest_input(audience_inputs)
        if audience_input is None:
            return None
        return AnswerCandidate(
            mode=OrchestratorMode.ONSITE_EXPLAINER,
            input=audience_input,
            reason="onsite_explainer_asr_priority",
        )


class ModePolicy:
    """Factory namespace for concrete Orchestrator mode policies."""

    @staticmethod
    def lecturer(state: LecturerState) -> LecturerModePolicy:
        """Build lecturer mode policy from current lecture state."""
        return LecturerModePolicy(state=state)

    @staticmethod
    def virtual_streamer(*, topic: str) -> VirtualStreamerModePolicy:
        """Build virtual streamer policy with topic context."""
        return VirtualStreamerModePolicy(topic=topic)

    @staticmethod
    def onsite_explainer() -> OnsiteExplainerModePolicy:
        """Build onsite explainer policy."""
        return OnsiteExplainerModePolicy()


def parse_orchestrator_mode(raw_mode: str) -> OrchestratorMode:
    """Parse an external mode string into the closed Orchestrator mode enum."""
    try:
        mode = OrchestratorMode(raw_mode)
    except ValueError as error:
        raise UnknownModeError(raw_mode=raw_mode) from error
    return mode


def _oldest_input(audience_inputs: Sequence[AudienceInput]) -> AudienceInput | None:
    if len(audience_inputs) == 0:
        return None
    return min(
        audience_inputs,
        key=lambda audience_input: audience_input.received_at_ms,
    )


def _oldest_source(
    audience_inputs: Sequence[AudienceInput],
    source: AudienceSource,
) -> AudienceInput | None:
    matching_inputs = tuple(
        audience_input
        for audience_input in audience_inputs
        if audience_input.source is source
    )
    return _oldest_input(matching_inputs)
