"""Strict, non-planning Brain proposals and inline speech cues."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, cast

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value

_MAX_SPEECH_CHARS: Final = 4_000
_MAX_INTENT_CHARS: Final = 128
_MAX_CUES: Final = 8
_CUE_PATTERN: Final = re.compile(
    r'<(?P<kind>action|expression) name="(?P<name>[a-z][a-z0-9_]{0,31})"/>'
)
_CONTROL_LIKE_PATTERN: Final = re.compile(r"<(?:action|expression)\b[^>]*>")


@unique
class BrainDecision(StrEnum):
    ACCEPT = "accept"
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class OperationProposal:
    """Untrusted model-proposed operation and its isolated arguments."""

    intent: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ResponseProposal:
    decision: BrainDecision
    speech: str
    operation: OperationProposal | None

    @property
    def reply(self) -> str:
        """Compatibility alias for transport-free legacy helpers."""
        return self.speech

    @property
    def intent(self) -> str:
        return "answer" if self.operation is None else self.operation.intent

    @property
    def used_text_fallback(self) -> bool:
        return False


@unique
class CueKind(StrEnum):
    ACTION = "action"
    EXPRESSION = "expression"


@dataclass(frozen=True, slots=True)
class InlineCue:
    kind: CueKind
    name: str
    text_offset: int


@dataclass(frozen=True, slots=True)
class CueParseResult:
    marked_text: str
    spoken_text: str
    cues: tuple[InlineCue, ...]
    rejected_cues: int


def parse_response_proposal(raw: str) -> ResponseProposal | None:  # noqa: PLR0911
    """Parse one strict proposal; malformed output has no fallback effect."""
    if len(raw) > _MAX_SPEECH_CHARS * 2:
        return None
    try:
        value = parse_json_value(raw)
    except JsonBoundaryError:
        return None
    if not isinstance(value, dict) or set(value) != {"decision", "speech", "operation"}:
        return None
    parsed = cast("dict[str, object]", value)
    decision = parsed["decision"]
    speech = parsed["speech"]
    operation_value = parsed["operation"]
    if decision not in {BrainDecision.ACCEPT, BrainDecision.DISCARD}:
        return None
    if not isinstance(speech, str) or len(speech) > _MAX_SPEECH_CHARS:
        return None
    if decision == BrainDecision.DISCARD:
        if speech != "" or operation_value is not None:
            return None
        return ResponseProposal(BrainDecision.DISCARD, "", None)
    if not speech.strip():
        return None
    operation = _parse_operation(operation_value)
    if operation_value is not None and operation is None:
        return None
    return ResponseProposal(BrainDecision.ACCEPT, speech, operation)


def _parse_operation(value: object) -> OperationProposal | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    parsed = cast("dict[str, object]", cast("object", value))
    if set(parsed) != {"intent", "arguments"}:
        return None
    intent = parsed["intent"]
    arguments = parsed["arguments"]
    if (
        not isinstance(intent, str)
        or not intent.strip()
        or len(intent) > _MAX_INTENT_CHARS
        or not isinstance(arguments, dict)
    ):
        return None
    parsed_arguments = cast("dict[str, object]", cast("object", arguments))
    return OperationProposal(intent, parsed_arguments)


def parse_final_speech_proposal(raw: str) -> ResponseProposal | None:
    """The observation follow-up can only accept non-empty speech."""
    proposal = parse_response_proposal(raw)
    if (
        proposal is None
        or proposal.decision is not BrainDecision.ACCEPT
        or proposal.operation is not None
    ):
        return None
    return proposal


def parse_inline_cues(
    speech: str,
    *,
    allowed_actions: frozenset[str],
    allowed_expressions: frozenset[str],
) -> CueParseResult:
    """Strip control tags for TTS while retaining only allowlisted timeline cues."""
    marked_parts: list[str] = []
    spoken_parts: list[str] = []
    cues: list[InlineCue] = []
    rejected = 0
    cursor = 0
    text_offset = 0
    for match in _CUE_PATTERN.finditer(speech):
        prefix = _CONTROL_LIKE_PATTERN.sub("", speech[cursor : match.start()])
        marked_parts.append(prefix)
        spoken_parts.append(prefix)
        text_offset += len(prefix)
        kind = CueKind(match.group("kind"))
        name = match.group("name")
        allowed = (
            name in allowed_actions
            if kind is CueKind.ACTION
            else name in allowed_expressions
        )
        if allowed and len(cues) < _MAX_CUES:
            marked_parts.append(match.group(0))
            cues.append(InlineCue(kind, name, text_offset))
        else:
            rejected += 1
        cursor = match.end()
    suffix_source = speech[cursor:]
    suffix = _CONTROL_LIKE_PATTERN.sub("", suffix_source)
    if suffix != suffix_source:
        rejected += 1
    marked_parts.append(suffix)
    spoken_parts.append(suffix)
    return CueParseResult(
        "".join(marked_parts), "".join(spoken_parts), tuple(cues), rejected
    )
