"""Small, fault-tolerant model response and inline-cue contracts.

The model is deliberately not asked to construct execution plans.  It may only
return a reply plus one allowlisted intent.  The reducer owns all effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value

_MAX_REPLY_CHARS: Final = 4_000
_MAX_CUES: Final = 8
_CUE_PATTERN: Final = re.compile(
    r'<(?P<kind>action|expression) name="(?P<name>[a-z][a-z0-9_]{0,31})"/>'
)
_CONTROL_LIKE_PATTERN: Final = re.compile(r"<(?:action|expression)\b[^>]*>")


@unique
class CueKind(StrEnum):
    ACTION = "action"
    EXPRESSION = "expression"


@dataclass(frozen=True, slots=True)
class ResponseProposal:
    reply: str
    intent: str
    used_text_fallback: bool = False


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


def parse_response_proposal(
    raw: str, *, allowed_intents: frozenset[str]
) -> ResponseProposal:
    """Parse the two-field proposal, degrading malformed output to plain text.

    A model formatting mistake must never require a repair call or turn a user
    answer into an unbounded control surface.
    """
    if len(raw) > _MAX_REPLY_CHARS:
        raw = raw[:_MAX_REPLY_CHARS]
    try:
        value = parse_json_value(raw)
    except JsonBoundaryError:
        return ResponseProposal(raw, "answer", used_text_fallback=True)
    if (
        not isinstance(value, dict)
        or set(value) != {"reply", "intent"}
        or not isinstance(value.get("reply"), str)
        or not isinstance(value.get("intent"), str)
        or value["intent"] not in allowed_intents
        or len(value["reply"]) > _MAX_REPLY_CHARS
    ):
        return ResponseProposal(raw, "answer", used_text_fallback=True)
    return ResponseProposal(value["reply"], value["intent"])


def parse_inline_cues(
    reply: str,
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
    for match in _CUE_PATTERN.finditer(reply):
        prefix = _CONTROL_LIKE_PATTERN.sub("", reply[cursor : match.start()])
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
    suffix = _CONTROL_LIKE_PATTERN.sub("", reply[cursor:])
    marked_parts.append(suffix)
    spoken_parts.append(suffix)
    # Malformed control-like tags were removed from each non-control text span.
    spoken_text = "".join(spoken_parts)
    marked_text = "".join(marked_parts)
    return CueParseResult(marked_text, spoken_text, tuple(cues), rejected)
