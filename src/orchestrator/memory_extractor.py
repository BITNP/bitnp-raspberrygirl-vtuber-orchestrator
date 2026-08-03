"""Low-priority, bounded extraction of memory candidates from accepted turns."""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.memory import MemoryConfidence, MemoryKey

_MAX_KEY_CHARS = 128
_MAX_VALUE_CHARS = 512
_MAX_CONFIDENCE = 100

@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    key: MemoryKey
    value: str
    confidence: MemoryConfidence


def parse_memory_candidate(raw: str) -> MemoryCandidate | None:
    """Accept one ordinary preference candidate; malformed output is discarded."""
    try:
        value = parse_json_value(raw)
    except JsonBoundaryError:
        return None
    if not isinstance(value, dict) or set(value) != {"key", "value", "confidence"}:
        return None
    key = value.get("key")
    candidate_value = value.get("value")
    confidence = value.get("confidence")
    if (
        not isinstance(key, str)
        or not key.strip()
        or len(key) > _MAX_KEY_CHARS
        or not isinstance(candidate_value, str)
        or not candidate_value.strip()
        or len(candidate_value) > _MAX_VALUE_CHARS
        or type(confidence) is not int
        or not 0 <= confidence <= _MAX_CONFIDENCE
    ):
        return None
    return MemoryCandidate(
        MemoryKey(key), candidate_value, MemoryConfidence(confidence)
    )
