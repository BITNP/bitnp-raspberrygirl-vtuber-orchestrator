"""Deterministic acceptance evidence for effect-free response shadow replays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from orchestrator.task_registry import TaskState

if TYPE_CHECKING:
    from orchestrator.operational_journal import RedactedOperationalRecord
    from orchestrator.task_registry import TaskRecord

_TERMINAL_TASK_STATES = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.TIMED_OUT,
        TaskState.CANCELLED,
        TaskState.SUPERSEDED,
    }
)

_SHADOW_FORBIDDEN_STAGES = frozenset(
    {
        "response_compiled",
        "caption_timeline",
        "media_emitted",
        "memory_candidate",
        "context_compaction",
    }
)


@dataclass(frozen=True, slots=True)
class ShadowReplayReport:
    """Privacy-safe evidence used to decide whether a replay may graduate.

    This report deliberately consumes only redacted operational records and
    lifecycle metadata.  It never serializes an audience input, model reply,
    tool observation, or memory value.
    """

    shadow_turns: int
    text_fallbacks: int
    selected_intents: frozenset[str]
    violations: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.shadow_turns > 0 and not self.violations


@dataclass(frozen=True, slots=True)
class ShadowReplayEvidence:
    records: tuple[RedactedOperationalRecord, ...]
    task_records: tuple[TaskRecord, ...]
    context_revision_before: int
    context_revision_after: int
    memory_revision_before: int
    memory_revision_after: int


def audit_shadow_replay(evidence: ShadowReplayEvidence) -> ShadowReplayReport:
    """Evaluate the no-side-effect invariants for one shadow replay session."""
    shadow_records = tuple(
        record for record in evidence.records if record.stage == "response_shadow"
    )
    intents: set[str] = set()
    fallbacks = 0
    violations: list[str] = []
    for record in shadow_records:
        fields = _parse_outcome(record.outcome)
        intent = fields.get("intent")
        if intent is None:
            violations.append("shadow_missing_intent")
            continue
        intents.add(intent)
        if fields.get("fallback") == "True":
            fallbacks += 1
        if fields.get("phase") != "completed":
            violations.append("shadow_turn_not_completed")

    if evidence.context_revision_before != evidence.context_revision_after:
        violations.append("shadow_context_mutated")
    if evidence.memory_revision_before != evidence.memory_revision_after:
        violations.append("shadow_memory_mutated")
    if any(
        record.stage in _SHADOW_FORBIDDEN_STAGES for record in evidence.records
    ):
        violations.append("shadow_effect_stage_emitted")
    if any(
        record.state not in _TERMINAL_TASK_STATES for record in evidence.task_records
    ):
        violations.append("shadow_task_not_terminal")

    return ShadowReplayReport(
        shadow_turns=len(shadow_records),
        text_fallbacks=fallbacks,
        selected_intents=frozenset(intents),
        violations=tuple(violations),
    )


def _parse_outcome(outcome: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in outcome.split(";"):
        key, separator, value = item.partition("=")
        if separator and key:
            fields[key] = value
    return fields
