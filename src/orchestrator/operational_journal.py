
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Final

_REDACTED_ID_LENGTH: Final = 16


@dataclass(frozen=True, slots=True)
class OperationalRecord:

    stage: str

    trace_id: str

    session_id: str

    turn_id: str | None

    segment_id: str | None

    task_id: str | None

    outcome: str


@dataclass(frozen=True, slots=True)
class RedactedOperationalRecord:

    stage: str

    trace_id: str

    session_id: str

    turn_id: str | None

    segment_id: str | None

    task_id: str | None

    outcome: str


class OperationalJournal:

    def __init__(self) -> None:
        self._records: list[RedactedOperationalRecord] = []

    @property
    def records(self) -> tuple[RedactedOperationalRecord, ...]:
        return tuple(self._records)

    def append(self, record: OperationalRecord) -> None:
        self._records.append(
            RedactedOperationalRecord(
                stage=record.stage,
                trace_id=_redact(record.trace_id),
                session_id=_redact(record.session_id),
                turn_id=_redact_optional(record.turn_id),
                segment_id=_redact_optional(record.segment_id),
                task_id=_redact_optional(record.task_id),
                outcome=record.outcome,
            )
        )


def _redact(value: str) -> str:
    return sha256(value.encode()).hexdigest()[:_REDACTED_ID_LENGTH]


def _redact_optional(value: str | None) -> str | None:
    if value is None:
        return None

    return _redact(value)
