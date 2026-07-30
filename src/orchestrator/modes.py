
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique


@unique
class AudienceSource(StrEnum):

    ASR = "asr"
    COMMENT = "comment"


@dataclass(frozen=True, slots=True)
class AudienceInput:

    source: AudienceSource
    text: str
    received_at_ms: int


@dataclass(frozen=True, slots=True)
class AnswerCandidate:

    input: AudienceInput


class AdaptiveAgentPolicy:

    def select_answer_candidate(
        self,
        audience_inputs: Sequence[AudienceInput],
    ) -> AnswerCandidate | None:
        audience_input = _oldest_input(audience_inputs)

        if audience_input is None:
            return None

        return AnswerCandidate(input=audience_input)


def _oldest_input(audience_inputs: Sequence[AudienceInput]) -> AudienceInput | None:
    if len(audience_inputs) == 0:
        return None

    return min(
        audience_inputs,
        key=lambda audience_input: audience_input.received_at_ms,
    )
