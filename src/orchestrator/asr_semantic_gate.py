"""Closed structured semantic admission for finalized ASR input."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Protocol

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value

_INSTRUCTION: Final = (
    "判断这条已完成的语音输入是否应开启有意义的对话轮次。"
    '仅返回 JSON 对象 {"decision":"accept"} 或 {"decision":"discard"}。'
)


@dataclass(frozen=True, slots=True)
class AsrGateRequest:
    """One finalized ASR transcript evaluated at the interactive boundary."""

    transcript: str
    instruction: str = _INSTRUCTION


class AsrGateProvider(Protocol):
    """Produces one untrusted structured gate response."""

    def __call__(self, request: AsrGateRequest) -> str:
        """Return the raw model response before boundary parsing."""
        ...


@unique
class AsrGateDecision(StrEnum):
    """Closed outcomes available to the scheduler after semantic evaluation."""

    ACCEPT = "accept"
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class AsrSemanticGate:
    """Fail closed when the gate response is unavailable or malformed."""

    provider: AsrGateProvider

    def evaluate(self, transcript: str) -> AsrGateDecision:
        """Return a typed decision without exposing raw model output downstream."""
        try:
            value = parse_json_value(self.provider(AsrGateRequest(transcript)))
        except (JsonBoundaryError, OSError, TimeoutError):
            return AsrGateDecision.DISCARD
        if not isinstance(value, dict) or set(value) != {"decision"}:
            return AsrGateDecision.DISCARD
        decision = value["decision"]
        match decision:
            case "accept":
                return AsrGateDecision.ACCEPT
            case "discard":
                return AsrGateDecision.DISCARD
            case _:
                return AsrGateDecision.DISCARD
