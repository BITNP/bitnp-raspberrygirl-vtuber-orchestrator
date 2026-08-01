from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Protocol, cast

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value

_INSTRUCTION: Final = (
    "判断这条已完成的语音输入是否应开启有意义的对话轮次。"
    "若当前正在播放回答, 只有用户明确要求停止、纠正、提问或切换话题时才返回 interrupt;"
    "回声、附和、无意义片段和误识别返回 discard。"
    '仅返回 JSON 对象 {"decision":"accept"|"interrupt"|"discard"}。'
)


@dataclass(frozen=True, slots=True)
class AsrGateRequest:
    transcript: str

    active_answer_excerpt: str = ""

    is_playing: bool = False

    instruction: str = _INSTRUCTION


class AsrGateProvider(Protocol):
    def __call__(self, request: AsrGateRequest) -> str: ...


class AsyncAsrGateProvider(Protocol):
    async def __call__(self, request: AsrGateRequest) -> str: ...


@unique
class AsrGateDecision(StrEnum):
    ACCEPT = "accept"

    DISCARD = "discard"

    INTERRUPT = "interrupt"


@dataclass(frozen=True, slots=True)
class AsrSemanticGate:
    provider: AsrGateProvider

    def evaluate(
        self,
        transcript: str,
        *,
        active_answer_excerpt: str = "",
        is_playing: bool = False,
    ) -> AsrGateDecision:
        """Fail closed: an unavailable semantic gate must not start a turn."""
        value = self._response(
            transcript,
            active_answer_excerpt=active_answer_excerpt,
            is_playing=is_playing,
        )
        if not isinstance(value, dict):
            return AsrGateDecision.DISCARD
        parsed = cast("dict[str, object]", value)
        if set(parsed) != {"decision"}:
            return AsrGateDecision.DISCARD
        decision = parsed["decision"]
        match decision:
            case "accept":
                return AsrGateDecision.DISCARD if is_playing else AsrGateDecision.ACCEPT

            case "interrupt" if is_playing:
                return AsrGateDecision.INTERRUPT

            case "discard":
                return AsrGateDecision.DISCARD

            case _:
                return AsrGateDecision.DISCARD

    def _response(
        self, transcript: str, *, active_answer_excerpt: str, is_playing: bool
    ) -> object | None:
        if transcript.strip() == "":
            return None
        try:
            return parse_json_value(
                self.provider(
                    AsrGateRequest(
                        transcript=transcript,
                        active_answer_excerpt=active_answer_excerpt,
                        is_playing=is_playing,
                    )
                )
            )
        except (JsonBoundaryError, OSError, TimeoutError):
            return None


@dataclass(frozen=True, slots=True)
class AsyncAsrSemanticGate:
    """Async equivalent of :class:`AsrSemanticGate` for live LLM requests."""

    provider: AsyncAsrGateProvider

    async def evaluate(
        self,
        transcript: str,
        *,
        active_answer_excerpt: str = "",
        is_playing: bool = False,
    ) -> AsrGateDecision:
        if transcript.strip() == "":
            return AsrGateDecision.DISCARD
        try:
            value = parse_json_value(
                await self.provider(
                    AsrGateRequest(
                        transcript=transcript,
                        active_answer_excerpt=active_answer_excerpt,
                        is_playing=is_playing,
                    )
                )
            )
        except (JsonBoundaryError, OSError, TimeoutError):
            return AsrGateDecision.DISCARD
        if not isinstance(value, dict) or set(value) != {"decision"}:
            return AsrGateDecision.DISCARD
        decision = cast("dict[str, object]", value)["decision"]
        if decision == "accept" and not is_playing:
            return AsrGateDecision.ACCEPT
        if decision == "interrupt" and is_playing:
            return AsrGateDecision.INTERRUPT
        return AsrGateDecision.DISCARD
