from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Protocol, cast

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value

_ECHO_FRAGMENT_LENGTH: Final = 4

_INSTRUCTION: Final = (
    "判断这条已完成的语音输入是否应开启有意义的对话轮次。"
    "只要输入可能是上一轮语音输出的回声、复述、片段或部分重合、"
    "无论当前是否仍在播放都必须返回 discard、绝不能返回 accept 或 interrupt。"
    "若当前正在播放回答且输入不与上一轮输出重合、"
    "只有用户明确要求停止、纠正、提问或切换话题时才返回 interrupt;"
    "附和、无意义片段和误识别也返回 discard。"
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
        if _may_echo_previous_answer(transcript, active_answer_excerpt):
            return AsrGateDecision.DISCARD
        value = self._response(
            transcript,
            active_answer_excerpt=active_answer_excerpt,
            is_playing=is_playing,
        )
        return _decision_from_response(value, is_playing=is_playing)

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
        if _may_echo_previous_answer(transcript, active_answer_excerpt):
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
        return _decision_from_response(value, is_playing=is_playing)


def _decision_from_response(value: object, *, is_playing: bool) -> AsrGateDecision:
    if not isinstance(value, dict):
        return AsrGateDecision.DISCARD
    parsed = cast("dict[str, object]", value)
    if set(parsed) != {"decision"}:
        return AsrGateDecision.DISCARD
    decision = parsed["decision"]
    if decision == "accept" and not is_playing:
        return AsrGateDecision.ACCEPT
    if decision == "interrupt" and is_playing:
        return AsrGateDecision.INTERRUPT
    return AsrGateDecision.DISCARD


def _may_echo_previous_answer(transcript: str, previous_answer: str) -> bool:
    """Conservatively reject text that could have come from the last TTS reply.

    ASR may add or omit punctuation and whitespace, so comparison operates on
    normalized text.  A whole echoed utterance is rejected at any length; a
    longer utterance is rejected when it contains a four-character (or longer)
    contiguous fragment from the preceding spoken reply.  This check is local
    and precedes the model gate so a permissive model response cannot reopen an
    echo-induced turn.
    """
    normalized_transcript = "".join(
        character.casefold() for character in transcript if character.isalnum()
    )
    normalized_answer = "".join(
        character.casefold() for character in previous_answer if character.isalnum()
    )
    if normalized_transcript == "" or normalized_answer == "":
        return False
    if normalized_transcript in normalized_answer:
        return True
    if len(normalized_transcript) < _ECHO_FRAGMENT_LENGTH:
        return False
    return any(
        normalized_transcript[index : index + _ECHO_FRAGMENT_LENGTH]
        in normalized_answer
        for index in range(len(normalized_transcript) - _ECHO_FRAGMENT_LENGTH + 1)
    )
