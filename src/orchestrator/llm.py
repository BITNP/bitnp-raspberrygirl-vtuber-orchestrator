
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, Self, TypedDict, override

from orchestrator.media_adapters import OpenAICompatibleASRAdapter, VllmOmniTTSAdapter
from orchestrator.modes import AnswerCandidate
from orchestrator.prompt_composition import PromptSnapshot, compose_prompt
from orchestrator.provider_streaming import ProviderCancellationHandle
from orchestrator.retrieval import KnowledgeRef, RetrievalResult, RetrievalSnapshot
from orchestrator.state_snapshots import (
    CorpusRevision,
    IndexRevision,
    TaskStateSnapshot,
)

__all__ = ["OpenAICompatibleASRAdapter", "VllmOmniTTSAdapter"]


DEFAULT_TEMPERATURE: Final = 0.2

DEFAULT_TIMEOUT_SECONDS: Final = 30.0


class OpenAIMessagePayload(TypedDict):

    role: Literal["system", "user"]

    content: str


class OpenAIChatPayload(TypedDict):

    model: str

    messages: list[OpenAIMessagePayload]

    stream: bool

    temperature: float

    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class AdapterConfigError(ValueError):

    field_name: str

    @override
    def __str__(self) -> str:
        return f"LLM adapter config field is blank: {self.field_name}"


@dataclass(frozen=True, slots=True)
class LLMTimeoutError(Exception):

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class LLMPrompt:

    system: str

    user: str


@dataclass(frozen=True, slots=True)
class LLMRequest:

    prompt: LLMPrompt

    temperature: float = DEFAULT_TEMPERATURE

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class LLMChunk:

    index: int

    text: str


@dataclass(frozen=True, slots=True)
class LLMFinal:

    text: str

    used_fallback: bool


@dataclass(frozen=True, slots=True)
class LLMError:

    code: str

    message: str

    cancel_pending_media: bool


type LLMStreamEvent = LLMChunk | LLMFinal | LLMError


class CancellationToken(ProviderCancellationHandle):
    ...


class LLMAdapter(Protocol):

    @property
    def capability(self) -> Literal["streaming", "final_only"]:
        ...

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        ...


@dataclass(frozen=True, slots=True)
class MockLLMAdapter:

    answer_chunks: tuple[str, ...]

    capability: Literal["streaming"] = "streaming"

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        _ = request

        emitted_chunks: list[str] = []

        for index, chunk in enumerate(self.answer_chunks):
            if _is_cancelled(cancellation):
                return

            emitted_chunks.append(chunk)

            yield LLMChunk(index=index, text=chunk)

        if _is_cancelled(cancellation):
            return

        yield LLMFinal(text="".join(emitted_chunks), used_fallback=False)


@dataclass(frozen=True, slots=True)
class TimeoutLLMAdapter:

    timeout_reason: str

    capability: Literal["final_only"] = "final_only"

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        _ = request

        _ = cancellation

        return _TimeoutStream(reason=self.timeout_reason)


@dataclass(frozen=True, slots=True)
class _TimeoutStream:

    reason: str

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> LLMStreamEvent:
        raise LLMTimeoutError(reason=self.reason)


@dataclass(frozen=True, slots=True)
class FallbackLLMAdapter:

    primary: LLMAdapter

    fallback_text: str

    @property
    def capability(self) -> Literal["streaming", "final_only"]:
        return self.primary.capability

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        try:
            yield from self.primary.stream(request, cancellation=cancellation)

        except LLMTimeoutError as error:
            yield LLMError(
                code="llm_timeout",
                message=str(error),
                cancel_pending_media=True,
            )

            if not _is_cancelled(cancellation):
                yield LLMFinal(text=self.fallback_text, used_fallback=True)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleAdapter:

    model: str

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    temperature: float = DEFAULT_TEMPERATURE

    capability: Literal["streaming"] = "streaming"

    def build_payload(self, request: LLMRequest) -> OpenAIChatPayload:
        model = self.model.strip()

        if model == "":
            raise AdapterConfigError(field_name="model")

        return {
            "model": model,
            "messages": [
                {"role": "system", "content": request.prompt.system},
                {"role": "user", "content": request.prompt.user},
            ],
            "stream": True,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
        }


def build_llm_request(
    candidate: AnswerCandidate,
    *,
    retrieval: RetrievalResult | None = None,
    prompt_snapshot: PromptSnapshot | None = None,
    context_refs: Sequence[KnowledgeRef] | None = None,
) -> LLMRequest:
    if retrieval is None:
        retrieval = RetrievalResult(
            snapshot=RetrievalSnapshot(
                "fixture-corpus",
                CorpusRevision(1),
                "fixture-index",
                IndexRevision(1),
            ),
            refs=tuple(context_refs or ()),
        )

    snapshot = prompt_snapshot or PromptSnapshot(
        task_state=TaskStateSnapshot.initial(),
        context_entries=(),
        max_context_chars=4_000,
    )

    fields = compose_prompt(candidate, retrieval, snapshot)

    return LLMRequest(prompt=LLMPrompt(system=fields.system, user=fields.user))


def _is_cancelled(cancellation: CancellationToken | None) -> bool:
    if cancellation is None:
        return False

    return cancellation.cancelled
