"""模块契约说明.

职责: 提供 orchestrator.llm
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 定义 OpenAIMessagePayload
    的状态、行为和对外协作边界。
    契约: 字段: role、content。
    """

    role: Literal["system", "user"]

    content: str


class OpenAIChatPayload(TypedDict):
    """类契约说明.

    职责: 定义 OpenAIChatPayload
    的状态、行为和对外协作边界。
    契约: 字段: model、messages、stream、temper
    ature、timeout_seconds。
    """

    model: str

    messages: list[OpenAIMessagePayload]

    stream: bool

    temperature: float

    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class AdapterConfigError(ValueError):
    """类契约说明.

    职责: 保存 AdapterConfigError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: field_name。 方法: __str__。
    """

    field_name: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"LLM adapter config field is blank: {self.field_name}"


@dataclass(frozen=True, slots=True)
class LLMTimeoutError(Exception):
    """类契约说明.

    职责: 保存 LLMTimeoutError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。 方法: __str__。
    """

    reason: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return self.reason


@dataclass(frozen=True, slots=True)
class LLMPrompt:
    """类契约说明.

    职责: 保存 LLMPrompt
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: system、user。
    """

    system: str

    user: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """类契约说明.

    职责: 保存 LLMRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    prompt、temperature、timeout_seconds。
    """

    prompt: LLMPrompt

    temperature: float = DEFAULT_TEMPERATURE

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class LLMChunk:
    """类契约说明.

    职责: 保存 LLMChunk 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: index、text。
    """

    index: int

    text: str


@dataclass(frozen=True, slots=True)
class LLMFinal:
    """类契约说明.

    职责: 保存 LLMFinal 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: text、used_fallback。
    """

    text: str

    used_fallback: bool


@dataclass(frozen=True, slots=True)
class LLMError:
    """类契约说明.

    职责: 保存 LLMError 不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    code、message、cancel_pending_media。
    """

    code: str

    message: str

    cancel_pending_media: bool


type LLMStreamEvent = LLMChunk | LLMFinal | LLMError


class CancellationToken(ProviderCancellationHandle):
    """类契约说明.

    职责: 定义 CancellationToken
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """


class LLMAdapter(Protocol):
    """类契约说明.

    职责: 声明 LLMAdapter 协议接口,约束实现方必须提供的行为。
    契约: 方法: capability、stream。
    """

    @property
    def capability(self) -> Literal["streaming", "final_only"]:
        """函数契约说明.

        功能: 执行 capability 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `Literal['streaming',
        'final_only']`。
        """
        ...

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        """函数契约说明.

        功能: 执行 stream 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 request:
        LLMRequest。 必填。 cancellation:
        CancellationToken | None。 可省略。
        契约: 同步调用。 返回
        `Iterator[LLMStreamEvent]`。
        """
        ...


@dataclass(frozen=True, slots=True)
class MockLLMAdapter:
    """类契约说明.

    职责: 保存 MockLLMAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: answer_chunks、capability。
    方法: stream。
    """

    answer_chunks: tuple[str, ...]

    capability: Literal["streaming"] = "streaming"

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        """函数契约说明.

        功能: 执行 stream 的同步逻辑,并协调
        enumerate, _is_cancelled,
        append, LLMFinal。
        参数: self 表示当前实例。 request:
        LLMRequest。 必填。 cancellation:
        CancellationToken | None。 可省略。
        契约: 同步调用。 返回迭代或生成器协议。 返回
        `Iterator[LLMStreamEvent]`。
        """
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
    """类契约说明.

    职责: 保存 TimeoutLLMAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: timeout_reason、capability。
    方法: stream。
    """

    timeout_reason: str

    capability: Literal["final_only"] = "final_only"

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        """函数契约说明.

        功能: 执行 stream 的同步逻辑,并协调
        _TimeoutStream。
        参数: self 表示当前实例。 request:
        LLMRequest。 必填。 cancellation:
        CancellationToken | None。 可省略。
        契约: 同步调用。 返回
        `Iterator[LLMStreamEvent]`。
        """
        _ = request

        _ = cancellation

        return _TimeoutStream(reason=self.timeout_reason)


@dataclass(frozen=True, slots=True)
class _TimeoutStream:
    """类契约说明.

    职责: 保存 _TimeoutStream
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。 方法:
    __iter__、__next__。
    """

    reason: str

    def __iter__(self) -> Self:
        """函数契约说明.

        功能: 提供对象内容的迭代协议入口。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `Self`。
        """
        return self

    def __next__(self) -> LLMStreamEvent:
        """函数契约说明.

        功能: 执行 __next__ 的同步逻辑,并协调
        LLMTimeoutError。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `LLMStreamEvent`。
        可能抛出 LLMTimeoutError。
        """
        raise LLMTimeoutError(reason=self.reason)


@dataclass(frozen=True, slots=True)
class FallbackLLMAdapter:
    """类契约说明.

    职责: 保存 FallbackLLMAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: primary、fallback_text。 方法:
    capability、stream。
    """

    primary: LLMAdapter

    fallback_text: str

    @property
    def capability(self) -> Literal["streaming", "final_only"]:
        """函数契约说明.

        功能: 执行 capability 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `Literal['streaming',
        'final_only']`。
        """
        return self.primary.capability

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        """函数契约说明.

        功能: 执行 stream 的同步逻辑,并协调 stream,
        LLMError, _is_cancelled,
        LLMFinal。
        参数: self 表示当前实例。 request:
        LLMRequest。 必填。 cancellation:
        CancellationToken | None。 可省略。
        契约: 同步调用。 返回迭代或生成器协议。 返回
        `Iterator[LLMStreamEvent]`。 可能抛出
        LLMError。
        """
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
    """类契约说明.

    职责: 保存 OpenAICompatibleAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: model、timeout_seconds、temper
    ature、capability。 方法: build_payload。
    """

    model: str

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    temperature: float = DEFAULT_TEMPERATURE

    capability: Literal["streaming"] = "streaming"

    def build_payload(self, request: LLMRequest) -> OpenAIChatPayload:
        """函数契约说明.

        功能: 构造协议对象、配置或测试夹具。
        参数: self 表示当前实例。 request:
        LLMRequest。 必填。
        契约: 同步调用。 返回
        `OpenAIChatPayload`。 可能抛出
        AdapterConfigError。
        """
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
    """函数契约说明.

    功能: 构造协议对象、配置或测试夹具。
    参数: candidate: AnswerCandidate。 必填。
    retrieval: RetrievalResult | None。
    可省略。 prompt_snapshot: PromptSnapshot
    | None。 可省略。 context_refs:
    Sequence[KnowledgeRef] | None。 可省略。
    契约: 同步调用。 返回 `LLMRequest`。
    """
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
    """函数契约说明.

    功能: 执行 _is_cancelled 的同步逻辑,并维持签名契约。
    参数: cancellation: CancellationToken
    | None。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    if cancellation is None:
        return False

    return cancellation.cancelled
