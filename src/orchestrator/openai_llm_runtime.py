from __future__ import annotations

import asyncio
import logging
import ssl
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

if TYPE_CHECKING:
    import json
    from collections.abc import AsyncIterator
    from pathlib import Path

    from openai.types.chat import ChatCompletionMessageParam


from orchestrator.llm import (
    AdapterConfigError,
    CancellationToken,
    LLMChunk,
    LLMFinal,
    LLMRequest,
    LLMStreamEvent,
)
from orchestrator.provider_streaming import ProviderDeadlines, ProviderResponseError

_LOGGER = logging.getLogger(__name__)


def _noop() -> None:
    return


def _consume_task_result(task: asyncio.Task[None]) -> None:
    with suppress(asyncio.CancelledError, Exception):
        task.result()


def chat_messages(request: LLMRequest) -> list[ChatCompletionMessageParam]:
    return cast(
        "list[ChatCompletionMessageParam]",
        [
            {"role": "system", "content": request.prompt.system},
            {"role": "user", "content": request.prompt.user},
        ],
    )


def provider_error(
    error: APIError | httpx.HTTPError | json.JSONDecodeError,
) -> ProviderResponseError:
    if isinstance(error, APIStatusError):
        return ProviderResponseError(stage="llm", reason=f"status_{error.status_code}")
    if isinstance(error, APITimeoutError):
        return ProviderResponseError(stage="llm", reason="read")
    if isinstance(error, httpx.TimeoutException):
        return ProviderResponseError(stage="llm", reason="read")
    if isinstance(error, APIConnectionError):
        return ProviderResponseError(stage="llm", reason="connect")
    return ProviderResponseError(stage="llm", reason="response")


@dataclass(slots=True)
class AsyncOpenAICompatibleLLMRuntime:
    """Shared asynchronous Chat Completions client used by the live runtime.

    The synchronous adapter above remains only for compatibility with old
    one-shot tools and their tests.  The service path uses this class so no
    blocking SDK call is ever made from the event-loop thread.
    """

    endpoint: str
    model: str
    api_key: str
    timeout_seconds: float = 120.0
    deadlines: ProviderDeadlines = field(default_factory=ProviderDeadlines)
    ca_path: Path | None = None
    http_client: httpx.AsyncClient | None = None
    _client: AsyncOpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.endpoint.strip() == "":
            raise AdapterConfigError(field_name="endpoint")
        if self.model.strip() == "":
            raise AdapterConfigError(field_name="model")
        if self.api_key.strip() == "":
            raise AdapterConfigError(field_name="api_key")
        timeout = self._request_timeout(self.timeout_seconds)
        client = self.http_client
        if client is None:
            verify: bool | ssl.SSLContext = (
                True
                if self.ca_path is None
                else (
                    ssl.create_default_context(cafile=self.ca_path)
                    if self.ca_path.exists()
                    else True
                )
            )
            client = httpx.AsyncClient(verify=verify, timeout=timeout, trust_env=False)
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=f"{self.endpoint.rstrip('/')}/",
            timeout=timeout,
            max_retries=0,
            http_client=client,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def complete_gate(self, request: LLMRequest) -> str:
        """Return the gate JSON response, failing closed at the caller."""
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=chat_messages(request),
                temperature=0.0,
                stream=False,
                response_format={"type": "json_object"},
                reasoning_effort="none",
                timeout=self._request_timeout(5.0),
            )
            if len(response.choices) != 1:
                raise ProviderResponseError(stage="llm", reason="missing_final")
            content = response.choices[0].message.content
            if not isinstance(content, str) or content.strip() == "":
                raise ProviderResponseError(stage="llm", reason="missing_final")
            return content  # noqa: TRY300
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
        ) as error:
            raise provider_error(error) from error

    async def stream(  # noqa: C901
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        if cancellation is not None and cancellation.cancelled:
            return
        stream = None
        release = _noop
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=chat_messages(request),
                temperature=request.temperature,
                stream=True,
                timeout=self._request_timeout(request.timeout_seconds),
            )
            if cancellation is not None:
                loop = asyncio.get_running_loop()

                def close_stream() -> None:
                    task = loop.create_task(stream.close())
                    task.add_done_callback(_consume_task_result)

                release = cancellation.bind(close_stream)
            chunks: list[str] = []
            async for chunk in stream:
                if cancellation is not None and cancellation.cancelled:
                    return
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                if text is None:
                    continue
                chunks.append(text)
                yield LLMChunk(index=len(chunks) - 1, text=text)
            if cancellation is None or not cancellation.cancelled:
                if not chunks:
                    raise ProviderResponseError(stage="llm", reason="missing_final")
                yield LLMFinal(text="".join(chunks), used_fallback=False)
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
        ) as error:
            if cancellation is None or not cancellation.cancelled:
                raise provider_error(error) from error
        finally:
            release()
            if stream is not None:
                await stream.close()

    def _request_timeout(self, requested_seconds: float) -> httpx.Timeout:
        total = min(
            requested_seconds, self.timeout_seconds, self.deadlines.total_seconds
        )
        return httpx.Timeout(
            timeout=total,
            connect=self.deadlines.connect_seconds,
            read=self.deadlines.read_seconds,
            write=total,
        )
