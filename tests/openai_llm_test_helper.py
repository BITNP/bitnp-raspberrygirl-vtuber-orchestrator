"""Synchronous OpenAI SDK adapter retained exclusively for legacy test fixtures."""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    Stream,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from openai.types.chat import ChatCompletionChunk

from orchestrator.llm import (
    AdapterConfigError,
    CancellationToken,
    LLMChunk,
    LLMFinal,
    LLMRequest,
    LLMStreamEvent,
)
from orchestrator.openai_llm_runtime import chat_messages, provider_error
from orchestrator.provider_streaming import (
    ProviderCapability,
    ProviderDeadlines,
    ProviderResponseError,
)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLLMRuntimeAdapter:
    """Test-only synchronous adapter for existing provider-boundary fixtures."""

    endpoint: str
    model: str
    api_key: str
    timeout_seconds: float = 30.0
    capability: ProviderCapability = "final_only"
    deadlines: ProviderDeadlines = field(default_factory=ProviderDeadlines)
    ca_path: Path | None = None

    def __post_init__(self) -> None:
        if self.endpoint.strip() == "":
            raise AdapterConfigError(field_name="endpoint")
        if self.model.strip() == "":
            raise AdapterConfigError(field_name="model")
        if self.api_key.strip() == "":
            raise AdapterConfigError(field_name="api_key")

    def stream(
        self,
        request: LLMRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[LLMStreamEvent]:
        if cancellation is not None and cancellation.cancelled:
            return
        if self.capability == "final_only":
            yield from self._final(request, cancellation)
        else:
            yield from self._streaming(request, cancellation)

    def _final(
        self, request: LLMRequest, cancellation: CancellationToken | None
    ) -> Iterator[LLMStreamEvent]:
        client = self._client(request)
        release = _bind(cancellation, client.close)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=chat_messages(request),
                temperature=request.temperature,
                timeout=self._timeout(request),
            )
            if cancellation is not None and cancellation.cancelled:
                return
            if len(response.choices) != 1:
                raise AdapterConfigError(field_name="response.choices")
            content = response.choices[0].message.content
            if not isinstance(content, str) or content.strip() == "":
                raise AdapterConfigError(field_name="response.message.content")
            yield LLMFinal(text=content.strip(), used_fallback=False)
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
            json.JSONDecodeError,
        ) as error:
            if cancellation is None or not cancellation.cancelled:
                raise provider_error(error) from error
        finally:
            release()
            client.close()

    def _streaming(
        self, request: LLMRequest, cancellation: CancellationToken | None
    ) -> Iterator[LLMStreamEvent]:
        client = self._client(request)
        stream: Stream[ChatCompletionChunk] | None = None
        release = _bind(cancellation, client.close)
        stream_release = _noop
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=chat_messages(request),
                temperature=request.temperature,
                stream=True,
                timeout=self._timeout(request),
            )
            stream_release = _bind(cancellation, stream.close)
            parts: list[str] = []
            for chunk in stream:
                if cancellation is not None and cancellation.cancelled:
                    return
                if not chunk.choices or chunk.choices[0].delta.content is None:
                    continue
                text = chunk.choices[0].delta.content
                parts.append(text)
                yield LLMChunk(index=len(parts) - 1, text=text)
            if cancellation is None or not cancellation.cancelled:
                if not parts:
                    raise ProviderResponseError(stage="llm", reason="missing_final")
                yield LLMFinal(text="".join(parts), used_fallback=False)
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
            json.JSONDecodeError,
        ) as error:
            if cancellation is None or not cancellation.cancelled:
                raise provider_error(error) from error
        finally:
            stream_release()
            if stream is not None:
                stream.close()
            release()
            client.close()

    def _client(self, request: LLMRequest) -> OpenAI:
        timeout = self._timeout(request)
        verify: bool | ssl.SSLContext = (
            True
            if self.ca_path is None
            else ssl.create_default_context(cafile=self.ca_path)
        )
        return OpenAI(
            api_key=self.api_key,
            base_url=f"{self.endpoint.rstrip('/')}/",
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(verify=verify, timeout=timeout, trust_env=False),
        )

    def _timeout(self, request: LLMRequest) -> httpx.Timeout:
        total = min(
            request.timeout_seconds, self.timeout_seconds, self.deadlines.total_seconds
        )
        return httpx.Timeout(
            timeout=total,
            connect=self.deadlines.connect_seconds,
            read=self.deadlines.read_seconds,
            write=total,
        )


def _bind(
    cancellation: CancellationToken | None, callback: Callable[[], None]
) -> Callable[[], None]:
    return (lambda: None) if cancellation is None else cancellation.bind(callback)


def _noop() -> None:
    return
