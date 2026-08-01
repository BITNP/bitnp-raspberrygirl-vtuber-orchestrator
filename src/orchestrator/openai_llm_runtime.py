from __future__ import annotations

import json
import logging
import ssl
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

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

    from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam


from orchestrator.llm import (
    AdapterConfigError,
    CancellationToken,
    LLMChunk,
    LLMFinal,
    LLMRequest,
    LLMStreamEvent,
)
from orchestrator.provider_streaming import (
    ProviderCapability,
    ProviderDeadlines,
    ProviderResponseError,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLLMRuntimeAdapter:
    """LLM adapter backed by the official OpenAI Python SDK."""

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

        _LOGGER.debug(
            "llm_request endpoint=%s model=%s capability=%s system=%d user=%d",
            self.endpoint,
            self.model,
            self.capability,
            len(request.prompt.system),
            len(request.prompt.user),
        )
        match self.capability:
            case "final_only":
                yield from self._stream_final_only(request, cancellation)
            case "streaming":
                yield from self._stream_chat_completions(request, cancellation)

    def _stream_final_only(
        self, request: LLMRequest, cancellation: CancellationToken | None
    ) -> Iterator[LLMStreamEvent]:
        client = self._client(request)
        release = _bind_cancellation(cancellation, client.close)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=_messages(request),
                temperature=request.temperature,
                timeout=self._request_timeout(request),
            )
            if cancellation is not None and cancellation.cancelled:
                return
            if len(response.choices) != 1:
                raise AdapterConfigError(field_name="response.choices")
            text = response.choices[0].message.content
            if not isinstance(text, str) or text.strip() == "":
                raise AdapterConfigError(field_name="response.message.content")
            final = text.strip()
            _LOGGER.debug("llm_response kind=final chars=%d", len(final))
            yield LLMFinal(text=final, used_fallback=False)
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
            json.JSONDecodeError,
        ) as error:
            if cancellation is not None and cancellation.cancelled:
                return
            raise _provider_error(error) from error
        finally:
            release()
            client.close()

    def _stream_chat_completions(
        self, request: LLMRequest, cancellation: CancellationToken | None
    ) -> Iterator[LLMStreamEvent]:
        client = self._client(request)
        release = _bind_cancellation(cancellation, client.close)
        stream: Stream[ChatCompletionChunk] | None = None
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=_messages(request),
                temperature=request.temperature,
                stream=True,
                timeout=self._request_timeout(request),
            )
            stream_release = _bind_cancellation(cancellation, stream.close)
            try:
                chunks: list[str] = []
                for chunk in stream:
                    if cancellation is not None and cancellation.cancelled:
                        return
                    if len(chunk.choices) == 0:
                        continue
                    text = chunk.choices[0].delta.content
                    if text is None:
                        continue
                    chunks.append(text)
                    _LOGGER.debug(
                        "llm_response kind=chunk index=%d chars=%d",
                        len(chunks) - 1,
                        len(text),
                    )
                    yield LLMChunk(index=len(chunks) - 1, text=text)
                if cancellation is None or not cancellation.cancelled:
                    if len(chunks) == 0:
                        raise ProviderResponseError(stage="llm", reason="missing_final")
                    final = "".join(chunks)
                    _LOGGER.debug(
                        "llm_response kind=final chars=%d chunks=%d",
                        len(final),
                        len(chunks),
                    )
                    yield LLMFinal(text=final, used_fallback=False)
            finally:
                stream_release()
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            APIError,
            httpx.HTTPError,
            json.JSONDecodeError,
        ) as error:
            if cancellation is not None and cancellation.cancelled:
                return
            raise _provider_error(error) from error
        finally:
            if stream is not None:
                stream.close()
            release()
            client.close()

    def _client(self, request: LLMRequest) -> OpenAI:
        timeout = self._request_timeout(request)
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

    def _request_timeout(self, request: LLMRequest) -> httpx.Timeout:
        total = min(
            request.timeout_seconds, self.timeout_seconds, self.deadlines.total_seconds
        )
        return httpx.Timeout(
            timeout=total,
            connect=self.deadlines.connect_seconds,
            read=self.deadlines.read_seconds,
            write=total,
        )


def _messages(request: LLMRequest) -> list[ChatCompletionMessageParam]:
    return cast(
        "list[ChatCompletionMessageParam]",
        [
            {"role": "system", "content": request.prompt.system},
            {"role": "user", "content": request.prompt.user},
        ],
    )


def _bind_cancellation(
    cancellation: CancellationToken | None, callback: Callable[[], None]
) -> Callable[[], None]:
    if cancellation is None:
        return lambda: None
    return cancellation.bind(callback)


def _provider_error(
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
