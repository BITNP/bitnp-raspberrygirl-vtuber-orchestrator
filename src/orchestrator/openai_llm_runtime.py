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

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value

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
    LLMWorkload,
)
from orchestrator.llm_settings import LLMGenerationConfig, ReasoningDialect
from orchestrator.provider_streaming import ProviderDeadlines, ProviderResponseError

_LOGGER = logging.getLogger(__name__)
_MAX_JSON_RESPONSE_BYTES = 1_048_576
_JSON_READ_CHUNK_BYTES = 16_384

_JSON_REQUEST_LOG = "llm_json_request workload=%s model=%s schema=%s default_dialect=%s request_reasoning=%s request_temperature=%s request_max_completion_tokens=%d system=%r user=%r"  # noqa: E501

_STREAM_REQUEST_LOG = "llm_request workload=%s model=%s default_dialect=%s request_reasoning=%s request_temperature=%s request_max_completion_tokens=%d system=%r user=%r"  # noqa: E501


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
    reasoning_dialect: ReasoningDialect
    brain_model: str | None = None
    maintenance_model: str | None = None
    timeout_seconds: float = 120.0
    deadlines: ProviderDeadlines = field(default_factory=ProviderDeadlines)
    generation: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    ca_path: Path | None = None
    http_client: httpx.AsyncClient | None = None
    _client: AsyncOpenAI = field(init=False, repr=False)
    _http_client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.endpoint.strip() == "":
            raise AdapterConfigError(field_name="endpoint")
        if self.model.strip() == "":
            raise AdapterConfigError(field_name="model")
        if self.api_key.strip() == "":
            raise AdapterConfigError(field_name="api_key")
        if self.reasoning_dialect not in {"deepseek", "openai", "none"}:
            raise AdapterConfigError(field_name="reasoning_dialect")
        for field_name, configured_model in (
            ("brain_model", self.brain_model),
            ("maintenance_model", self.maintenance_model),
        ):
            if configured_model is not None and configured_model.strip() == "":
                raise AdapterConfigError(field_name=field_name)
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
        self._http_client = client
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=f"{self.endpoint.rstrip('/')}/",
            timeout=timeout,
            max_retries=0,
            http_client=client,
        )

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def complete_json(
        self,
        request: LLMRequest,
        *,
        schema_name: str,
        schema: dict[str, object],
    ) -> str:
        """Request one non-streaming strict JSON proposal from the LLM Brain."""
        if schema_name.strip() == "":
            raise AdapterConfigError(field_name="schema_name")
        # Compatible providers support json_object. The caller still validates
        # the proposal against this schema before it can cause effects.
        _ = schema
        model = self._model_for(request.workload)
        _LOGGER.debug(
            _JSON_REQUEST_LOG,
            request.workload,
            model,
            schema_name,
            self.reasoning_dialect,
            request.reasoning,
            request.temperature,
            request.max_completion_tokens,
            request.prompt.system,
            request.prompt.user,
        )
        try:
            body: dict[str, object] = {
                "model": model,
                "messages": chat_messages(request),
                **self._generation_parameters(request),
                "stream": False,
                "response_format": {"type": "json_object"},
            }
            async with self._http_client.stream(
                "POST",
                f"{self.endpoint.rstrip('/')}/chat/completions",
                json=body,
                timeout=self._request_timeout(request.timeout_seconds),
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as response:
                _ = response.raise_for_status()
                response_text = await _read_bounded_json_response(response)
            _LOGGER.debug(
                "llm_json_http_response model=%s schema=%s body=%r",
                model,
                schema_name,
                response_text,
            )
            try:
                payload = parse_json_value(response_text)
            except JsonBoundaryError as error:
                raise ProviderResponseError(
                    stage="llm", reason="missing_final"
                ) from error
            if not isinstance(payload, dict):
                raise ProviderResponseError(stage="llm", reason="missing_final")
            parsed_payload = cast("dict[str, object]", payload)
            choices = parsed_payload.get("choices")
            parsed_choices = (
                [] if not isinstance(choices, list) else cast("list[object]", choices)
            )
            if len(parsed_choices) != 1:
                details = (
                    f"schema={schema_name} reason=choice_count "
                    f"count={len(parsed_choices)}"
                )
                _LOGGER.debug(
                    "llm_json_invalid_response model=%s %s",
                    model,
                    details,
                )
                raise ProviderResponseError(stage="llm", reason="missing_final")
            choice_value = parsed_choices[0]
            if not isinstance(choice_value, dict):
                raise ProviderResponseError(stage="llm", reason="missing_final")
            choice = cast("dict[str, object]", choice_value)
            message_value = choice.get("message")
            if not isinstance(message_value, dict):
                raise ProviderResponseError(stage="llm", reason="missing_final")
            message = cast("dict[str, object]", message_value)
            content = message.get("content")
            if not isinstance(content, str) or content.strip() == "":
                reasoning = message.get("reasoning_content")
                reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
                details = (
                    f"schema={schema_name} reason=missing_content "
                    f"finish_reason={choice.get('finish_reason')} "
                    f"content_type={type(content).__name__} "
                    f"reasoning_chars={reasoning_chars}"
                )
                _LOGGER.debug(
                    "llm_json_invalid_response model=%s %s",
                    model,
                    details,
                )
                raise ProviderResponseError(stage="llm", reason="missing_final")
            _LOGGER.debug(
                "llm_json_response model=%s schema=%s text=%r",
                model,
                schema_name,
                content,
            )
            return content  # noqa: TRY300 - exception conversion belongs below.
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
        model = self._model_for(request.workload)
        _LOGGER.debug(
            _STREAM_REQUEST_LOG,
            request.workload,
            model,
            self.reasoning_dialect,
            request.reasoning,
            request.temperature,
            request.max_completion_tokens,
            request.prompt.system,
            request.prompt.user,
        )
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=chat_messages(request),
                stream=True,
                extra_body=self._generation_parameters(request),
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
                final_text = "".join(chunks)
                _LOGGER.debug("llm_response text=%r", final_text)
                yield LLMFinal(text=final_text, used_fallback=False)
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

    def _model_for(self, workload: LLMWorkload) -> str:
        match workload:
            case LLMWorkload.BRAIN:
                return self.brain_model or self.model
            case LLMWorkload.MAINTENANCE:
                return self.maintenance_model or self.model

    def _generation_parameters(self, request: LLMRequest) -> dict[str, object]:
        settings = (
            self.generation.brain
            if request.workload is LLMWorkload.BRAIN
            else self.generation.maintenance
        )
        parameters = settings.parameters(
            temperature=request.temperature,
            reasoning=request.reasoning.value,
            max_completion_tokens=request.max_completion_tokens,
            dialect=self.reasoning_dialect,
        )
        _LOGGER.debug(
            "llm_generation_parameters workload=%s model=%s parameters=%r",
            request.workload,
            self._model_for(request.workload),
            parameters,
        )
        return parameters


async def _read_bounded_json_response(response: httpx.Response) -> str:
    """Bound decoded response bytes before parsing or logging provider content."""
    content = bytearray()
    async for chunk in response.aiter_bytes(chunk_size=_JSON_READ_CHUNK_BYTES):
        if len(content) + len(chunk) > _MAX_JSON_RESPONSE_BYTES:
            raise ProviderResponseError(stage="llm", reason="response_too_large")
        content.extend(chunk)
    try:
        return content.decode("utf-8")
    except UnicodeError as error:
        raise ProviderResponseError(stage="llm", reason="invalid_encoding") from error
