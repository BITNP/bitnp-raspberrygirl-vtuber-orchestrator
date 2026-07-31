
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
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
    ProviderRequest,
    ProviderResponseError,
    post_bytes,
    post_sse,
)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLLMRuntimeAdapter:

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

        match self.capability:
            case "final_only":
                yield from self._stream_final_only(request, cancellation)

            case "streaming":
                yield from self._stream_sse(request, cancellation)

    def _stream_final_only(
        self, request: LLMRequest, cancellation: CancellationToken | None
    ) -> Iterator[LLMStreamEvent]:
        response = post_bytes(
            ProviderRequest(
                f"{self.endpoint.rstrip('/')}/chat/completions",
                json.dumps(
                    {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": request.prompt.system},
                            {"role": "user", "content": request.prompt.user},
                        ],
                        "stream": False,
                        "temperature": request.temperature,
                    }
                ).encode(),
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                "llm",
                self.ca_path,
            ),
            deadlines=ProviderDeadlines(
                connect_seconds=self.deadlines.connect_seconds,
                read_seconds=self.deadlines.read_seconds,
                total_seconds=min(self.timeout_seconds, self.deadlines.total_seconds),
            ),
            cancellation=cancellation,
        )

        if cancellation is not None and cancellation.cancelled:
            return

        try:
            payload = parse_json_value(response.decode())

        except JsonBoundaryError as error:
            raise AdapterConfigError(field_name=error.field_name) from error

        if not isinstance(payload, dict):
            raise AdapterConfigError(field_name="response")

        choices = payload.get("choices")

        if not isinstance(choices, list) or len(choices) != 1:
            raise AdapterConfigError(field_name="response.choices")

        choice = choices[0]

        if not isinstance(choice, dict):
            raise AdapterConfigError(field_name="response.choices")

        message = choice.get("message")

        if not isinstance(message, dict):
            raise AdapterConfigError(field_name="response.message")

        text = message.get("content")

        if not isinstance(text, str) or text.strip() == "":
            raise AdapterConfigError(field_name="response.message.content")

        yield LLMFinal(text=text.strip(), used_fallback=False)

    def _stream_sse(
        self, request: LLMRequest, cancellation: CancellationToken | None
    ) -> Iterator[LLMStreamEvent]:
        chunks: list[str] = []

        done = False

        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": request.prompt.system},
                    {"role": "user", "content": request.prompt.user},
                ],
                "stream": True,
                "temperature": request.temperature,
            }
        ).encode()

        for data in post_sse(
            ProviderRequest(
                f"{self.endpoint.rstrip('/')}/chat/completions",
                body,
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                "llm",
                self.ca_path,
            ),
            deadlines=self.deadlines,
            cancellation=cancellation,
        ):
            if data == "[DONE]":
                done = True

                break

            text = _sse_delta(data)

            chunks.append(text)

            yield LLMChunk(index=len(chunks) - 1, text=text)

        if cancellation is None or not cancellation.cancelled:
            if not done or len(chunks) == 0:
                raise ProviderResponseError(stage="llm", reason="missing_final")

            yield LLMFinal(text="".join(chunks), used_fallback=False)


def _sse_delta(data: str) -> str:
    try:
        payload = parse_json_value(data)

    except JsonBoundaryError as error:
        raise ProviderResponseError(stage="llm", reason="json") from error

    if not isinstance(payload, dict):
        raise ProviderResponseError(stage="llm", reason="event")

    choices = payload.get("choices")

    if not isinstance(choices, list) or len(choices) != 1:
        raise ProviderResponseError(stage="llm", reason="event")

    choice = choices[0]

    if not isinstance(choice, dict):
        raise ProviderResponseError(stage="llm", reason="event")

    delta = choice.get("delta")

    if not isinstance(delta, dict):
        raise ProviderResponseError(stage="llm", reason="event")

    content = delta.get("content")

    if not isinstance(content, str) or content == "":
        raise ProviderResponseError(stage="llm", reason="event")

    return content
