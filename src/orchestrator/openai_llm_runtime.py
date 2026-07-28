"""OpenAI-compatible synchronous LLM provider for onsite turn processing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.llm import (
    AdapterConfigError,
    CancellationToken,
    LLMFinal,
    LLMRequest,
    LLMStreamEvent,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLLMRuntimeAdapter:
    """Turns one OpenAI-compatible completion into the pipeline's final event."""

    endpoint: str
    model: str
    api_key: str
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate required provider configuration before issuing HTTP requests."""
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
        """Request one completion and yield its normalized final answer."""
        if cancellation is not None and cancellation.cancelled:
            return
        response = _post(
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
            self.api_key,
            self.timeout_seconds,
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


def _post(url: str, body: bytes, api_key: str, timeout_seconds: float) -> bytes:
    parsed = urlsplit(url)
    path = parsed.path if parsed.path != "" else "/"
    connection: HTTPConnection | HTTPSConnection
    if parsed.scheme == "http":
        connection = HTTPConnection(parsed.netloc, timeout=timeout_seconds)
    elif parsed.scheme == "https":
        connection = HTTPSConnection(parsed.netloc, timeout=timeout_seconds)
    else:
        raise AdapterConfigError(field_name="endpoint")
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        return connection.getresponse().read()
    finally:
        connection.close()
