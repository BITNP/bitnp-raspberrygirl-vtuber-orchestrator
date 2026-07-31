from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.media_adapters import (
    ASRPartialEvent,
    ASRStreamEvent,
    ASRStreamRequest,
    MediaAdapterConfigError,
)
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import (
    ProviderCancellationHandle,
    ProviderDeadlines,
    ProviderResponseError,
)
from orchestrator.tls import build_tls_context

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class _FunASRConnection(Protocol):
    def send(self, message: str | bytes) -> None: ...

    def recv(self, timeout: float | None = None) -> str | bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FunASRWebSocketAdapter:
    endpoint: str

    model: str

    ca_path: Path | None = None

    deadlines: ProviderDeadlines = field(default_factory=ProviderDeadlines)

    def __post_init__(self) -> None:
        if self.endpoint.strip() == "":
            raise MediaAdapterConfigError(field_name="endpoint")

        if self.model.strip() == "":
            raise MediaAdapterConfigError(field_name="model")

    @property
    def capability(self) -> str:
        return "streaming"

    def transcribe(  # noqa: PLR0913
        self,
        *,
        audio: bytes,
        filename: str,
        received_at_ms: int,
        segment_id: str,
        seq: int,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> ASRAudienceEvent | None:
        final: ASRAudienceEvent | None = None

        for event in self.stream(
            ASRStreamRequest(audio, filename, received_at_ms, segment_id, seq),
            cancellation=cancellation,
        ):
            match event:
                case ASRPartialEvent():
                    continue

                case ASRAudienceEvent():
                    final = event

        return final

    def stream(
        self,
        request: ASRStreamRequest,
        *,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> Iterator[ASRStreamEvent]:
        if cancellation is not None and cancellation.cancelled:
            return

        _LOGGER.debug(
            "asr_funasr_request endpoint=%s segment=%s audio_bytes=%d filename=%s",
            self.endpoint,
            request.segment_id,
            len(request.audio),
            request.filename,
        )

        tls_context = build_tls_context(self.ca_path)

        if self.endpoint.startswith("wss://") and tls_context is not None:
            connection = connect(
                self.endpoint,
                open_timeout=self.deadlines.connect_seconds,
                ssl=tls_context,
            )
        else:
            connection = connect(
                self.endpoint, open_timeout=self.deadlines.connect_seconds
            )

        release = _noop if cancellation is None else cancellation.bind(connection.close)

        try:
            if cancellation is not None and cancellation.cancelled:
                return

            connection.send(_start_message(request.filename))

            connection.send(request.audio)

            connection.send(_end_message())

            yield from _receive_events(
                connection, request, self.deadlines, cancellation
            )

        finally:
            release()

            connection.close()


def _start_message(filename: str) -> str:
    return json.dumps(
        {
            "mode": "2pass",
            "chunk_size": [5, 10, 5],
            "wav_name": filename,
            "wav_format": "pcm",
            "audio_fs": 16_000,
            "is_speaking": True,
        },
        separators=(",", ":"),
    )


def _end_message() -> str:
    return json.dumps({"is_speaking": False}, separators=(",", ":"))


def _receive_events(
    connection: _FunASRConnection,
    request: ASRStreamRequest,
    deadlines: ProviderDeadlines,
    cancellation: ProviderCancellationHandle | None,
) -> Iterator[ASRStreamEvent]:
    final_emitted = False

    started = time.monotonic()

    while True:
        try:
            message = connection.recv(timeout=deadlines.read_seconds)

        except (ConnectionClosed, OSError, TimeoutError) as error:
            if cancellation is not None and cancellation.cancelled:
                return

            if final_emitted:
                return

            raise ProviderResponseError(stage="asr", reason="read") from error

        if time.monotonic() - started > deadlines.total_seconds:
            raise ProviderResponseError(stage="asr", reason="total")

        event = _normalize_funasr_message(message, request)

        if event is None:
            continue

        if isinstance(event, ASRPartialEvent):
            _LOGGER.debug(
                "asr_funasr_response kind=partial segment=%s chars=%d",
                request.segment_id,
                len(event.text),
            )
            yield event

            continue

        if final_emitted:
            raise ProviderResponseError(stage="asr", reason="duplicate_final")

        final_emitted = True

        _LOGGER.debug(
            "asr_funasr_response kind=final segment=%s chars=%d",
            request.segment_id,
            len(event.text),
        )

        yield event


def _normalize_funasr_message(
    message: str | bytes, request: ASRStreamRequest
) -> ASRStreamEvent | None:
    if not isinstance(message, str):
        raise ProviderResponseError(stage="asr", reason="event")

    try:
        payload = parse_json_value(message)

    except JsonBoundaryError as error:
        raise ProviderResponseError(stage="asr", reason="json") from error

    if not isinstance(payload, dict):
        raise ProviderResponseError(stage="asr", reason="event")

    text = payload.get("text")

    if not isinstance(text, str) or text.strip() == "":
        return None

    is_final = payload.get("is_final")

    if is_final is not None and not isinstance(is_final, bool):
        raise ProviderResponseError(stage="asr", reason="event")

    mode = payload.get("mode")

    if mode is not None and not isinstance(mode, str):
        raise ProviderResponseError(stage="asr", reason="event")

    normalized_text = text.strip()

    if is_final is True or mode == "2pass-offline":
        return ASRAudienceEvent(
            normalized_text,
            request.received_at_ms,
            request.segment_id,
            request.seq,
        )

    return ASRPartialEvent(
        normalized_text,
        request.received_at_ms,
        request.segment_id,
        request.seq,
    )


def _noop() -> None:
    return
