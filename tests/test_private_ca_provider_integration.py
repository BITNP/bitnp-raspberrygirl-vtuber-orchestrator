from __future__ import annotations

import ssl
from typing import TYPE_CHECKING, Literal

import pytest

from orchestrator.funasr_adapter import FunASRWebSocketAdapter
from orchestrator.llm import (
    LLMFinal,
    LLMPrompt,
    LLMRequest,
    LLMStreamEvent,
    OpenAICompatibleASRAdapter,
    VllmOmniTTSAdapter,
)
from orchestrator.media_adapters import ASRStreamRequest
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import ProviderResponseError
from tests.openai_llm_test_helper import OpenAICompatibleLLMRuntimeAdapter
from tests.private_ca import (
    PrivateCA,
    PrivateHttpsServer,
    PrivateWssServer,
    create_private_ca,
)

if TYPE_CHECKING:
    from pathlib import Path

type HttpProvider = Literal["llm", "asr", "tts"]


@pytest.fixture
def private_ca(tmp_path: Path) -> PrivateCA:
    return create_private_ca(tmp_path)


@pytest.fixture(autouse=True)
def localhost_only_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "WS_PROXY",
        "WSS_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "ws_proxy",
        "wss_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")


def test_openai_llm_https_accepts_configured_private_ca(private_ca: PrivateCA) -> None:
    # Given: a local LLM endpoint whose certificate chains to only the test CA.

    with PrivateHttpsServer(private_ca) as server:
        # When: the LLM adapter uses the configured CA path.

        events = tuple(
            OpenAICompatibleLLMRuntimeAdapter(
                endpoint=server.endpoint,
                model="local-llm",
                api_key="test-key",
                ca_path=private_ca.ca_path,
            ).stream(LLMRequest(prompt=LLMPrompt(system="system", user="question")))
        )

    # Then: the provider payload is reached and parsed through the live HTTPS path.

    assert events == (LLMFinal(text="private CA", used_fallback=False),)
    assert server.request_paths == ["/v1/chat/completions"]


def test_openai_asr_https_accepts_configured_private_ca(private_ca: PrivateCA) -> None:
    # Given: a local ASR endpoint whose certificate chains to only the test CA.

    with PrivateHttpsServer(private_ca) as server:
        # When: the ASR adapter transcribes through HTTPS with the configured CA.

        event = OpenAICompatibleASRAdapter(
            endpoint=server.endpoint,
            model="local-asr",
            ca_path=private_ca.ca_path,
        ).transcribe(
            audio=b"audio",
            filename="utterance.pcm",
            received_at_ms=10,
            segment_id="segment-1",
            seq=1,
        )

    # Then: the provider payload is reached and normalized.

    assert event == ASRAudienceEvent("private CA transcription", 10, "segment-1", 1)
    assert server.request_paths == ["/v1/audio/transcriptions"]


def test_vllm_omni_tts_https_accepts_configured_private_ca(
    private_ca: PrivateCA,
) -> None:
    # Given: a local TTS endpoint whose certificate chains to only the test CA.

    with PrivateHttpsServer(private_ca) as server:
        # When: the TTS adapter synthesizes through HTTPS with the configured CA.

        audio = VllmOmniTTSAdapter(
            endpoint=server.endpoint,
            model="local-tts",
            ca_path=private_ca.ca_path,
        ).synthesize(
            text="hello",
            voice="raspberry",
            ref_audio="file:///voice.wav",
            ref_text="reference",
        )

    # Then: the provider response is received on the live HTTPS path.

    assert audio.data == b"private-ca-audio"
    assert audio.media_type == "audio/wav"
    assert server.request_paths == ["/v1/audio/speech"]


@pytest.mark.parametrize("ca_source", ["default", "unrelated", "hostname_mismatch"])
@pytest.mark.parametrize("provider", ["llm", "asr", "tts"])
def test_https_provider_rejects_untrusted_or_hostname_mismatched_private_ca(
    private_ca: PrivateCA,
    ca_source: Literal["default", "unrelated", "hostname_mismatch"],
    provider: HttpProvider,
) -> None:
    # Given: a private-CA endpoint and a trust selection that cannot verify it.

    with PrivateHttpsServer(private_ca) as server:
        endpoint, ca_path = _https_failure_target(server, private_ca, ca_source)

        # When / Then: TLS fails before the provider can receive or parse a payload.

        with pytest.raises((ProviderResponseError, ssl.SSLCertVerificationError)):
            _ = _request_http_provider(provider, endpoint, ca_path)

    assert server.request_paths == []


def test_native_funasr_wss_accepts_configured_private_ca(private_ca: PrivateCA) -> None:
    # Given: a local FunASR WSS endpoint whose certificate chains to only the test CA.

    with PrivateWssServer(private_ca) as server:
        # When: the native FunASR adapter sends an utterance with the configured CA.

        events = tuple(
            FunASRWebSocketAdapter(
                endpoint=server.endpoint,
                model="paraformer",
                ca_path=private_ca.ca_path,
            ).stream(_asr_request())
        )

    # Then: the complete native protocol reaches the server and parses its final event.

    assert events == (ASRAudienceEvent("private CA transcription", 10, "segment-1", 1),)
    assert len(server.received_messages) == 3


@pytest.mark.parametrize("ca_source", ["default", "unrelated", "hostname_mismatch"])
def test_native_funasr_wss_rejects_untrusted_or_hostname_mismatched_private_ca(
    private_ca: PrivateCA,
    ca_source: Literal["default", "unrelated", "hostname_mismatch"],
) -> None:
    # Given: a private-CA native WSS endpoint and an invalid trust selection.

    with PrivateWssServer(private_ca) as server:
        endpoint, ca_path = _wss_failure_target(server, private_ca, ca_source)

        # When / Then: the TLS handshake fails before the FunASR protocol starts.

        with pytest.raises(ssl.SSLCertVerificationError):
            _ = tuple(
                FunASRWebSocketAdapter(endpoint, "paraformer", ca_path=ca_path).stream(
                    _asr_request()
                )
            )

    assert server.received_messages == []


def _request_http_provider(
    provider: HttpProvider, endpoint: str, ca_path: Path | None
) -> tuple[LLMStreamEvent, ...] | ASRAudienceEvent | bytes | None:
    match provider:
        case "llm":
            return tuple(
                OpenAICompatibleLLMRuntimeAdapter(
                    endpoint=endpoint,
                    model="local-llm",
                    api_key="test-key",
                    ca_path=ca_path,
                ).stream(LLMRequest(prompt=LLMPrompt(system="system", user="question")))
            )
        case "asr":
            return OpenAICompatibleASRAdapter(
                endpoint=endpoint,
                model="local-asr",
                ca_path=ca_path,
            ).transcribe(
                audio=b"audio",
                filename="utterance.pcm",
                received_at_ms=10,
                segment_id="segment-1",
                seq=1,
            )
        case "tts":
            return (
                VllmOmniTTSAdapter(
                    endpoint=endpoint,
                    model="local-tts",
                    ca_path=ca_path,
                )
                .synthesize(
                    text="hello",
                    voice="raspberry",
                    ref_audio="file:///voice.wav",
                    ref_text="reference",
                )
                .data
            )


def _https_failure_target(
    server: PrivateHttpsServer,
    private_ca: PrivateCA,
    ca_source: Literal["default", "unrelated", "hostname_mismatch"],
) -> tuple[str, Path | None]:
    match ca_source:
        case "default":
            return server.endpoint, None
        case "unrelated":
            return server.endpoint, private_ca.unrelated_ca_path
        case "hostname_mismatch":
            return server.endpoint.replace("localhost", "127.0.0.1"), private_ca.ca_path


def _wss_failure_target(
    server: PrivateWssServer,
    private_ca: PrivateCA,
    ca_source: Literal["default", "unrelated", "hostname_mismatch"],
) -> tuple[str, Path | None]:
    match ca_source:
        case "default":
            return server.endpoint, None
        case "unrelated":
            return server.endpoint, private_ca.unrelated_ca_path
        case "hostname_mismatch":
            return server.endpoint.replace("localhost", "127.0.0.1"), private_ca.ca_path


def _asr_request() -> ASRStreamRequest:
    return ASRStreamRequest(
        audio=b"audio",
        filename="utterance.pcm",
        received_at_ms=10,
        segment_id="segment-1",
        seq=1,
    )
