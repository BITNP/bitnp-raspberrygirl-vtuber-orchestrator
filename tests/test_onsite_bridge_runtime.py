
from __future__ import annotations

import asyncio
import io
import json
import threading
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.llm import MockLLMAdapter
from orchestrator.media_adapters import SynthesizedAudio
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import ASRAudienceEvent, PipelineConfig
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.transport_config import TransportConfig
from orchestrator.transport_hub import RtpHub
from orchestrator.transport_runtime import ControlHandler, TransportRuntime

if TYPE_CHECKING:
    from orchestrator.json_boundary import JsonValue
    from orchestrator.provider_streaming import ProviderCancellationHandle
    from orchestrator.transport_hub import RtpHub


@dataclass(slots=True)
class _Datagrams:

    sent: list[tuple[bytes, tuple[str, int]]] = field(default_factory=list)

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:

        self.sent.append((data, addr))

    def close(self) -> None:

        return


@dataclass(frozen=True, slots=True)
class _ControlServer:

    def close(self) -> None:

        return

    async def wait_closed(self) -> None:

        return


@dataclass(slots=True)
class _DatagramListener:

    transport: _Datagrams

    hub: RtpHub | None = None

    async def listen(self, _host: str, _port: int, hub: RtpHub) -> _Datagrams:

        self.hub = hub

        return self.transport


@dataclass(slots=True)
class _DelayedAsr:

    started: threading.Event = field(default_factory=threading.Event)

    release: threading.Event = field(default_factory=threading.Event)

    completed: threading.Event = field(default_factory=threading.Event)

    cancelled: threading.Event = field(default_factory=threading.Event)

    def transcribe(  # noqa: PLR0913
        self,
        *,
        audio: bytes,
        filename: str,
        received_at_ms: int,
        segment_id: str,
        seq: int,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> ASRAudienceEvent:

        _ = (audio, filename)

        if cancellation is not None:
            _ = cancellation.bind(self._cancel)

        _ = self.started.set()

        _ = self.release.wait()

        _ = self.completed.set()

        return ASRAudienceEvent("Explain BitNet", received_at_ms, segment_id, seq)

    def _cancel(self) -> None:

        _ = self.cancelled.set()

        _ = self.release.set()


@dataclass(frozen=True, slots=True)
class _Tts:

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:

        _ = (text, voice, ref_audio, ref_text, cancellation)

        return SynthesizedAudio(_wav(b"\x10\x20" * 320), "audio/wav")


def test_runtime_processes_cancellation_while_provider_runs_and_drops_stale_rtp() -> (
    None
):

    asyncio.run(_cancellation_proof())


async def _cancellation_proof() -> None:
    # Given: a registered onsite route whose provider work waits on an explicit signal.


    transport = _Datagrams()

    listener = _DatagramListener(transport)

    asr = _DelayedAsr()

    runtime = TransportRuntime(
        _loopback_config(),
        datagram_listener=listener.listen,
        control_listener=_control_listener,
        onsite_bridge=_bridge(asr),
    )

    await runtime.start()

    assert listener.hub is not None

    listener.hub.register_control(_source_registration(), "127.0.0.1")

    listener.hub.register_control(_sink_registration(), "127.0.0.1")

    # When: Mic work begins, control cancels its route, then the provider completes.

    assert runtime.route_datagram(_rtp_packet(), ("127.0.0.1", 41_000)) is False

    try:
        _ = await asyncio.to_thread(asr.started.wait)

        await runtime.cancel_stream("session-onsite-runtime", "stream-onsite-runtime")

    finally:
        asr.release.set()

    await runtime.wait_for_onsite_jobs()

    await runtime.close()

    # Then: the callback released the loop and no stale RTP reached Sound.

    assert transport.sent == []


def test_runtime_close_cancels_blocking_asr_and_drops_its_late_output() -> None:

    asyncio.run(_close_before_release_proof())


async def _close_before_release_proof() -> None:
    # Given: an onsite ASR worker blocked until its provider resource is cancelled.


    transport = _Datagrams()

    listener = _DatagramListener(transport)

    asr = _DelayedAsr()

    runtime = TransportRuntime(
        _loopback_config(),
        datagram_listener=listener.listen,
        control_listener=_control_listener,
        onsite_bridge=_bridge(asr),
    )

    await runtime.start()

    assert listener.hub is not None

    listener.hub.register_control(_source_registration(), "127.0.0.1")

    listener.hub.register_control(_sink_registration(), "127.0.0.1")

    assert runtime.route_datagram(_rtp_packet(), ("127.0.0.1", 41_000)) is False

    _ = await asyncio.to_thread(asr.started.wait)

    # When: runtime shutdown invalidates the route before the blocking worker releases.

    await runtime.close()

    # Then: shutdown cancels the provider and its result cannot emit stale RTP.

    assert transport.sent == []

    _ = await asyncio.to_thread(asr.completed.wait)

    assert asr.cancelled.is_set()

    assert transport.sent == []


def _source_registration() -> str:

    return _registration(
        "media.rtp.source.register",
        "mic",
        {
            "stream_id": "stream-onsite-runtime",
            "ssrc": 0x10203040,
            "codec": _codec(),
            "rtp_endpoint": {"host": "127.0.0.1", "port": 5004},
        },
    )


def _sink_registration() -> str:

    return _registration(
        "media.rtp.sink.register",
        "sound",
        {
            "stream_id": "stream-onsite-runtime",
            "codec": _codec(),
            "rtp_endpoint": {"host": "127.0.0.1", "port": 5006},
        },
    )


def _registration(event_type: str, source: str, data: dict[str, JsonValue]) -> str:

    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": event_type,
            "event_id": event_type,
            "source": source,
            "time": "2026-07-28T00:00:00Z",
            "trace_id": "trace-onsite-runtime",
            "session_id": "session-onsite-runtime",
            "seq": 1,
            "data": data,
        }
    )


def _codec() -> dict[str, JsonValue]:

    return {
        "format": "L16",
        "clock_rate_hz": 16_000,
        "channels": 1,
        "payload_type": 96,
        "samples_per_frame": 320,
    }


def _rtp_packet() -> bytes:

    return b"\x80\x60\x00\x01\x00\x00\x00\x01\x10\x20\x30\x40" + (b"\x7f\xff" * 320)


def _bridge(asr: _DelayedAsr) -> OnsiteExplainerBridge:

    return OnsiteExplainerBridge(
        asr=asr,
        tts=_Tts(),
        pipeline_factory=lambda: OrchestratorTurnPipeline(
            adapters=PipelineAdapters(
                mode_policy=AdaptiveAgentPolicy(),
                llm=MockLLMAdapter(("onsite answer",)),
                retrieval=RetrievalFixtureProvider(()),
            ),
            config=PipelineConfig(1, "turn-onsite", "segment-onsite"),
        ),
        voice="raspberry",
        ref_audio="file:///voice.wav",
        ref_text="reference",
        frames_per_utterance=1,
        legacy_keyed_frames_per_utterance=1,
    )


def _wav(payload: bytes) -> bytes:

    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(2)

        audio.setframerate(16_000)

        audio.writeframes(payload)

    return output.getvalue()


def _loopback_config() -> TransportConfig:

    return TransportConfig(
        "127.0.0.1",
        8765,
        "127.0.0.1",
        5004,
        "127.0.0.1",
        8765,
        5004,
        "ws",
        None,
        None,
        None,
    )


async def _control_listener(
    _config: TransportConfig, _handler: ControlHandler
) -> _ControlServer:

    return _ControlServer()
