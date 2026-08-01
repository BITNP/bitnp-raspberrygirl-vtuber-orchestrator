
from __future__ import annotations

import asyncio
import io
import json
import threading
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar, Final, Literal, cast, override

from sound.receive import ReceiveRuntime
from sound.receive_config import SoundReceiveConfig

if TYPE_CHECKING:
    from pathlib import Path

from orchestrator.media_adapters import OpenAICompatibleASRAdapter, VllmOmniTTSAdapter
from orchestrator.modes import AdaptiveAgentPolicy
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.openai_llm_runtime import OpenAICompatibleLLMRuntimeAdapter
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import PipelineConfig
from orchestrator.retrieval import RetrievalFixtureProvider

if TYPE_CHECKING:
    import ssl
    from collections.abc import Callable

    from sound.rtp_playback import L16PlaybackFrame


_Mode = Literal["success", "asr_failure", "qwen_24k"]

_SAMPLE_RATE: Final = 16_000


@dataclass(slots=True)
class _FakeProvider:

    mode: _Mode

    _server: ThreadingHTTPServer = field(init=False)

    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:

        _ProviderHandler.mode = self.mode

        _ProviderHandler.speech_bodies = []

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> str:

        self._thread.start()

        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:

        self._server.shutdown()

        self._thread.join(timeout=1.0)

        self._server.server_close()


class _ProviderHandler(BaseHTTPRequestHandler):

    mode: ClassVar[_Mode]

    speech_bodies: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:

        body = self.rfile.read(int(self.headers["content-length"]))

        if self.path.endswith("/audio/transcriptions") and self.mode == "asr_failure":
            self.send_response(503)

            self.end_headers()

            return

        match self.path:
            case path if path.endswith("/audio/transcriptions"):
                self._json(b'{"text":"Explain BitNet"}')

            case path if path.endswith("/chat/completions"):
                self._sse(
                    (
                        'data: {"choices":[{"delta":{"content":"provider "}}]}\n\n',
                        'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n',
                        "data: [DONE]\n\n",
                    )
                )

            case path if path.endswith("/audio/speech"):
                self.speech_bodies.append(body)

                self._wav()

            case _:
                self.send_response(404)

                self.end_headers()

    def _json(self, body: bytes) -> None:

        self.send_response(200)

        self.send_header("content-type", "application/json")

        self.send_header("content-length", str(len(body)))

        self.end_headers()

        _ = self.wfile.write(body)

    def _sse(self, events: tuple[str, ...]) -> None:

        self.send_response(200)

        self.send_header("content-type", "text/event-stream")

        self.end_headers()

        for event in events:
            _ = self.wfile.write(event.encode())

            self.wfile.flush()

    def _wav(self) -> None:

        if self.mode == "qwen_24k":
            data = _wav(b"\x10\x20" * 480, sample_rate=24_000)

        else:
            data = _wav(b"\x10\x20" * 320)

        self.send_response(200)

        self.send_header("content-type", "audio/wav")

        self.send_header("content-length", str(len(data)))

        self.end_headers()

        _ = self.wfile.write(data)

    @override
    def log_message(self, format: str, *args: object) -> None:

        _ = (format, args)


@dataclass(slots=True)
class _Binding:

    handler: Callable[[bytes], None] | None = None

    @property
    def port(self) -> int:

        return 50_006

    def set_packet_handler(self, handler: Callable[[bytes], None]) -> None:

        self.handler = handler

    def deliver(self, packet: bytes) -> None:

        assert self.handler is not None

        self.handler(packet)

    def close(self) -> None:

        return


@dataclass(frozen=True, slots=True)
class _Binder:

    binding: _Binding

    async def bind(self, host: str, port: int) -> _Binding:

        assert (host, port) == ("127.0.0.1", 50_006)

        return self.binding


@dataclass(slots=True)
class _Control:

    command: str

    packet: bytes

    binding: _Binding

    sent: list[str] = field(default_factory=list)

    delivered: bool = False

    async def send(self, message: str) -> None:

        self.sent.append(message)

    async def recv(self) -> str | None:

        if not self.delivered:
            self.delivered = True

            return self.command

        self.binding.deliver(self.packet)

        return None

    async def close(self) -> None:

        return


@dataclass(frozen=True, slots=True)
class _Connector:

    control: _Control

    async def connect(
        self, url: str, headers: dict[str, str], ssl_context: ssl.SSLContext | None
    ) -> _Control:

        assert url == "wss://orchestrator.example.test/control"

        assert headers == {}

        assert ssl_context is None

        return self.control


@dataclass(slots=True)
class _Playback:

    frames: list[L16PlaybackFrame] = field(default_factory=list)

    def write(self, frame: L16PlaybackFrame) -> None:

        self.frames.append(frame)

    def close_stream(self, stream_id: str) -> None:

        _ = stream_id

    def close(self) -> None:

        return


def test_fake_local_provider_chain_generates_sound_queued_then_playing() -> None:

    asyncio.run(_assert_full_chain())


async def _assert_full_chain() -> None:
    # Given: fake-local providers and a real Sound runtime.


    with _FakeProvider("success") as endpoint:
        bridge = _bridge(endpoint)

        mic_packet = _rtp(b"\x01\x02" * 320)

        generated = await bridge.ingest_mic_rtp(mic_packet)

    assert isinstance(generated, tuple)

    binding = _Binding()

    control = _Control(_command(generated[0]), generated[0], binding)

    playback = _Playback()

    runtime = ReceiveRuntime(
        config=SoundReceiveConfig(
            orchestrator_ws_url="wss://orchestrator.example.test/control",
            trusted_lan_token=None,
            stream_id="onsite-answer",
            rtp_host="127.0.0.1",
            rtp_port=50_006,
            advertised_rtp_host="sound.example.test",
        ),
        udp_binder=_Binder(binding),
        control_connector=_Connector(control),
        playback_sink=playback,
    )

    # When: the generated RTP is announced and delivered through Sound's control path.

    await runtime.run()

    await asyncio.sleep(0)

    # Then: Sound reports queued and playing for generated, non-Mic L16 RTP.

    state_messages = [
        message
        for message in control.sent
        if '"event_type":"media.stream.state"' in message
    ]

    assert len(state_messages) == 2

    assert '"state":"queued"' in state_messages[0]

    assert '"state":"playing"' in state_messages[1]

    assert len(playback.frames) == 1

    assert generated[0] != mic_packet

    assert generated[0][12:] == b"\x20\x10" * 320


def test_fake_local_provider_non_success_drops_mic_without_raw_fallback() -> None:
    # Given: a fake-local ASR provider that returns a non-2xx response.


    with _FakeProvider("asr_failure") as endpoint:
        bridge = _bridge(endpoint)

        # When: Mic RTP reaches the provider chain.

        generated = asyncio.run(bridge.ingest_mic_rtp(_rtp(b"\x01\x02" * 320)))

    # Then: the chain fails closed and never returns the Mic RTP as fallback media.

    assert generated is None


def test_qwen_shaped_tts_response_becomes_canonical_generated_rtp(
    tmp_path: Path,
) -> None:
    # Given: local Qwen requires portable reference audio and emits 24 kHz WAV.


    reference = tmp_path / "voice.wav"
    _ = reference.write_bytes(b"RIFFvoice")

    with _FakeProvider("qwen_24k") as endpoint:
        bridge = _bridge(endpoint, ref_audio=str(reference))

        mic_packet = _rtp(b"\x01\x02" * 320)

        # When: Mic RTP reaches the real onsite ASR, LLM, TTS, and RTP stages.

        generated = asyncio.run(bridge.ingest_mic_rtp(mic_packet))

        speech_bodies = tuple(_ProviderHandler.speech_bodies)

    # Then: the request is Qwen-compatible and Sound-facing RTP remains canonical L16.

    assert isinstance(generated, tuple)
    assert len(speech_bodies) == 1
    payload = cast("dict[str, str]", json.loads(speech_bodies[0]))
    assert payload["ref_audio"].startswith("data:audio/wav;base64,")
    assert generated[0][0:2] == b"\x80\x60"
    assert len(generated[0][12:]) == 640
    assert generated[0][12:] == b"\x20\x10" * 320
    assert generated[0] != mic_packet


def _bridge(
    endpoint: str, *, ref_audio: str = "data:audio/wav;base64,UklGRg=="
) -> OnsiteExplainerBridge:

    adapters = PipelineAdapters(
        mode_policy=AdaptiveAgentPolicy(),
        llm=OpenAICompatibleLLMRuntimeAdapter(
            endpoint=endpoint,
            model="local-llm",
            api_key="fake-local-key",
            capability="streaming",
        ),
        retrieval=RetrievalFixtureProvider(()),
    )

    return OnsiteExplainerBridge(
        asr=OpenAICompatibleASRAdapter(endpoint=endpoint, model="local-asr"),
        tts=VllmOmniTTSAdapter(endpoint=endpoint, model="local-tts"),
        pipeline_factory=lambda: OrchestratorTurnPipeline(
            adapters=adapters,
            config=PipelineConfig(1, "turn-onsite", "segment-onsite"),
        ),
        voice="raspberry",
        ref_audio=ref_audio,
        ref_text="reference",
        frames_per_utterance=1,
    )


def _command(packet: bytes) -> str:

    return json.dumps(
        {
            "schema_version": "1.0.0",
            "event_type": "media.stream.command",
            "event_id": "onsite-acceptance-command",
            "source": "orchestrator",
            "time": "2026-07-28T00:00:00Z",
            "trace_id": "trace-onsite-acceptance",
            "session_id": "session-onsite-acceptance",
            "seq": 1,
            "data": {
                "command_id": "onsite-acceptance-command",
                "stream_id": "onsite-answer",
                "start_rtp_timestamp": 96_000,
                "ssrc": int.from_bytes(packet[8:12]),
                "codec": {
                    "format": "L16",
                    "clock_rate_hz": _SAMPLE_RATE,
                    "channels": 1,
                    "payload_type": 96,
                    "samples_per_frame": 320,
                },
                "rtp_endpoint": {"host": "sound.example.test", "port": 50_006},
            },
        }
    )


def _rtp(payload: bytes) -> bytes:

    return b"\x80\x60\x00\x01\x00\x00\x00\x01\x10\x20\x30\x40" + payload


def _wav(payload: bytes, *, sample_rate: int = _SAMPLE_RATE) -> bytes:

    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(2)

        audio.setframerate(sample_rate)

        audio.writeframes(payload)

    return output.getvalue()
