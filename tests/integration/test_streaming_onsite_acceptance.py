"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar, Final, Literal, override

from sound.receive import ReceiveRuntime
from sound.receive_config import SoundReceiveConfig

from orchestrator.media_adapters import OpenAICompatibleASRAdapter, VllmOmniTTSAdapter
from orchestrator.modes import ModePolicy
from orchestrator.onsite_bridge import OnsiteExplainerBridge
from orchestrator.openai_llm_runtime import OpenAICompatibleLLMRuntimeAdapter
from orchestrator.pipeline import OrchestratorTurnPipeline, PipelineAdapters
from orchestrator.pipeline_contracts import PipelineConfig
from orchestrator.retrieval import RetrievalFixtureProvider

if TYPE_CHECKING:
    from collections.abc import Callable

    from sound.rtp_playback import L16PlaybackFrame


_Mode = Literal["success", "asr_failure"]

_SAMPLE_RATE: Final = 16_000


@dataclass(slots=True)
class _FakeProvider:
    """类契约说明.

    职责: 保存 _FakeProvider
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: mode、_server、_thread。 方法:
    __post_init__、__enter__、__exit__。
    """

    mode: _Mode

    _server: ThreadingHTTPServer = field(init=False)

    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        """函数契约说明.

        功能: 初始化 _FakeProvider
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        _ProviderHandler.mode = self.mode

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        """函数契约说明.

        功能: 执行 __enter__ 的同步逻辑,并协调
        start。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """

        self._thread.start()

        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """函数契约说明.

        功能: 执行 __exit__ 的同步逻辑,并协调
        shutdown, join, server_close。
        参数: self 表示当前实例。 exc_type:
        object。 必填。 exc: object。 必填。
        traceback: object。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self._server.shutdown()

        self._thread.join(timeout=1.0)

        self._server.server_close()


class _ProviderHandler(BaseHTTPRequestHandler):
    """类契约说明.

    职责: 定义 _ProviderHandler
    的状态、行为和对外协作边界。
    契约: 字段: mode。 方法:
    do_POST、_json、_sse、_wav、log_message。
    """

    mode: ClassVar[_Mode]

    def do_POST(self) -> None:
        """函数契约说明.

        功能: 执行 do_POST 的同步逻辑,并协调 read,
        int, endswith, send_response。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        _ = self.rfile.read(int(self.headers["content-length"]))

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
                self._wav()

            case _:
                self.send_response(404)

                self.end_headers()

    def _json(self, body: bytes) -> None:
        """函数契约说明.

        功能: 执行 _json 的同步逻辑,并协调
        send_response, send_header,
        end_headers, write。
        参数: self 表示当前实例。 body: bytes。
        必填。
        契约: 同步调用。 返回 `None`。
        """

        self.send_response(200)

        self.send_header("content-type", "application/json")

        self.send_header("content-length", str(len(body)))

        self.end_headers()

        _ = self.wfile.write(body)

    def _sse(self, events: tuple[str, ...]) -> None:
        """函数契约说明.

        功能: 执行 _sse 的同步逻辑,并协调
        send_response, send_header,
        end_headers, write。
        参数: self 表示当前实例。 events:
        tuple[str, ...]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.send_response(200)

        self.send_header("content-type", "text/event-stream")

        self.end_headers()

        for event in events:
            _ = self.wfile.write(event.encode())

            self.wfile.flush()

    def _wav(self) -> None:
        """函数契约说明.

        功能: 执行 _wav 的同步逻辑,并协调 _wav,
        send_response, send_header,
        end_headers。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        data = _wav(b"\x10\x20" * 320)

        self.send_response(200)

        self.send_header("content-type", "audio/wav")

        self.send_header("content-length", str(len(data)))

        self.end_headers()

        _ = self.wfile.write(data)

    @override
    def log_message(self, format: str, *args: object) -> None:
        """函数契约说明.

        功能: 执行 log_message 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 format: str。
        必填。 *args: object。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = (format, args)


@dataclass(slots=True)
class _Binding:
    """类契约说明.

    职责: 保存 _Binding 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: handler。 方法: port、set_packet
    _handler、deliver、close。
    """

    handler: Callable[[bytes], None] | None = None

    @property
    def port(self) -> int:
        """函数契约说明.

        功能: 执行 port 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `int`。
        """

        return 50_006

    def set_packet_handler(self, handler: Callable[[bytes], None]) -> None:
        """函数契约说明.

        功能: 执行 set_packet_handler
        的同步逻辑,并产出 handler。
        参数: self 表示当前实例。 handler:
        Callable[[bytes], None]。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.handler = handler

    def deliver(self, packet: bytes) -> None:
        """函数契约说明.

        功能: 执行 deliver 的同步逻辑,并协调
        handler。
        参数: self 表示当前实例。 packet: bytes。
        必填。
        契约: 同步调用。 返回 `None`。
        """

        assert self.handler is not None

        self.handler(packet)

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return


@dataclass(frozen=True, slots=True)
class _Binder:
    """类契约说明.

    职责: 保存 _Binder 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: binding。 方法: bind。
    """

    binding: _Binding

    async def bind(self, host: str, port: int) -> _Binding:
        """函数契约说明.

        功能: 执行 bind 的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。 host: str。 必填。
        port: int。 必填。
        契约: 异步调用。 返回 `_Binding`。
        """

        assert (host, port) == ("127.0.0.1", 50_006)

        return self.binding


@dataclass(slots=True)
class _Control:
    """类契约说明.

    职责: 保存 _Control 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: command、packet、binding、sent、
    delivered。 方法: send、recv、close。
    """

    command: str

    packet: bytes

    binding: _Binding

    sent: list[str] = field(default_factory=list)

    delivered: bool = False

    async def send(self, message: str) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 message: str。
        必填。
        契约: 异步调用。 返回 `None`。
        """

        self.sent.append(message)

    async def recv(self) -> str | None:
        """函数契约说明.

        功能: 执行 recv 的异步逻辑,并协调 deliver。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `str | None`。
        """

        if not self.delivered:
            self.delivered = True

            return self.command

        self.binding.deliver(self.packet)

        return None

    async def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """

        return


@dataclass(frozen=True, slots=True)
class _Connector:
    """类契约说明.

    职责: 保存 _Connector
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: control。 方法: connect。
    """

    control: _Control

    async def connect(self, url: str, headers: dict[str, str]) -> _Control:
        """函数契约说明.

        功能: 执行 connect 的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。 url: str。 必填。
        headers: dict[str, str]。 必填。
        契约: 异步调用。 返回 `_Control`。
        """

        assert url == "wss://orchestrator.example.test/control"

        assert headers == {}

        return self.control


@dataclass(slots=True)
class _Playback:
    """类契约说明.

    职责: 保存 _Playback
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: frames。 方法:
    write、close_stream、close。
    """

    frames: list[L16PlaybackFrame] = field(default_factory=list)

    def write(self, frame: L16PlaybackFrame) -> None:
        """函数契约说明.

        功能: 执行 write 的同步逻辑,并协调 append。
        参数: self 表示当前实例。 frame:
        L16PlaybackFrame。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.frames.append(frame)

    def close_stream(self, stream_id: str) -> None:
        """函数契约说明.

        功能: 执行 close_stream 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 stream_id: str。
        必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = stream_id

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        return


def test_fake_local_provider_chain_generates_sound_queued_then_playing() -> None:
    """函数契约说明.

    功能: 验证 fake local provider chain
    generates sound queued then playing
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    asyncio.run(_assert_full_chain())


async def _assert_full_chain() -> None:
    # Given: fake-local providers and a real Sound runtime.

    """函数契约说明.

    功能: 执行 _assert_full_chain 的异步逻辑,并协调
    isinstance, _Binding, _Control,
    _Playback。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 fake local provider non
    success drops mic without raw
    fallback 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    with _FakeProvider("asr_failure") as endpoint:
        bridge = _bridge(endpoint)

        # When: Mic RTP reaches the provider chain.

        generated = asyncio.run(bridge.ingest_mic_rtp(_rtp(b"\x01\x02" * 320)))

    # Then: the chain fails closed and never returns the Mic RTP as fallback media.

    assert generated is None


def _bridge(endpoint: str) -> OnsiteExplainerBridge:
    """函数契约说明.

    功能: 执行 _bridge 的同步逻辑,并协调
    PipelineAdapters,
    OnsiteExplainerBridge,
    onsite_explainer,
    OpenAICompatibleLLMRuntimeAdapter。
    参数: endpoint: str。 必填。
    契约: 同步调用。 返回
    `OnsiteExplainerBridge`。
    """

    adapters = PipelineAdapters(
        mode_policy=ModePolicy.onsite_explainer(),
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
        ref_audio="file:///voice.wav",
        ref_text="reference",
        frames_per_utterance=1,
    )


def _command(packet: bytes) -> str:
    """函数契约说明.

    功能: 执行 _command 的同步逻辑,并协调 dumps,
    from_bytes。
    参数: packet: bytes。 必填。
    契约: 同步调用。 返回 `str`。
    """

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
    """函数契约说明.

    功能: 执行 _rtp 的同步逻辑,并维持签名契约。
    参数: payload: bytes。 必填。
    契约: 同步调用。 返回 `bytes`。
    """

    return b"\x80\x60\x00\x01\x00\x00\x00\x01\x10\x20\x30\x40" + payload


def _wav(payload: bytes) -> bytes:
    """函数契约说明.

    功能: 执行 _wav 的同步逻辑,并协调 BytesIO,
    getvalue, open, setnchannels。
    参数: payload: bytes。 必填。
    契约: 同步调用。 返回 `bytes`。
    """

    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(2)

        audio.setframerate(_SAMPLE_RATE)

        audio.writeframes(payload)

    return output.getvalue()
