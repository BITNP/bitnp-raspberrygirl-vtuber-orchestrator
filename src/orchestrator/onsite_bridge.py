from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Final, cast

from orchestrator.llm import CancellationToken
from orchestrator.media_adapters import (
    AudioCppTTSAdapter,
    MediaAdapterConfigError,
    VllmOmniTTSAdapter,
)
from orchestrator.onsite_bridge_contracts import (
    OnsiteBridgeConfigError,
    OnsiteBridgeMediaError,
    TtsAdapter,
    l16_from_wav,
)
from orchestrator.openai_llm_runtime import AsyncOpenAICompatibleLLMRuntime
from orchestrator.provider_streaming import ProviderDeadlines
from orchestrator.response_coordinator import run_blocking_provider
from orchestrator.streaming_contracts import CancellationEpoch, SegmentId, StreamKey
from orchestrator.transport_hub import (
    L16_FRAME_BYTES,
    RTP_PAYLOAD_TYPE,
    RTP_V2_HEADER,
)
from orchestrator.tts_rtp import (
    Pcm16leChunk,
    StreamingTtsAdapter,
    TtsPcmRtpPacketizer,
    generated_ssrc,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from orchestrator.config import OrchestratorConfig
    from orchestrator.observability import OnsiteObservability


type OnsiteOutput = Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]]

type OnsiteOutputFinished = Callable[[StreamKey, CancellationEpoch], Awaitable[None]]

type OnsiteOutputAuthorization = Callable[[StreamKey, CancellationEpoch], bool]

type OnsiteResponseOutputPreparation = Callable[
    [StreamKey, CancellationEpoch, str], Awaitable[CancellationEpoch | None]
]

type OnsiteReplacement = Callable[
    [StreamKey, SegmentId], Awaitable[CancellationEpoch | None]
]

type ResponseOutputStarted = Callable[[], bool]


_STREAMING_TTS_STARTUP_PCM_BYTES: Final = 16_000 * 2
_STREAMING_TTS_CHUNK_QUEUE_CAPACITY: Final = 8


async def _discard_output(
    stream: StreamKey, epoch: CancellationEpoch, packet: bytes
) -> None:
    _ = (stream, epoch, packet)


async def _discard_finished(stream: StreamKey, epoch: CancellationEpoch) -> None:
    _ = (stream, epoch)


def _allow_output(stream: StreamKey, epoch: CancellationEpoch) -> bool:
    _ = (stream, epoch)

    return True


async def _prepare_reuse_requested_output_epoch(
    stream: StreamKey, epoch: CancellationEpoch, turn_id: str
) -> CancellationEpoch:
    _ = (stream, turn_id)
    return epoch


async def _discard_replacement(
    stream: StreamKey, segment_id: SegmentId
) -> CancellationEpoch | None:
    _ = (stream, segment_id)
    return None


__all__ = (
    "OnsiteBridgeConfigError",
    "OnsiteBridgeMediaError",
    "OnsiteExplainerBridge",
    "build_onsite_bridge",
    "generated_ssrc",
)


_RTP_HEADER_PREFIX = bytes((RTP_V2_HEADER, RTP_PAYLOAD_TYPE))

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OnsiteExplainerBridge:
    tts: TtsAdapter

    voice: str

    ref_audio: str

    ref_text: str

    llm: AsyncOpenAICompatibleLLMRuntime | None = None

    output: OnsiteOutput = field(default=_discard_output)

    output_finished: OnsiteOutputFinished = field(default=_discard_finished)

    begin_replacement: OnsiteReplacement = field(default=_discard_replacement)

    authorize_output: OnsiteOutputAuthorization = field(default=_allow_output)

    prepare_response_output: OnsiteResponseOutputPreparation = field(
        default=_prepare_reuse_requested_output_epoch
    )

    observability: OnsiteObservability | None = None

    def set_output_callback(self, callback: OnsiteOutput) -> None:
        self.output = callback

    def set_output_finished_callback(self, callback: OnsiteOutputFinished) -> None:
        self.output_finished = callback

    def set_output_authorizer(self, callback: OnsiteOutputAuthorization) -> None:
        """Install the transport's scheduler-owned output admission callback."""
        self.authorize_output = callback

    def set_response_output_preparer(
        self, callback: OnsiteResponseOutputPreparation
    ) -> None:
        """Install the async first-frame/cutover admission callback."""
        self.prepare_response_output = callback

    def set_replacement_callback(self, callback: OnsiteReplacement) -> None:
        self.begin_replacement = callback

    def set_observability(self, observability: OnsiteObservability) -> None:
        self.observability = observability

    def invalidate_stream(
        self, stream: StreamKey, next_epoch: CancellationEpoch
    ) -> None:
        _ = stream, next_epoch

    async def wait_quiescent(self) -> None:
        return

    async def aclose(self) -> None:
        if self.llm is not None:
            await self.llm.aclose()

    async def prepare_replacement(
        self, stream: StreamKey, segment_id: SegmentId
    ) -> CancellationEpoch | None:
        return await self.begin_replacement(stream, segment_id)

    def synthesize(
        self, text: str, cancellation: CancellationToken
    ) -> tuple[Pcm16leChunk, ...] | None:
        started_at = perf_counter()
        _LOGGER.debug("onsite_tts_request text=%r voice=%s", text, self.voice)
        for attempt in range(2):
            try:
                response = self.tts.synthesize(
                    text=text,
                    voice=self.voice,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                    cancellation=cancellation,
                )

                l16 = l16_from_wav(response)

                pcm16le = b"".join(
                    l16[offset : offset + 2][::-1] for offset in range(0, len(l16), 2)
                )
                chunks = (Pcm16leChunk(pcm16le),)
            except (MediaAdapterConfigError, OnsiteBridgeMediaError, wave.Error):
                _LOGGER.exception("onsite_tts_permanent_failure")
                return None
            except OSError:
                if cancellation.cancelled:
                    return None
                if attempt == 1:
                    _LOGGER.exception("onsite_tts_failed")
                    return None
                _LOGGER.warning("onsite_tts_retry attempt=%d", attempt + 1)
                continue
            _LOGGER.debug(
                "onsite_tts_complete pcm_bytes=%d latency_ms=%.1f",
                len(pcm16le),
                (perf_counter() - started_at) * 1_000,
            )
            return chunks
        return None

    async def speak_response(
        self,
        stream: StreamKey,
        text: str,
        epoch: CancellationEpoch,
        turn_id: str,
        output_started: ResponseOutputStarted,
    ) -> bool:
        """Synthesize one accepted Brain reply and deliver paced RTP to Sound.

        ``output_started`` commits the scheduler task only after the first
        valid RTP frame has crossed the output adapter.  This keeps context and
        caption effects behind the same first-frame boundary as playback.
        """
        cancellation = CancellationToken()
        try:
            stream_synthesis = await run_blocking_provider(
                self.stream_synthesize, text, cancellation
            )
            if stream_synthesis is not None:
                return await self._speak_streaming_response(
                    stream,
                    epoch,
                    turn_id,
                    stream_synthesis,
                    cancellation,
                    output_started,
                )
            chunks = await run_blocking_provider(self.synthesize, text, cancellation)
            output_epoch = await self.prepare_response_output(stream, epoch, turn_id)
            if not chunks or output_epoch is None:
                return False
            packetizer = TtsPcmRtpPacketizer(
                stream=stream, cancellation_epoch=output_epoch
            )
            packets = tuple(
                packet for chunk in chunks for packet in packetizer.push(chunk)
            )
            packets += packetizer.finish()
            if not packets:
                return False
            loop = asyncio.get_running_loop()
            deadline = loop.time()
            for index, packet in enumerate(packets):
                await self.output(stream, output_epoch, packet)
                if index == 0 and not output_started():
                    return False
                deadline += 0.02
                delay = deadline - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            await self.output_finished(stream, output_epoch)
        except asyncio.CancelledError:
            _ = cancellation.cancel(reason="response_tts_cancelled")
            raise
        else:
            return True

    async def _speak_streaming_response(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
        self,
        stream: StreamKey,
        epoch: CancellationEpoch,
        turn_id: str,
        synthesis: Iterator[Pcm16leChunk],
        cancellation: CancellationToken,
        output_started: ResponseOutputStarted,
    ) -> bool:
        """Start RTP after one second of PCM is ready, or when a short stream ends."""
        packetizer: TtsPcmRtpPacketizer | None = None
        output_epoch: CancellationEpoch | None = None
        committed = False
        loop = asyncio.get_running_loop()
        deadline: float | None = None

        async def emit(packets: tuple[bytes, ...]) -> bool:
            nonlocal committed, deadline
            if not packets:
                return True
            if output_epoch is None:
                return False
            now = loop.time()
            # TTS startup buffering and later provider stalls are not RTP media
            # time.  Never repay either wait by bursting delayed packets.
            deadline = now if deadline is None else max(deadline, now)
            for _index, packet in enumerate(packets):
                await self.output(stream, output_epoch, packet)
                if not committed:
                    committed = output_started()
                    if not committed:
                        return False
                deadline += 0.02
                delay = deadline - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            return True

        buffered = bytearray()
        received_chunks: asyncio.Queue[Pcm16leChunk | Exception | None] = asyncio.Queue(
            maxsize=_STREAMING_TTS_CHUNK_QUEUE_CAPACITY
        )

        async def receive_chunks() -> None:
            pending_next: asyncio.Task[Pcm16leChunk | None] | None = None
            chunk_index = 0
            previous_arrival = loop.time()
            try:
                while not cancellation.cancelled:
                    pending_next = asyncio.create_task(
                        run_blocking_provider(next, synthesis, None)
                    )
                    try:
                        chunk = await asyncio.shield(pending_next)
                    except asyncio.CancelledError:
                        if not pending_next.done():
                            with suppress(asyncio.CancelledError, OSError, ValueError):
                                _ = await asyncio.shield(pending_next)
                        raise
                    pending_next = None
                    if chunk is None:
                        await received_chunks.put(None)
                        return
                    now = loop.time()
                    _LOGGER.debug(
                        "onsite_tts_chunk_received session=%s stream=%s turn=%s chunk=%d pcm_bytes=%d gap_ms=%.1f",  # noqa: E501
                        stream.session_id,
                        stream.stream_id,
                        turn_id,
                        chunk_index,
                        len(chunk.data),
                        (now - previous_arrival) * 1_000,
                    )
                    previous_arrival = now
                    chunk_index += 1
                    await received_chunks.put(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                await received_chunks.put(error)

        receiver_task = asyncio.create_task(receive_chunks())
        try:
            while not cancellation.cancelled:
                item = await received_chunks.get()
                if isinstance(item, Exception):
                    raise item
                if item is None:
                    break
                chunk = item
                buffered.extend(chunk.data)
                if packetizer is None:
                    if len(buffered) < _STREAMING_TTS_STARTUP_PCM_BYTES:
                        continue
                    output_epoch = await self.prepare_response_output(
                        stream, epoch, turn_id
                    )
                    if output_epoch is None:
                        return False
                    packetizer = TtsPcmRtpPacketizer(stream, output_epoch)
                elif len(buffered) < L16_FRAME_BYTES:
                    continue
                packets = packetizer.push(Pcm16leChunk(bytes(buffered)))
                buffered.clear()
                if not await emit(packets):
                    return False
            if cancellation.cancelled:
                return False
            await receiver_task
            if packetizer is None:
                if not buffered:
                    return False
                output_epoch = await self.prepare_response_output(
                    stream, epoch, turn_id
                )
                if output_epoch is None:
                    return False
                packetizer = TtsPcmRtpPacketizer(stream, output_epoch)
            packets = (
                packetizer.push(Pcm16leChunk(bytes(buffered))) + packetizer.finish()
            )
            if not await emit(packets):
                return False
        except asyncio.CancelledError:
            _ = cancellation.cancel(reason="response_tts_cancelled")
            raise
        finally:
            if not receiver_task.done():
                _ = cancellation.cancel(reason="response_tts_stopped")
                _ = receiver_task.cancel()
            with suppress(asyncio.CancelledError, OSError, ValueError):
                _ = await asyncio.shield(receiver_task)
            close = cast(
                "Callable[[], object] | None", getattr(synthesis, "close", None)
            )
            if close is not None:
                _ = await run_blocking_provider(close)
        if not committed or output_epoch is None:
            return False
        await self.output_finished(stream, output_epoch)
        return True

    def stream_synthesize(
        self, text: str, cancellation: CancellationToken
    ) -> Iterator[Pcm16leChunk] | None:
        if getattr(
            self.tts, "capability", "final_only"
        ) != "streaming_sse" or not isinstance(self.tts, StreamingTtsAdapter):
            return None
        return self.tts.stream_pcm16le(
            text=text,
            voice=self.voice,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            cancellation=cancellation,
        )


def build_onsite_bridge(
    config: OrchestratorConfig,
    *,
    voice: str,
    ref_audio: str,
    ref_text: str,
) -> OnsiteExplainerBridge:
    if config.tts_provider not in {"vllm_omni", "audio_cpp"}:
        raise OnsiteBridgeConfigError(field_name="tts_provider")

    if config.tts_endpoint is None or config.tts_model is None:
        raise OnsiteBridgeConfigError(field_name="tts_endpoint_or_tts_model")

    if (
        config.llm_provider != "openai_compatible"
        or config.llm_endpoint is None
        or config.llm_model is None
        or config.llm_api_key is None
        or config.llm_reasoning_dialect is None
    ):
        raise OnsiteBridgeConfigError(field_name="llm_provider_or_llm_configuration")

    if config.tts_provider == "vllm_omni" and (
        voice.strip() == "" or ref_audio.strip() == "" or ref_text.strip() == ""
    ):
        raise OnsiteBridgeConfigError(field_name="voice_reference")

    llm = AsyncOpenAICompatibleLLMRuntime(
        config.llm_endpoint,
        config.llm_model,
        config.llm_api_key,
        config.llm_reasoning_dialect,
        brain_model=config.llm_brain_model,
        maintenance_model=config.llm_maintenance_model,
        timeout_seconds=120.0,
        deadlines=ProviderDeadlines(read_seconds=60.0, total_seconds=120.0),
        ca_path=config.tls_ca_path,
    )

    return OnsiteExplainerBridge(
        tts=(
            AudioCppTTSAdapter
            if config.tts_provider == "audio_cpp"
            else VllmOmniTTSAdapter
        )(
            config.tts_endpoint,
            config.tts_model,
            config.tts_api_key,
            capability=config.tts_mode,
            ca_path=config.tls_ca_path,
        ),
        voice=voice,
        ref_audio=ref_audio,
        ref_text=ref_text,
        llm=llm,
    )
