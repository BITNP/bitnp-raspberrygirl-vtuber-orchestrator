
from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, override

from orchestrator.transport_hub import L16_FRAME_BYTES

if TYPE_CHECKING:
    from orchestrator.media_adapters import SynthesizedAudio
    from orchestrator.pipeline_contracts import ASRAudienceEvent
    from orchestrator.provider_streaming import ProviderCancellationHandle


_SAMPLE_RATE: Final = 16_000

_PCM16_BYTES: Final = 2


@dataclass(frozen=True, slots=True)
class OnsiteBridgeConfigError(ValueError):

    field_name: str

    @override
    def __str__(self) -> str:
        return f"onsite bridge configuration is incomplete: {self.field_name}"


@dataclass(frozen=True, slots=True)
class OnsiteBridgeMediaError(ValueError):

    reason: Literal["media_type", "wave_format"]

    @override
    def __str__(self) -> str:
        match self.reason:
            case "media_type":
                return "onsite TTS must return audio/wav"

            case "wave_format":
                return "onsite TTS WAV must be 16 kHz mono PCM16"


class AsrAdapter(Protocol):

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
        ...


class TtsAdapter(Protocol):

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        ref_audio: str,
        ref_text: str,
        cancellation: ProviderCancellationHandle | None = None,
    ) -> SynthesizedAudio:
        ...


def wav_from_l16(payload: bytes) -> bytes:
    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(_PCM16_BYTES)

        audio.setframerate(_SAMPLE_RATE)

        audio.writeframes(_swap_pcm16_byte_order(payload))

    return output.getvalue()


def pcm16le_from_l16(payload: bytes) -> bytes:
    return _swap_pcm16_byte_order(payload)


def l16_from_wav(response: SynthesizedAudio) -> bytes:
    if response.media_type != "audio/wav":
        raise OnsiteBridgeMediaError(reason="media_type")

    with wave.open(io.BytesIO(response.data), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != _PCM16_BYTES
            or audio.getframerate() != _SAMPLE_RATE
            or audio.getcomptype() != "NONE"
        ):
            raise OnsiteBridgeMediaError(reason="wave_format")

        return _swap_pcm16_byte_order(audio.readframes(audio.getnframes()))


def pad_l16_frames(audio: bytes) -> bytes:
    remainder = len(audio) % L16_FRAME_BYTES

    if remainder == 0:
        return audio

    return audio + bytes(L16_FRAME_BYTES - remainder)


def generated_ssrc(mic_ssrc: int) -> int:
    generated = mic_ssrc ^ 0xA5A5_A5A5

    return 1 if generated == 0 else generated


def _swap_pcm16_byte_order(payload: bytes) -> bytes:
    samples = range(0, len(payload), _PCM16_BYTES)

    return b"".join(payload[offset : offset + _PCM16_BYTES][::-1] for offset in samples)
