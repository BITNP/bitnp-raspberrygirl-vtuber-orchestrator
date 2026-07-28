"""Typed media and adapter contracts for the onsite explainer bridge."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, override

from orchestrator.transport_hub import L16_FRAME_BYTES

if TYPE_CHECKING:
    from orchestrator.media_adapters import SynthesizedAudio
    from orchestrator.pipeline_contracts import ASRAudienceEvent

_SAMPLE_RATE: Final = 16_000
_PCM16_BYTES: Final = 2


@dataclass(frozen=True, slots=True)
class OnsiteBridgeConfigError(ValueError):
    """Raised for incomplete onsite provider or voice-reference configuration."""

    field_name: str

    @override
    def __str__(self) -> str:
        return f"onsite bridge configuration is incomplete: {self.field_name}"


@dataclass(frozen=True, slots=True)
class OnsiteBridgeMediaError(ValueError):
    """Raised when synthesized audio violates the fixed onsite L16 contract."""

    reason: Literal["media_type", "wave_format"]

    @override
    def __str__(self) -> str:
        match self.reason:
            case "media_type":
                return "onsite TTS must return audio/wav"
            case "wave_format":
                return "onsite TTS WAV must be 16 kHz mono PCM16"


class AsrAdapter(Protocol):
    """Transcribes one deterministic L16 utterance."""

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        received_at_ms: int,
        segment_id: str,
        seq: int,
    ) -> ASRAudienceEvent | None:
        """Return a normalized ASR final event."""
        ...


class TtsAdapter(Protocol):
    """Synthesizes a completed onsite answer."""

    def synthesize(
        self, *, text: str, voice: str, ref_audio: str, ref_text: str
    ) -> SynthesizedAudio:
        """Return a WAV response under the fixed onsite media contract."""
        ...


def wav_from_l16(payload: bytes) -> bytes:
    """Wrap canonical L16 payload bytes in a mono PCM16 WAV container."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(_PCM16_BYTES)
        audio.setframerate(_SAMPLE_RATE)
        audio.writeframes(_swap_pcm16_byte_order(payload))
    return output.getvalue()


def l16_from_wav(response: SynthesizedAudio) -> bytes:
    """Validate an onsite WAV response and return its canonical L16 payload."""
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
    """Pad one L16 payload to an integral number of RTP frame payloads."""
    remainder = len(audio) % L16_FRAME_BYTES
    if remainder == 0:
        return audio
    return audio + bytes(L16_FRAME_BYTES - remainder)


def generated_ssrc(mic_ssrc: int) -> int:
    """Derive a deterministic nonzero SSRC that cannot equal the Mic SSRC."""
    generated = mic_ssrc ^ 0xA5A5_A5A5
    return 1 if generated == 0 else generated


def _swap_pcm16_byte_order(payload: bytes) -> bytes:
    samples = range(0, len(payload), _PCM16_BYTES)
    return b"".join(payload[offset : offset + _PCM16_BYTES][::-1] for offset in samples)
