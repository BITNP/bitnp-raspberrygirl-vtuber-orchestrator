from __future__ import annotations

import io
import wave

from orchestrator.media_adapters import SynthesizedAudio
from orchestrator.onsite_bridge_contracts import l16_from_wav, wav_from_l16


def test_wav_from_l16_converts_network_samples_to_pcm16_little_endian() -> None:
    # Given: non-symmetric signed PCM samples in canonical RTP L16 network order.
    network_l16 = b"\x12\x34\xab\xcd"

    # When: the payload crosses into the WAV/ASR boundary.
    wav = wav_from_l16(network_l16)

    # Then: WAV PCM16 bytes use the required little-endian sample representation.
    assert _wav_payload(wav) == b"\x34\x12\xcd\xab"


def test_l16_from_wav_converts_pcm16_little_endian_to_network_samples() -> None:
    # Given: non-symmetric PCM16 little-endian samples from the TTS/WAV boundary.
    response = SynthesizedAudio(_wav(b"\x34\x12\xcd\xab"), "audio/wav")

    # When: the response crosses into canonical RTP L16 payload bytes.
    network_l16 = l16_from_wav(response)

    # Then: outgoing RTP receives big-endian network-order PCM samples.
    assert network_l16 == b"\x12\x34\xab\xcd"


def _wav(payload: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(payload)
    return output.getvalue()


def _wav_payload(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as audio:
        return audio.readframes(audio.getnframes())
