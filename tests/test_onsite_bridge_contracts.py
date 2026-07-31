
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


def test_l16_from_wav_converts_qwen_24khz_pcm16_to_16khz_l16() -> None:
    # Given: Qwen TTS returns mono PCM16 WAV at 24 kHz.


    response = SynthesizedAudio(
        _wav(
            b"\x01\x00"
            b"\x02\x00"
            b"\x03\x00"
            b"\x04\x00"
            b"\x05\x00"
            b"\x06\x00",
            sample_rate=24_000,
        ),
        "audio/wav",
    )

    # When: the response crosses into canonical RTP L16 payload bytes.

    network_l16 = l16_from_wav(response)

    # Then: complete 24 kHz sample triplets become deterministic 16 kHz samples.

    assert network_l16 == b"\x00\x01\x00\x02\x00\x04\x00\x05"


def test_l16_from_wav_rejects_unsupported_sample_rate() -> None:
    # Given: a provider returns mono PCM16 WAV at an unsupported rate.


    response = SynthesizedAudio(_wav(b"\x01\x00", sample_rate=22_050), "audio/wav")

    # When / Then: Orchestrator refuses media it cannot make canonical.

    try:
        _ = l16_from_wav(response)
    except ValueError as error:
        assert str(error) == "onsite TTS WAV must become 16 kHz mono PCM16"

    else:
        raise AssertionError("unsupported TTS sample rate was accepted")


def _wav(payload: bytes, *, sample_rate: int = 16_000) -> bytes:

    output = io.BytesIO()

    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)

        audio.setsampwidth(2)

        audio.setframerate(sample_rate)

        audio.writeframes(payload)

    return output.getvalue()


def _wav_payload(data: bytes) -> bytes:

    with wave.open(io.BytesIO(data), "rb") as audio:
        return audio.readframes(audio.getnframes())
