from __future__ import annotations

import logging

from orchestrator.log_summary import (
    binary_summary,
    reference_audio_summary,
)
from orchestrator.transport_app import configure_dependency_loggers


def test_binary_summary_limits_preview_to_thirty_bytes() -> None:
    summary = binary_summary(bytes(range(31)))

    assert "bytes=31" in summary
    assert (
        "hex=000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d…"
        in summary
    )


def test_reference_audio_summary_never_contains_base64_payload() -> None:
    audio = "data:audio/wav;base64," + ("private-audio" * 100)

    summary = reference_audio_summary(audio)

    assert summary == (
        "kind=data_url media_type='audio/wav' encoding=base64 payload_chars=1300"
    )
    assert "private-audio" not in summary


def test_dependency_debug_logs_are_suppressed() -> None:
    loggers = tuple(logging.getLogger(name) for name in ("openai", "httpcore"))
    original_levels = tuple(logger.level for logger in loggers)
    try:
        configure_dependency_loggers()

        for logger in loggers:
            assert logger.level == logging.INFO
            assert not logger.isEnabledFor(logging.DEBUG)
            assert logger.isEnabledFor(logging.INFO)
    finally:
        for logger, original_level in zip(loggers, original_levels, strict=True):
            logger.setLevel(original_level)
