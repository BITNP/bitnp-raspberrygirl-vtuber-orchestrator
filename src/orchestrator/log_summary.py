"""Safe, compact descriptions for provider payloads in operational logs."""

from __future__ import annotations

from typing import Final
from urllib.parse import urlsplit

_PREVIEW_BYTES: Final = 30


def binary_summary(value: bytes, *, preview_bytes: int = _PREVIEW_BYTES) -> str:
    """Describe binary data using a bounded hexadecimal prefix."""
    preview = value[:preview_bytes].hex()
    suffix = "" if len(value) <= preview_bytes else "…"
    return f"bytes={len(value)} hex={preview}{suffix}"


def reference_audio_summary(value: str) -> str:
    """Describe a voice reference without ever logging its encoded audio."""
    stripped = value.strip()
    if stripped.startswith("data:"):
        header, separator, encoded = stripped.partition(",")
        encoding = "base64" if ";base64" in header.lower() else "inline"
        payload_chars = len(encoded) if separator else 0
        media_type = header.removeprefix("data:").split(";", 1)[0]
        return (
            f"kind=data_url media_type={media_type!r} "
            f"encoding={encoding} payload_chars={payload_chars}"
        )

    parsed = urlsplit(stripped)
    if parsed.scheme:
        return f"kind=uri scheme={parsed.scheme!r} chars={len(stripped)}"
    return f"kind=path chars={len(stripped)}"
