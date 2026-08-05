from __future__ import annotations

import json
import math
import os
import struct
from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import Final, final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from orchestrator.identity import EncryptedVoiceTemplate, VoiceProfileId
from orchestrator.json_boundary import JsonBoundaryError, JsonValue, parse_json_value

FORMAT_VERSION: Final = 1
FORMAT_NAME: Final = "float32-unit-vector"
AES_KEY_BYTES: Final = 32
NONCE_BYTES: Final = 12
RTP_HALF_RANGE: Final = 0x8000_0000


class VoiceTemplateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecryptedVoiceTemplate:
    model_revision: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VoiceMatch:
    profile_id: VoiceProfileId | None
    confidence: float


@final
class VoiceTemplateProtector:
    def __init__(self, key: bytes) -> None:
        if len(key) != AES_KEY_BYTES:
            raise VoiceTemplateError
        self._aead: AESGCM = AESGCM(key)

    def encrypt(
        self,
        *,
        session_id: str,
        profile_id: VoiceProfileId,
        model_revision: str,
        embedding: tuple[float, ...],
    ) -> EncryptedVoiceTemplate:
        normalized = _normalize(embedding)
        associated = _associated_data(
            session_id, profile_id, model_revision, len(normalized)
        )
        plaintext = struct.pack(f"<{len(normalized)}f", *normalized)
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = self._aead.encrypt(nonce, plaintext, associated)
        envelope = {
            "version": FORMAT_VERSION,
            "format": FORMAT_NAME,
            "model_revision": model_revision,
            "dimensions": len(normalized),
            "nonce": b64encode(nonce).decode("ascii"),
            "ciphertext": b64encode(ciphertext).decode("ascii"),
        }
        return EncryptedVoiceTemplate(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        )

    def decrypt(
        self,
        *,
        session_id: str,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> DecryptedVoiceTemplate:
        try:
            value = parse_json_value(template.ciphertext.decode("utf-8"))
        except (JsonBoundaryError, UnicodeError) as error:
            raise VoiceTemplateError from error
        if not isinstance(value, dict):
            raise VoiceTemplateError
        model_revision = value.get("model_revision")
        dimensions = value.get("dimensions")
        if (
            value.get("version") != FORMAT_VERSION
            or value.get("format") != FORMAT_NAME
            or not isinstance(model_revision, str)
            or type(dimensions) is not int
            or dimensions < 1
        ):
            raise VoiceTemplateError
        nonce = b64decode(_text(value, "nonce"), validate=True)
        ciphertext = b64decode(_text(value, "ciphertext"), validate=True)
        if len(nonce) != NONCE_BYTES:
            raise VoiceTemplateError
        plaintext = self._aead.decrypt(
            nonce,
            ciphertext,
            _associated_data(session_id, profile_id, model_revision, dimensions),
        )
        if len(plaintext) != dimensions * 4:
            raise VoiceTemplateError
        embedding = tuple(struct.unpack(f"<{dimensions}f", plaintext))
        return DecryptedVoiceTemplate(model_revision, embedding)


def match_voice(
    evidence_model_revision: str,
    evidence_embedding: tuple[float, ...],
    templates: dict[VoiceProfileId, DecryptedVoiceTemplate],
    *,
    threshold: float = 0.90,
    ambiguity_margin: float = 0.05,
) -> VoiceMatch:
    evidence = _normalize(evidence_embedding)
    scores = sorted(
        (
            (_cosine(evidence, template.embedding), profile_id)
            for profile_id, template in templates.items()
            if template.model_revision == evidence_model_revision
            and len(template.embedding) == len(evidence)
        ),
        reverse=True,
    )
    if not scores or scores[0][0] < threshold:
        return VoiceMatch(None, scores[0][0] if scores else 0.0)
    runner_up = scores[1][0] if len(scores) > 1 else -1.0
    if scores[0][0] - runner_up < ambiguity_margin:
        return VoiceMatch(None, scores[0][0])
    return VoiceMatch(scores[0][1], scores[0][0])


def modular_intervals_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> bool:
    first_span = (first_end - first_start) & 0xFFFF_FFFF
    second_span = (second_end - second_start) & 0xFFFF_FFFF
    return (
        ((second_start - first_start) & 0xFFFF_FFFF) <= first_span
        or ((first_start - second_start) & 0xFFFF_FFFF) <= second_span
    ) and first_span < RTP_HALF_RANGE and second_span < RTP_HALF_RANGE


def _normalize(embedding: tuple[float, ...]) -> tuple[float, ...]:
    if not embedding or any(not math.isfinite(value) for value in embedding):
        raise VoiceTemplateError
    magnitude = math.sqrt(sum(value * value for value in embedding))
    if magnitude == 0:
        raise VoiceTemplateError
    return tuple(value / magnitude for value in embedding)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    normalized_right = _normalize(right)
    return sum(a * b for a, b in zip(left, normalized_right, strict=True))


def _associated_data(
    session_id: str,
    profile_id: VoiceProfileId,
    model_revision: str,
    dimensions: int,
) -> bytes:
    return json.dumps(
        {
            "session_id": session_id,
            "profile_id": str(profile_id),
            "model_revision": model_revision,
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "dimensions": dimensions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _text(value: dict[str, JsonValue], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise VoiceTemplateError
    return item
