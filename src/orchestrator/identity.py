"""Immutable consented voice-profile types and opaque template boundary."""

from dataclasses import dataclass
from typing import NewType, Protocol, final, override

VoiceProfileId = NewType("VoiceProfileId", str)
RecognitionConfidence = NewType("RecognitionConfidence", int)


@dataclass(frozen=True, slots=True)
class EncryptedVoiceTemplate:
    """Opaque ciphertext accepted only by an access-controlled vault boundary."""

    ciphertext: bytes


class VoiceProfileVault(Protocol):
    """Encrypted, access-controlled biometric template storage boundary."""

    def store_encrypted(
        self,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> None:
        """Persist a template without exposing it to scheduler state."""
        ...

    def delete(self, profile_id: VoiceProfileId) -> None:
        """Permanently remove a profile template."""
        ...


@final
class InMemoryVoiceProfileVault:
    """Test-only encrypted vault fake; production implementations enforce access."""

    def __init__(self) -> None:
        """Create an empty test vault."""
        self._templates: dict[VoiceProfileId, EncryptedVoiceTemplate] = {}

    def store_encrypted(
        self,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> None:
        """Retain one opaque encrypted template for a test profile."""
        self._templates[profile_id] = template

    def delete(self, profile_id: VoiceProfileId) -> None:
        """Remove a test template."""
        _ = self._templates.pop(profile_id, None)

    def template(self, profile_id: VoiceProfileId) -> EncryptedVoiceTemplate | None:
        """Expose ciphertext only for tests, never to personalization callers."""
        return self._templates.get(profile_id)


@dataclass(frozen=True, slots=True)
class ProfileEnrollment:
    """Explicit opt-in enrollment carrying opaque encrypted biometric material."""

    profile_id: VoiceProfileId
    preferred_name: str
    encrypted_template: EncryptedVoiceTemplate
    consented: bool
    confirmed: bool = True
    expires_at_ms: int | None = None
    purpose: str = "personalization"


@dataclass(frozen=True, slots=True)
class ProfileRecognition:
    """Recognition candidate supplied by a consented recognition capability."""

    profile_id: VoiceProfileId | None
    confidence: RecognitionConfidence


@dataclass(frozen=True, slots=True)
class ProfileCorrection:
    """User-confirmed correction of non-biometric personalization only."""

    profile_id: VoiceProfileId
    preferred_name: str


@dataclass(frozen=True, slots=True)
class ProfileRecognitionKnown:
    """Non-authorizing personalization context returned for a trusted match."""

    profile_id: VoiceProfileId
    preferred_name: str


@dataclass(frozen=True, slots=True)
class ProfileRecognitionUnknown:
    """Safe default for unknown, revoked, deleted, or low-confidence voices."""


type ProfileRecognitionResult = ProfileRecognitionKnown | ProfileRecognitionUnknown


@dataclass(frozen=True, slots=True)
class VoiceProfileConsentError(ValueError):
    """Raised when enrollment omits the explicit consent requirement."""

    profile_id: VoiceProfileId

    @override
    def __str__(self) -> str:
        """Render the safe enrollment refusal."""
        return f"voice profile enrollment requires consent: {self.profile_id}"
