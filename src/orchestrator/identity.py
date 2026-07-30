
from dataclasses import dataclass
from typing import NewType, Protocol, final, override

VoiceProfileId = NewType("VoiceProfileId", str)

RecognitionConfidence = NewType("RecognitionConfidence", int)


@dataclass(frozen=True, slots=True)
class EncryptedVoiceTemplate:

    ciphertext: bytes


class VoiceProfileVault(Protocol):

    def store_encrypted(
        self,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> None:
        ...

    def delete(self, profile_id: VoiceProfileId) -> None:
        ...


@final
class InMemoryVoiceProfileVault:

    def __init__(self) -> None:
        self._templates: dict[VoiceProfileId, EncryptedVoiceTemplate] = {}

    def store_encrypted(
        self,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> None:
        self._templates[profile_id] = template

    def delete(self, profile_id: VoiceProfileId) -> None:
        _ = self._templates.pop(profile_id, None)

    def template(self, profile_id: VoiceProfileId) -> EncryptedVoiceTemplate | None:
        return self._templates.get(profile_id)


@dataclass(frozen=True, slots=True)
class ProfileEnrollment:

    profile_id: VoiceProfileId

    preferred_name: str

    encrypted_template: EncryptedVoiceTemplate

    consented: bool

    confirmed: bool = True

    expires_at_ms: int | None = None

    purpose: str = "personalization"


@dataclass(frozen=True, slots=True)
class ProfileRecognition:

    profile_id: VoiceProfileId | None

    confidence: RecognitionConfidence


@dataclass(frozen=True, slots=True)
class ProfileCorrection:

    profile_id: VoiceProfileId

    preferred_name: str


@dataclass(frozen=True, slots=True)
class ProfileRecognitionKnown:

    profile_id: VoiceProfileId

    preferred_name: str


@dataclass(frozen=True, slots=True)
class ProfileRecognitionUnknown:
    ...


type ProfileRecognitionResult = ProfileRecognitionKnown | ProfileRecognitionUnknown


@dataclass(frozen=True, slots=True)
class VoiceProfileConsentError(ValueError):

    profile_id: VoiceProfileId

    @override
    def __str__(self) -> str:
        return f"voice profile enrollment requires consent: {self.profile_id}"
