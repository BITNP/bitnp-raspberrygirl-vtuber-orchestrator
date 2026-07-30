"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import pytest

from orchestrator.identity import (
    EncryptedVoiceTemplate,
    InMemoryVoiceProfileVault,
    ProfileCorrection,
    ProfileEnrollment,
    ProfileRecognition,
    ProfileRecognitionKnown,
    ProfileRecognitionUnknown,
    RecognitionConfidence,
    VoiceProfileConsentError,
    VoiceProfileId,
)
from orchestrator.ids import SessionId
from orchestrator.memory import MemoryPolicy, MutableMemory
from orchestrator.voice_profile_service import VoiceProfileService


def test_unknown_or_low_confidence_voice_never_binds_profile_memory() -> None:
    # Given: a consented profile and a recognition threshold of 90.

    """函数契约说明.

    功能: 验证 unknown or low confidence
    voice never binds profile memory
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    service = _service()

    profile_id = _enroll(service)

    # When: recognition is unknown or below the confidence threshold.

    unknown = service.recognize(ProfileRecognition(None, RecognitionConfidence(100)))

    low_confidence = service.recognize(
        ProfileRecognition(profile_id, RecognitionConfidence(89))
    )

    # Then: neither result carries personalization or authorization state.

    assert unknown == ProfileRecognitionUnknown()

    assert low_confidence == ProfileRecognitionUnknown()


def test_revocation_deletion_and_correction_control_personalization() -> None:
    # Given: an enrolled profile whose preferred name can personalize a prompt.

    """函数契约说明.

    功能: 验证 revocation deletion and
    correction control personalization
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    service = _service()

    profile_id = _enroll(service)

    # When: the profile is corrected, consent is revoked, then it is deleted.

    corrected = service.correct(ProfileCorrection(profile_id, "莓莓"))

    service.revoke_consent(profile_id)

    revoked = service.recognize(
        ProfileRecognition(profile_id, RecognitionConfidence(99))
    )

    service.delete(profile_id)

    deleted = service.recognize(
        ProfileRecognition(profile_id, RecognitionConfidence(99))
    )

    # Then: correction updates personalization; revoked/deleted stay unknown.

    assert corrected == ProfileRecognitionKnown(profile_id, "莓莓")

    assert revoked == ProfileRecognitionUnknown()

    assert deleted == ProfileRecognitionUnknown()


def test_profile_lifecycle_binds_only_revisions_into_mutable_memory() -> None:
    # Given: consented profile metadata and separate ordinary prompt memory.

    """函数契约说明.

    功能: 验证 profile lifecycle binds only
    revisions into mutable memory
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    service = _service()

    memory = MutableMemory(session_id=SessionId("session-1"), policy=MemoryPolicy())

    # When: enrollment and consent revocation synchronize their lifecycle revisions.

    _ = _enroll(service)

    enrolled = service.bind_memory(memory)

    service.revoke_consent(VoiceProfileId("profile-1"))

    revoked = service.bind_memory(memory)

    # Then: memory receives invalidation revisions, never voice-template content.

    assert int(enrolled.profile_revision) == 1

    assert int(revoked.consent_revision) == 2

    assert enrolled.entries == ()


def test_no_consent_or_expired_profile_cannot_personalize_and_erases_template() -> None:
    # Given: explicit consent is required and a consented profile has a retention limit.

    """函数契约说明.

    功能: 验证 no consent or expired profile
    cannot personalize and erases
    template 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    vault = InMemoryVoiceProfileVault()

    service = VoiceProfileService(
        session_id=SessionId("session-1"), vault=vault, minimum_confidence=90
    )

    refused = ProfileEnrollment(
        profile_id=VoiceProfileId("refused"),
        preferred_name="小莓",
        encrypted_template=EncryptedVoiceTemplate(b"refused"),
        consented=False,
    )

    # When: enrollment omits consent, then a time-limited profile reaches expiry.

    with pytest.raises(VoiceProfileConsentError):
        _ = service.enroll(refused)

    profile_id = service.enroll(
        ProfileEnrollment(
            profile_id=VoiceProfileId("profile-1"),
            preferred_name="小莓",
            encrypted_template=EncryptedVoiceTemplate(b"ciphertext"),
            consented=True,
            expires_at_ms=100,
        )
    )

    assert service.expire(now_ms=100) is True

    # Then: neither lifecycle path leaves a reusable recognition or template.

    assert vault.template(VoiceProfileId("refused")) is None

    assert vault.template(profile_id) is None

    assert (
        service.recognize(
            ProfileRecognition(profile_id, RecognitionConfidence(99)), now_ms=100
        )
        == ProfileRecognitionUnknown()
    )


def _service() -> VoiceProfileService:
    """函数契约说明.

    功能: 执行 _service 的同步逻辑,并协调
    VoiceProfileService, SessionId,
    InMemoryVoiceProfileVault。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `VoiceProfileService`。
    """

    return VoiceProfileService(
        session_id=SessionId("session-1"),
        vault=InMemoryVoiceProfileVault(),
        minimum_confidence=90,
    )


def _enroll(service: VoiceProfileService) -> VoiceProfileId:
    """函数契约说明.

    功能: 执行 _enroll 的同步逻辑,并协调 enroll,
    ProfileEnrollment, VoiceProfileId,
    EncryptedVoiceTemplate。
    参数: service: VoiceProfileService。
    必填。
    契约: 同步调用。 返回 `VoiceProfileId`。
    """

    result = service.enroll(
        ProfileEnrollment(
            profile_id=VoiceProfileId("profile-1"),
            preferred_name="小莓",
            encrypted_template=EncryptedVoiceTemplate(b"ciphertext"),
            consented=True,
        )
    )

    assert result == VoiceProfileId("profile-1")

    return result
