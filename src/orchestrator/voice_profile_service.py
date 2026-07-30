"""模块契约说明.

职责: 提供
orchestrator.voice_profile_service
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import replace
from typing import final

from orchestrator.identity import (
    ProfileCorrection,
    ProfileEnrollment,
    ProfileRecognition,
    ProfileRecognitionKnown,
    ProfileRecognitionResult,
    ProfileRecognitionUnknown,
    VoiceProfileConsentError,
    VoiceProfileId,
    VoiceProfileVault,
)
from orchestrator.ids import SessionId
from orchestrator.memory import MutableMemory, MutableMemorySnapshot
from orchestrator.profile_store import (
    ProfileAuditEntry,
    ProfileLifecycle,
    VoiceProfileRecord,
    VoiceProfileSnapshot,
    VoiceProfileStore,
)
from orchestrator.state_snapshots import ConsentRevision, ProfileRevision


@final
class VoiceProfileService:
    """类契约说明.

    职责: 定义 VoiceProfileService
    的状态、行为和对外协作边界。
    契约: 方法: __init__、enroll、recognize、co
    rrect、confirm、revoke_consent。
    """

    def __init__(
        self,
        *,
        session_id: SessionId,
        vault: VoiceProfileVault,
        minimum_confidence: int,
        store: VoiceProfileStore | None = None,
    ) -> None:
        """函数契约说明.

        功能: 初始化 VoiceProfileService
        的字段并建立实例不变式。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。 vault:
        VoiceProfileVault。 必填。
        minimum_confidence: int。 必填。
        store: VoiceProfileStore | None。
        可省略。
        契约: 同步调用。 返回 `None`。
        """
        self._session_id = session_id

        self._vault = vault

        self._minimum_confidence = minimum_confidence

        self._store = store

        snapshot = store.load(session_id) if store is not None else None

        self._records = (
            {}
            if snapshot is None
            else {record.profile_id: record for record in snapshot.records}
        )

        self._profile_revision = (
            ProfileRevision(0) if snapshot is None else snapshot.profile_revision
        )

        self._consent_revision = (
            ConsentRevision(0) if snapshot is None else snapshot.consent_revision
        )

    def enroll(self, enrollment: ProfileEnrollment) -> VoiceProfileId:
        """函数契约说明.

        功能: 执行 enroll 的同步逻辑,并协调
        store_encrypted,
        ProfileRevision,
        ConsentRevision, dict。
        参数: self 表示当前实例。 enrollment:
        ProfileEnrollment。 必填。
        契约: 同步调用。 返回 `VoiceProfileId`。
        可能抛出 VoiceProfileConsentError。
        """
        if not enrollment.consented:
            raise VoiceProfileConsentError(enrollment.profile_id)

        self._vault.store_encrypted(
            enrollment.profile_id,
            enrollment.encrypted_template,
        )

        profile_revision = ProfileRevision(self._profile_revision + 1)

        consent_revision = ConsentRevision(self._consent_revision + 1)

        records = dict(self._records)

        records[enrollment.profile_id] = VoiceProfileRecord(
            profile_id=enrollment.profile_id,
            preferred_name=enrollment.preferred_name,
            purpose=enrollment.purpose,
            confirmed=enrollment.confirmed,
            expires_at_ms=enrollment.expires_at_ms,
            lifecycle=ProfileLifecycle.ACTIVE,
            revision=profile_revision,
            audit=(ProfileAuditEntry("enrolled", profile_revision),),
        )

        try:
            self._publish(records, profile_revision, consent_revision)

        except OSError:
            self._vault.delete(enrollment.profile_id)

            raise

        return enrollment.profile_id

    def recognize(
        self,
        recognition: ProfileRecognition,
        *,
        now_ms: int = 0,
    ) -> ProfileRecognitionResult:
        """函数契约说明.

        功能: 执行 recognize 的同步逻辑,并协调 get,
        ProfileRecognitionKnown,
        ProfileRecognitionUnknown,
        _is_expired。
        参数: self 表示当前实例。 recognition:
        ProfileRecognition。 必填。 now_ms:
        int。 可省略。
        契约: 同步调用。 返回
        `ProfileRecognitionResult`。
        """
        profile_id = recognition.profile_id

        if profile_id is None or recognition.confidence < self._minimum_confidence:
            return ProfileRecognitionUnknown()

        record = self._records.get(profile_id)

        if (
            record is None
            or record.lifecycle is not ProfileLifecycle.ACTIVE
            or not record.confirmed
            or _is_expired(record, now_ms)
        ):
            return ProfileRecognitionUnknown()

        return ProfileRecognitionKnown(profile_id, record.preferred_name)

    def correct(self, correction: ProfileCorrection) -> ProfileRecognitionResult:
        """函数契约说明.

        功能: 执行 correct 的同步逻辑,并协调 get,
        ProfileRevision, replace, dict。
        参数: self 表示当前实例。 correction:
        ProfileCorrection。 必填。
        契约: 同步调用。 返回
        `ProfileRecognitionResult`。
        """
        record = self._records.get(correction.profile_id)

        if record is None or record.lifecycle is not ProfileLifecycle.ACTIVE:
            return ProfileRecognitionUnknown()

        profile_revision = ProfileRevision(self._profile_revision + 1)

        updated = replace(
            record,
            preferred_name=correction.preferred_name,
            revision=profile_revision,
            audit=(
                *record.audit,
                ProfileAuditEntry("corrected", profile_revision),
            ),
        )

        records = dict(self._records)

        records[correction.profile_id] = updated

        self._publish(records, profile_revision, self._consent_revision)

        return ProfileRecognitionKnown(correction.profile_id, updated.preferred_name)

    def confirm(self, profile_id: VoiceProfileId) -> ProfileRecognitionResult:
        """函数契约说明.

        功能: 执行 confirm 的同步逻辑,并协调 get,
        ProfileRevision, replace, dict。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回
        `ProfileRecognitionResult`。
        """
        record = self._records.get(profile_id)

        if record is None or record.lifecycle is not ProfileLifecycle.ACTIVE:
            return ProfileRecognitionUnknown()

        profile_revision = ProfileRevision(self._profile_revision + 1)

        updated = replace(
            record,
            confirmed=True,
            revision=profile_revision,
            audit=(
                *record.audit,
                ProfileAuditEntry("confirmed", profile_revision),
            ),
        )

        records = dict(self._records)

        records[profile_id] = updated

        self._publish(records, profile_revision, self._consent_revision)

        return ProfileRecognitionKnown(profile_id, updated.preferred_name)

    def revoke_consent(self, profile_id: VoiceProfileId) -> None:
        """函数契约说明.

        功能: 执行 revoke_consent 的同步逻辑,并协调
        _transition, delete。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._transition(profile_id, ProfileLifecycle.REVOKED, "revoked")

        self._vault.delete(profile_id)

    def expire(self, *, now_ms: int) -> bool:
        """函数契约说明.

        功能: 执行 expire 的同步逻辑,并协调 tuple,
        _transition, delete, items。
        参数: self 表示当前实例。 now_ms: int。
        必填。
        契约: 同步调用。 返回 `bool`。
        """
        expired_ids = tuple(
            profile_id
            for profile_id, record in self._records.items()
            if record.lifecycle is ProfileLifecycle.ACTIVE
            and _is_expired(record, now_ms)
        )

        if not expired_ids:
            return False

        for profile_id in expired_ids:
            self._transition(profile_id, ProfileLifecycle.EXPIRED, "expired")

            self._vault.delete(profile_id)

        return True

    def delete(self, profile_id: VoiceProfileId) -> None:
        """函数契约说明.

        功能: 执行 delete 的同步逻辑,并协调
        _transition, delete。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._transition(profile_id, ProfileLifecycle.DELETED, "deleted")

        self._vault.delete(profile_id)

    def bind_memory(self, memory: MutableMemory) -> MutableMemorySnapshot:
        """函数契约说明.

        功能: 执行 bind_memory 的同步逻辑,并协调
        set_profile_revisions。
        参数: self 表示当前实例。 memory:
        MutableMemory。 必填。
        契约: 同步调用。 返回
        `MutableMemorySnapshot`。
        """
        return memory.set_profile_revisions(
            self._profile_revision,
            self._consent_revision,
        )

    def _transition(
        self,
        profile_id: VoiceProfileId,
        lifecycle: ProfileLifecycle,
        action: str,
    ) -> None:
        """函数契约说明.

        功能: 执行 _transition 的同步逻辑,并协调
        get, ProfileRevision,
        ConsentRevision, dict。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。 lifecycle:
        ProfileLifecycle。 必填。 action:
        str。 必填。
        契约: 同步调用。 返回 `None`。
        """
        record = self._records.get(profile_id)

        if record is None or record.lifecycle is lifecycle:
            return

        profile_revision = ProfileRevision(self._profile_revision + 1)

        consent_revision = ConsentRevision(self._consent_revision + 1)

        records = dict(self._records)

        records[profile_id] = replace(
            record,
            lifecycle=lifecycle,
            revision=profile_revision,
            audit=(
                *record.audit,
                ProfileAuditEntry(action, profile_revision),
            ),
        )

        self._publish(records, profile_revision, consent_revision)

    def _publish(
        self,
        records: dict[VoiceProfileId, VoiceProfileRecord],
        profile_revision: ProfileRevision,
        consent_revision: ConsentRevision,
    ) -> None:
        """函数契约说明.

        功能: 执行 _publish 的同步逻辑,并协调 save,
        VoiceProfileSnapshot, tuple,
        values。
        参数: self 表示当前实例。 records:
        dict[VoiceProfileId,
        VoiceProfileRecord]。 必填。
        profile_revision:
        ProfileRevision。 必填。
        consent_revision:
        ConsentRevision。 必填。
        契约: 同步调用。 返回 `None`。
        """
        if self._store is not None:
            self._store.save(
                VoiceProfileSnapshot(
                    session_id=self._session_id,
                    profile_revision=profile_revision,
                    consent_revision=consent_revision,
                    records=tuple(records.values()),
                )
            )

        self._records = records

        self._profile_revision = profile_revision

        self._consent_revision = consent_revision


def _is_expired(record: VoiceProfileRecord, now_ms: int) -> bool:
    """函数契约说明.

    功能: 执行 _is_expired 的同步逻辑,并维持签名契约。
    参数: record: VoiceProfileRecord。 必填。
    now_ms: int。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    return record.expires_at_ms is not None and now_ms >= record.expires_at_ms
