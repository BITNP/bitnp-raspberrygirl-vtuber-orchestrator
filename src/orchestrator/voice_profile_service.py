
from dataclasses import replace
from typing import final

from orchestrator.identity import (
    EncryptedVoiceTemplate,
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

    def __init__(
        self,
        *,
        session_id: SessionId,
        vault: VoiceProfileVault,
        minimum_confidence: int,
        store: VoiceProfileStore | None = None,
    ) -> None:
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

    def matchable_profile_ids(self, *, now_ms: int) -> tuple[VoiceProfileId, ...]:
        _ = self.expire(now_ms=now_ms)
        return tuple(
            profile_id
            for profile_id, record in self._records.items()
            if record.lifecycle is ProfileLifecycle.ACTIVE
            and record.confirmed
            and not record.re_enrollment_required
        )

    def encrypted_template(
        self, profile_id: VoiceProfileId
    ) -> EncryptedVoiceTemplate | None:
        return self._vault.load_encrypted(profile_id)

    @property
    def re_enrollment_required_ids(self) -> frozenset[VoiceProfileId]:
        return frozenset(
            profile_id
            for profile_id, record in self._records.items()
            if record.re_enrollment_required
        )

    def mark_re_enrollment_required(self, profile_id: VoiceProfileId) -> None:
        record = self._records.get(profile_id)
        if record is None or record.re_enrollment_required:
            return
        profile_revision = ProfileRevision(self._profile_revision + 1)
        updated = replace(
            record,
            re_enrollment_required=True,
            revision=profile_revision,
            audit=(
                *record.audit,
                ProfileAuditEntry("re_enrollment_required", profile_revision),
            ),
        )
        records = dict(self._records)
        records[profile_id] = updated
        self._publish(records, profile_revision, self._consent_revision)

    def correct(self, correction: ProfileCorrection) -> ProfileRecognitionResult:
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
        self._transition(profile_id, ProfileLifecycle.REVOKED, "revoked")

        self._vault.delete(profile_id)

    def expire(self, *, now_ms: int) -> bool:
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
        self._transition(profile_id, ProfileLifecycle.DELETED, "deleted")

        self._vault.delete(profile_id)

    def delete_all(self) -> None:
        for profile_id in tuple(self._records):
            self.delete(profile_id)

    def bind_memory(self, memory: MutableMemory) -> MutableMemorySnapshot:
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
    return record.expires_at_ms is not None and now_ms >= record.expires_at_ms
