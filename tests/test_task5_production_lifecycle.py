
from pathlib import Path

from orchestrator.identity import (
    EncryptedVoiceTemplate,
    ProfileCorrection,
    ProfileEnrollment,
    ProfileRecognition,
    ProfileRecognitionKnown,
    ProfileRecognitionUnknown,
    RecognitionConfidence,
    VoiceProfileId,
)
from orchestrator.ids import SessionId, TraceId, TurnId
from orchestrator.memory import (
    MemoryCategory,
    MemoryConfidence,
    MemoryKey,
    MemoryProposal,
    MemoryProvenance,
    MemorySource,
    ProposalRevision,
)
from orchestrator.memory_store import JsonMemoryStore
from orchestrator.profile_store import JsonVoiceProfileStore
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.session_data import ProfilePersistence, SessionDataState


def test_persisted_memory_reloads_across_session_data_restart(tmp_path: Path) -> None:
    # Given: a durable store used by two state instances for one session.


    path = tmp_path / "memory.json"

    first = _state(path)

    # When: an approved preference is committed before recreating state.

    _ = first.reduce_memory(_proposal())

    restarted = _state(path)

    # Then: the human-readable preference and revision survive the restart.

    assert restarted.memory.snapshot.revision == 1

    assert restarted.prompt_snapshot(max_context_chars=100).memory_entries == (
        "preferred_name=小莓",
    )


def test_unconfirmed_expired_and_revoked_profiles_never_personalize() -> None:
    # Given: an explicitly consented profile that still needs confirmation.


    state = _state(None)

    profile_id = VoiceProfileId("profile-1")

    _ = state.enroll_profile(
        ProfileEnrollment(
            profile_id=profile_id,
            preferred_name="小莓",
            encrypted_template=EncryptedVoiceTemplate(b"opaque"),
            consented=True,
            confirmed=False,
            expires_at_ms=100,
        )
    )

    # When: recognition happens before confirmation, after expiry, and after revocation.

    unconfirmed = state.recognize_profile(
        ProfileRecognition(profile_id, RecognitionConfidence(99))
    )

    _ = state.confirm_profile(profile_id)

    confirmed = state.recognize_profile(
        ProfileRecognition(profile_id, RecognitionConfidence(99))
    )

    assert state.expire_profiles(now_ms=100) is True

    expired = state.recognize_profile(
        ProfileRecognition(profile_id, RecognitionConfidence(99))
    )

    state.revoke_profile_consent(profile_id)

    revoked = state.recognize_profile(
        ProfileRecognition(profile_id, RecognitionConfidence(99))
    )

    # Then: only a confirmed, live profile yields personalization.

    assert unconfirmed == ProfileRecognitionUnknown()

    assert confirmed == ProfileRecognitionKnown(profile_id, "小莓")

    assert expired == ProfileRecognitionUnknown()

    assert revoked == ProfileRecognitionUnknown()


def test_profile_lifecycle_metadata_reloads_without_exposing_template(
    tmp_path: Path,
) -> None:
    # Given: separate durable metadata and opaque-template storage for one session.


    profile_path = tmp_path / "voice-profiles.json"

    vault_directory = tmp_path / "voice-templates"

    first = _state(
        None,
        profile_path=profile_path,
        vault_directory=vault_directory,
    )

    profile_id = VoiceProfileId("profile-1")

    # When: a consented profile is confirmed and corrected before restart.

    _ = first.enroll_profile(
        ProfileEnrollment(
            profile_id=profile_id,
            preferred_name="小莓",
            encrypted_template=EncryptedVoiceTemplate(b"opaque-template"),
            consented=True,
            confirmed=False,
            expires_at_ms=None,
        )
    )

    _ = first.confirm_profile(profile_id)

    _ = first.correct_profile(ProfileCorrection(profile_id, "莓莓"))

    restarted = _state(
        None,
        profile_path=profile_path,
        vault_directory=vault_directory,
    )

    # Then: metadata restores personalization while ciphertext remains out of records.

    assert restarted.recognize_profile(
        ProfileRecognition(profile_id, RecognitionConfidence(99))
    ) == ProfileRecognitionKnown(profile_id, "莓莓")

    assert restarted.task_snapshot.profile_revision == 3

    document = profile_path.read_text(encoding="utf-8")

    assert "personalization" in document

    assert "opaque-template" not in document


def test_revocation_and_deletion_remain_unknown_after_restart_and_invalidate_work(
    tmp_path: Path,
) -> None:
    # Given: pending profile-dependent work and durable profile lifecycle state.


    profile_path = tmp_path / "voice-profiles.json"

    vault_directory = tmp_path / "voice-templates"

    state = _state(
        None,
        profile_path=profile_path,
        vault_directory=vault_directory,
    )

    profile_id = VoiceProfileId("profile-1")

    invalidations: list[str] = []

    state.invalidate_pending = invalidations.append

    _ = state.enroll_profile(
        ProfileEnrollment(
            profile_id=profile_id,
            preferred_name="小莓",
            encrypted_template=EncryptedVoiceTemplate(b"opaque-template"),
            consented=True,
        )
    )

    captured = state.task_snapshot

    # When: revocation races the pending result, then deletion survives restart.

    state.revoke_profile_consent(profile_id)

    revoked = _state(
        None,
        profile_path=profile_path,
        vault_directory=vault_directory,
    )

    revoked.delete_profile(profile_id)

    deleted = _state(
        None,
        profile_path=profile_path,
        vault_directory=vault_directory,
    )

    # Then: old work is stale, both states reject recognition, and cancellation ran.

    assert state.is_current(captured) is False

    assert invalidations == ["identity_revoked:profile-1"]

    assert (
        revoked.recognize_profile(
            ProfileRecognition(profile_id, RecognitionConfidence(99))
        )
        == ProfileRecognitionUnknown()
    )

    assert (
        deleted.recognize_profile(
            ProfileRecognition(profile_id, RecognitionConfidence(99))
        )
        == ProfileRecognitionUnknown()
    )


def _state(
    path: Path | None,
    *,
    profile_path: Path | None = None,
    vault_directory: Path | None = None,
) -> SessionDataState:

    return SessionDataState.create(
        session_id=SessionId("session-1"),
        retrieval=RetrievalFixtureProvider(refs=()),
        memory_store=None if path is None else JsonMemoryStore(path),
        profile_persistence=ProfilePersistence(
            store=None if profile_path is None else JsonVoiceProfileStore(profile_path),
            vault_directory=vault_directory,
        ),
    )


def _proposal() -> MemoryProposal:

    return MemoryProposal(
        key=MemoryKey("preferred_name"),
        value="小莓",
        category=MemoryCategory.ORDINARY_PREFERENCE,
        confidence=MemoryConfidence(95),
        base_revision=ProposalRevision(0),
        provenance=MemoryProvenance(
            source=MemorySource.USER_REQUEST,
            trace_id=TraceId("trace-1"),
            session_id=SessionId("session-1"),
            turn_id=TurnId("turn-1"),
            evidence_id="accepted-turn-1",
        ),
    )
