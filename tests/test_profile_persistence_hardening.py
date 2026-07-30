
from collections.abc import Callable
from pathlib import Path

import pytest

from orchestrator import interaction_ingress, profile_store
from orchestrator.identity import (
    EncryptedVoiceTemplate,
    InMemoryVoiceProfileVault,
    ProfileEnrollment,
    ProfileRecognition,
    ProfileRecognitionKnown,
    ProfileRecognitionUnknown,
    RecognitionConfidence,
    VoiceProfileId,
)
from orchestrator.ids import SessionId
from orchestrator.profile_store import JsonVoiceProfileStore, VoiceProfileSnapshot
from orchestrator.profile_vault import FileVoiceProfileVault
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.session_data import ProfilePersistence, SessionDataState
from orchestrator.voice_profile_service import VoiceProfileService


def test_sessions_keep_profile_metadata_and_templates_isolated(tmp_path: Path) -> None:
    # Given: two sessions sharing a state root and equal profile identifiers.


    profile_id = VoiceProfileId("profile")

    first = _state(tmp_path, SessionId("one"))

    # When: only the first session enrolls a confirmed profile.

    _ = first.enroll_profile(_enrollment(profile_id, expires_at_ms=None))

    # Then: restart preserves only the first session's personalization.

    first_result = _state(tmp_path, SessionId("one")).recognize_profile(
        _recognition(profile_id)
    )

    second_result = _state(tmp_path, SessionId("two")).recognize_profile(
        _recognition(profile_id)
    )

    assert first_result == ProfileRecognitionKnown(profile_id, "小莓")

    assert second_result == ProfileRecognitionUnknown()


def test_session_storage_key_cannot_escape_state_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a traversal-shaped external session identifier and trusted state root.


    monkeypatch.setenv("ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))

    # When: production composition derives its durable session directory.

    storage_root = interaction_ingress.session_storage_root(SessionId("../../outside"))

    # Then: it is a deterministic child of state root rather than raw traversal.

    assert storage_root.parent == tmp_path / "state"

    assert storage_root.name != "outside"


def test_startup_tombstones_expired_profile_and_erases_template(tmp_path: Path) -> None:
    # Given: an enrolled profile whose declared retention deadline has passed.


    profile_id = VoiceProfileId("expired")

    first = _state(tmp_path, SessionId("one"), clock=lambda: 0)

    _ = first.enroll_profile(_enrollment(profile_id, expires_at_ms=10))

    # When: a restarted state hydrates through an injected real-time boundary.

    restarted = _state(tmp_path, SessionId("one"), clock=lambda: 10)

    # Then: it cannot recognize and its opaque template is absent.

    assert restarted.recognize_profile(_recognition(profile_id)) == (
        ProfileRecognitionUnknown()
    )

    assert list((tmp_path / "one" / "voice-templates").glob("*.template")) == []


def test_vault_encodes_traversal_profile_id_without_escaping_directory(
    tmp_path: Path,
) -> None:
    # Given: a malicious-looking profile identifier at the vault boundary.


    vault = FileVoiceProfileVault(tmp_path / "vault")

    # When: an opaque template is stored for that identifier.

    vault.store_encrypted(
        VoiceProfileId("../../escape"), EncryptedVoiceTemplate(b"opaque")
    )

    # Then: the only file is a fixed safe template filename inside the vault.

    assert list((tmp_path / "vault").glob("*.template"))

    assert not (tmp_path / "escape.template").exists()


def test_metadata_write_failure_removes_newly_written_template(tmp_path: Path) -> None:
    # Given: a vault and a metadata store that cannot persist enrollment.


    vault = FileVoiceProfileVault(tmp_path / "vault")

    service = VoiceProfileService(
        session_id=SessionId("one"),
        vault=vault,
        minimum_confidence=90,
        store=_FailingStore(),
    )

    # When: enrollment reaches the metadata commit boundary.

    with pytest.raises(_MetadataWriteError):
        _ = service.enroll(_enrollment(VoiceProfileId("profile"), expires_at_ms=None))

    # Then: no template orphan remains and recognition stays unknown.

    assert list((tmp_path / "vault").glob("*.template")) == []

    assert service.recognize(_recognition(VoiceProfileId("profile"))) == (
        ProfileRecognitionUnknown()
    )


def test_profile_deletion_invalidates_tasks_and_erases_the_separate_template(
    tmp_path: Path,
) -> None:
    # Given: a durable session with an enrolled, confirmed profile.
    profile_id = VoiceProfileId("profile")
    state = _state(tmp_path, SessionId("one"))
    _ = state.enroll_profile(_enrollment(profile_id, expires_at_ms=None))
    captured = state.task_snapshot

    # When: the user deletes the opt-in voice profile.
    state.delete_profile(profile_id)

    # Then: pending work is stale and only the vault material is erased.
    assert state.is_current(captured) is False
    assert list((tmp_path / "one" / "voice-templates").glob("*.template")) == []
    assert state.memory.snapshot.entries == ()


def test_confirmation_save_failure_does_not_publish_live_state() -> None:
    # Given: an unconfirmed profile and a store that fails its next save.


    vault = InMemoryVoiceProfileVault()

    store = _ToggleStore()

    service = _service(vault, store)

    profile_id = service.enroll(
        ProfileEnrollment(
            profile_id=VoiceProfileId("profile"),
            preferred_name="小莓",
            encrypted_template=EncryptedVoiceTemplate(b"opaque"),
            consented=True,
            confirmed=False,
        )
    )

    store.fail = True

    # When: confirmation cannot durably commit.

    with pytest.raises(_MetadataWriteError):
        _ = service.confirm(profile_id)

    # Then: both live and reloaded state remain unconfirmed and template remains.

    assert service.recognize(_recognition(profile_id)) == ProfileRecognitionUnknown()

    assert vault.template(profile_id) is not None


def test_revocation_save_failure_does_not_delete_live_template() -> None:
    # Given: a confirmed profile and a store that fails its next save.


    vault = InMemoryVoiceProfileVault()

    store = _ToggleStore()

    service = _service(vault, store)

    profile_id = service.enroll(_enrollment(VoiceProfileId("profile"), None))

    _ = service.confirm(profile_id)

    store.fail = True

    # When: revocation cannot durably commit.

    with pytest.raises(_MetadataWriteError):
        service.revoke_consent(profile_id)

    # Then: live recognition and template remain aligned with durable active state.

    assert service.recognize(_recognition(profile_id)) == (
        ProfileRecognitionKnown(profile_id, "小莓")
    )

    assert vault.template(profile_id) is not None


def test_directory_fsync_failure_reloads_completed_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: an atomic store whose parent-directory fsync fails after replacement.


    store = JsonVoiceProfileStore(tmp_path / "one" / "voice-profiles.json")

    state = _state(tmp_path, SessionId("one"))

    _ = state.enroll_profile(_enrollment(VoiceProfileId("profile"), None))

    snapshot = store.load(SessionId("one"))

    assert snapshot is not None

    monkeypatch.setattr(profile_store, "_fsync_directory", _raise_fsync)

    # When: the identical snapshot is saved after the replace completed.

    store.save(snapshot)

    # Then: reload reconciliation accepts the durable replacement.

    assert store.load(SessionId("one")) == snapshot


class _FailingStore:

    def save(self, snapshot: VoiceProfileSnapshot) -> None:

        _ = snapshot

        raise _MetadataWriteError

    def load(self, session_id: SessionId) -> None:

        _ = session_id


class _MetadataWriteError(OSError):
    ...



class _ToggleStore:

    def __init__(self) -> None:

        self.snapshot: VoiceProfileSnapshot | None = None

        self.fail: bool = False

    def save(self, snapshot: VoiceProfileSnapshot) -> None:

        if self.fail:
            raise _MetadataWriteError

        self.snapshot = snapshot

    def load(self, session_id: SessionId) -> VoiceProfileSnapshot | None:

        _ = session_id

        return self.snapshot


def _service(
    vault: InMemoryVoiceProfileVault, store: _ToggleStore
) -> VoiceProfileService:

    return VoiceProfileService(
        session_id=SessionId("one"), vault=vault, minimum_confidence=90, store=store
    )


def _raise_fsync(directory: Path) -> None:

    _ = directory

    raise OSError


def _state(
    root: Path,
    session_id: SessionId,
    clock: Callable[[], int] = lambda: 0,
) -> SessionDataState:

    session_root = root / str(session_id)

    return SessionDataState.create(
        session_id=session_id,
        retrieval=RetrievalFixtureProvider(refs=()),
        profile_persistence=ProfilePersistence(
            store=JsonVoiceProfileStore(session_root / "voice-profiles.json"),
            vault_directory=session_root / "voice-templates",
            clock=clock,
        ),
    )


def _enrollment(
    profile_id: VoiceProfileId,
    expires_at_ms: int | None,
) -> ProfileEnrollment:

    return ProfileEnrollment(
        profile_id=profile_id,
        preferred_name="小莓",
        encrypted_template=EncryptedVoiceTemplate(b"opaque"),
        consented=True,
        expires_at_ms=expires_at_ms,
    )


def _recognition(profile_id: VoiceProfileId) -> ProfileRecognition:

    return ProfileRecognition(profile_id, RecognitionConfidence(99))
