"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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

    """函数契约说明.

    功能: 验证 sessions keep profile
    metadata and templates isolated
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 session storage key cannot
    escape state root 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。 monkeypatch:
    pytest.MonkeyPatch。 必填。
    契约: 同步调用。 返回 `None`。
    """

    monkeypatch.setenv("ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))

    # When: production composition derives its durable session directory.

    storage_root = interaction_ingress.session_storage_root(SessionId("../../outside"))

    # Then: it is a deterministic child of state root rather than raw traversal.

    assert storage_root.parent == tmp_path / "state"

    assert storage_root.name != "outside"


def test_startup_tombstones_expired_profile_and_erases_template(tmp_path: Path) -> None:
    # Given: an enrolled profile whose declared retention deadline has passed.

    """函数契约说明.

    功能: 验证 startup tombstones expired
    profile and erases template
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 vault encodes traversal
    profile id without escaping
    directory 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 metadata write failure
    removes newly written template
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 confirmation save failure
    does not publish live state
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 revocation save failure does
    not delete live template
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 directory fsync failure
    reloads completed replacement
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。 monkeypatch:
    pytest.MonkeyPatch。 必填。
    契约: 同步调用。 返回 `None`。
    """

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
    """类契约说明.

    职责: 定义 _FailingStore 的状态、行为和对外协作边界。
    契约: 方法: save、load。
    """

    def save(self, snapshot: VoiceProfileSnapshot) -> None:
        """函数契约说明.

        功能: 执行 save 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 snapshot:
        VoiceProfileSnapshot。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = snapshot

        raise _MetadataWriteError

    def load(self, session_id: SessionId) -> None:
        """函数契约说明.

        功能: 执行 load 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。
        契约: 同步调用。 返回 `None`。
        """

        _ = session_id


class _MetadataWriteError(OSError):
    """类契约说明.

    职责: 表示 _MetadataWriteError
    错误类别,并携带调用方可处理的失败信息。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """



class _ToggleStore:
    """类契约说明.

    职责: 定义 _ToggleStore 的状态、行为和对外协作边界。
    契约: 方法: __init__、save、load。
    """

    def __init__(self) -> None:
        """函数契约说明.

        功能: 初始化 _ToggleStore
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        self.snapshot: VoiceProfileSnapshot | None = None

        self.fail: bool = False

    def save(self, snapshot: VoiceProfileSnapshot) -> None:
        """函数契约说明.

        功能: 执行 save 的同步逻辑,并产出 snapshot。
        参数: self 表示当前实例。 snapshot:
        VoiceProfileSnapshot。 必填。
        契约: 同步调用。 返回 `None`。
        """

        if self.fail:
            raise _MetadataWriteError

        self.snapshot = snapshot

    def load(self, session_id: SessionId) -> VoiceProfileSnapshot | None:
        """函数契约说明.

        功能: 执行 load 的同步逻辑,并产出 _。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。
        契约: 同步调用。 返回
        `VoiceProfileSnapshot | None`。
        """

        _ = session_id

        return self.snapshot


def _service(
    vault: InMemoryVoiceProfileVault, store: _ToggleStore
) -> VoiceProfileService:
    """函数契约说明.

    功能: 执行 _service 的同步逻辑,并协调
    VoiceProfileService, SessionId。
    参数: vault:
    InMemoryVoiceProfileVault。 必填。
    store: _ToggleStore。 必填。
    契约: 同步调用。 返回 `VoiceProfileService`。
    """

    return VoiceProfileService(
        session_id=SessionId("one"), vault=vault, minimum_confidence=90, store=store
    )


def _raise_fsync(directory: Path) -> None:
    """函数契约说明.

    功能: 执行 _raise_fsync 的同步逻辑,并产出 _。
    参数: directory: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    _ = directory

    raise OSError


def _state(
    root: Path,
    session_id: SessionId,
    clock: Callable[[], int] = lambda: 0,
) -> SessionDataState:
    """函数契约说明.

    功能: 执行 _state 的同步逻辑,并协调 create, str,
    RetrievalFixtureProvider,
    ProfilePersistence。
    参数: root: Path。 必填。 session_id:
    SessionId。 必填。 clock: Callable[[],
    int]。 可省略。
    契约: 同步调用。 返回 `SessionDataState`。
    """

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
    """函数契约说明.

    功能: 执行 _enrollment 的同步逻辑,并协调
    ProfileEnrollment,
    EncryptedVoiceTemplate。
    参数: profile_id: VoiceProfileId。 必填。
    expires_at_ms: int | None。 必填。
    契约: 同步调用。 返回 `ProfileEnrollment`。
    """

    return ProfileEnrollment(
        profile_id=profile_id,
        preferred_name="小莓",
        encrypted_template=EncryptedVoiceTemplate(b"opaque"),
        consented=True,
        expires_at_ms=expires_at_ms,
    )


def _recognition(profile_id: VoiceProfileId) -> ProfileRecognition:
    """函数契约说明.

    功能: 执行 _recognition 的同步逻辑,并协调
    ProfileRecognition,
    RecognitionConfidence。
    参数: profile_id: VoiceProfileId。 必填。
    契约: 同步调用。 返回 `ProfileRecognition`。
    """

    return ProfileRecognition(profile_id, RecognitionConfidence(99))
