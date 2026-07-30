"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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

    """函数契约说明.

    功能: 验证 persisted memory reloads
    across session data restart
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 unconfirmed expired and
    revoked profiles never personalize
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 profile lifecycle metadata
    reloads without exposing template
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

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

    """函数契约说明.

    功能: 验证 revocation and deletion
    remain unknown after restart and
    invalidate work 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

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
    """函数契约说明.

    功能: 执行 _state 的同步逻辑,并协调 create,
    SessionId, RetrievalFixtureProvider,
    ProfilePersistence。
    参数: path: Path | None。 必填。
    profile_path: Path | None。 可省略。
    vault_directory: Path | None。 可省略。
    契约: 同步调用。 返回 `SessionDataState`。
    """

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
    """函数契约说明.

    功能: 执行 _proposal 的同步逻辑,并协调
    MemoryProposal, MemoryKey,
    MemoryConfidence, ProposalRevision。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `MemoryProposal`。
    """

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
