"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from pathlib import Path

import pytest

from orchestrator.ids import SessionId
from orchestrator.profile_store import JsonVoiceProfileStore, ProfileStoreBoundaryError


def test_profile_store_rejects_malformed_durable_metadata(tmp_path: Path) -> None:
    # Given: a durable record with a malformed profile lifecycle field.

    """函数契约说明.

    功能: 验证 profile store rejects
    malformed durable metadata
    的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    path = tmp_path / "voice-profiles.json"

    _ = path.write_text(
        """{

  "session_id": "session-1",

  "profile_revision": 1,

  "consent_revision": 1,

  "profiles": [{

    "profile_id": "profile-1",

    "preferred_name": "小莓",

    "purpose": "personalization",

    "confirmed": true,

    "expires_at_ms": null,

    "lifecycle": "unknown",

    "revision": 1,

    "audit": []

  }]

}

""",
        encoding="utf-8",
    )

    store = JsonVoiceProfileStore(path)

    # When: production hydration parses the untrusted persisted document.

    with pytest.raises(ProfileStoreBoundaryError) as error:
        _ = store.load(SessionId("session-1"))

    # Then: malformed metadata is refused before it can enable recognition.

    assert error.value.field_name == "profiles[0].lifecycle"
