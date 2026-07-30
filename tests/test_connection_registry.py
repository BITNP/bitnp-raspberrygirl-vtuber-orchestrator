"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import pytest

from orchestrator.ids import ConnectionId
from orchestrator.registry import (
    ConnectionRegistry,
    DuplicateModuleIdentityError,
    ModuleIdentity,
    ModuleIdentityParseError,
)


def test_register_rejects_duplicate_active_module_identity() -> None:
    # Given: one active ASR module identity is already registered.

    """函数契约说明.

    功能: 验证 register rejects duplicate
    active module identity 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = ConnectionRegistry()

    identity = ModuleIdentity.parse(module_name="asr", instance_id="asr-local")

    registered = registry.register(identity, ConnectionId("conn-001"))

    # When: another active connection claims the same module identity.

    with pytest.raises(DuplicateModuleIdentityError) as error:
        _ = registry.register(identity, ConnectionId("conn-002"))

    # Then: the duplicate is rejected deterministically and the original remains.

    assert registered.connection_id == ConnectionId("conn-001")

    assert str(error.value) == (
        "active module identity already registered: asr/asr-local"
    )

    assert registry.active_identities() == (identity,)


def test_register_allows_same_module_name_with_distinct_instance() -> None:
    # Given: two instances of the same module type have distinct identities.

    """函数契约说明.

    功能: 验证 register allows same module
    name with distinct instance
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    registry = ConnectionRegistry()

    first = ModuleIdentity.parse(module_name="comments", instance_id="room-a")

    second = ModuleIdentity.parse(module_name="comments", instance_id="room-b")

    # When: both identities register active connections.

    first_registered = registry.register(first, ConnectionId("conn-001"))

    second_registered = registry.register(second, ConnectionId("conn-002"))

    # Then: both active identities are retained in deterministic order.

    assert first_registered.identity == first

    assert second_registered.identity == second

    assert registry.active_identities() == (first, second)


def test_module_identity_rejects_blank_fields() -> None:
    # Given: malformed module identity input from a future network boundary.

    # When: parsing attempts to create a typed identity.

    """函数契约说明.

    功能: 验证 module identity rejects blank
    fields 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    with pytest.raises(ModuleIdentityParseError) as error:
        _ = ModuleIdentity.parse(module_name="tts", instance_id=" ")

    # Then: parsing fails before the registry sees untrusted identity data.

    assert str(error.value) == "module identity field is blank: instance_id"
