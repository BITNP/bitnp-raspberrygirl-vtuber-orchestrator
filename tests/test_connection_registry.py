
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


    with pytest.raises(ModuleIdentityParseError) as error:
        _ = ModuleIdentity.parse(module_name="tts", instance_id=" ")

    # Then: parsing fails before the registry sees untrusted identity data.

    assert str(error.value) == "module identity field is blank: instance_id"
