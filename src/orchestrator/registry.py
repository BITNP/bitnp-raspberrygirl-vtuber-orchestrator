"""Active module connection registry for the Orchestrator hub."""

from dataclasses import dataclass
from typing import override

from orchestrator.ids import ConnectionId


@dataclass(frozen=True, slots=True, order=True)
class ModuleIdentity:
    """Stable identity claimed by one connected module instance."""

    module_name: str
    instance_id: str

    @classmethod
    def parse(cls, *, module_name: str, instance_id: str) -> "ModuleIdentity":
        """Parse boundary identity strings into a normalized value."""
        cleaned_module_name = module_name.strip()
        cleaned_instance_id = instance_id.strip()
        if cleaned_module_name == "":
            raise ModuleIdentityParseError(field_name="module_name")
        if cleaned_instance_id == "":
            raise ModuleIdentityParseError(field_name="instance_id")
        return cls(module_name=cleaned_module_name, instance_id=cleaned_instance_id)

    def label(self) -> str:
        """Return a deterministic human-readable identity label."""
        return f"{self.module_name}/{self.instance_id}"


@dataclass(frozen=True, slots=True)
class RegisteredConnection:
    """Connection record retained while a module is active."""

    identity: ModuleIdentity
    connection_id: ConnectionId


@dataclass(frozen=True, slots=True)
class ModuleIdentityParseError(Exception):
    """Raised when a module identity field is blank."""

    field_name: str

    @override
    def __str__(self) -> str:
        return f"module identity field is blank: {self.field_name}"


@dataclass(frozen=True, slots=True)
class DuplicateModuleIdentityError(Exception):
    """Raised when a live module identity is registered twice."""

    identity: ModuleIdentity

    @override
    def __str__(self) -> str:
        return f"active module identity already registered: {self.identity.label()}"


class ConnectionRegistry:
    """Tracks active module identities without owning peer dependencies."""

    def __init__(self) -> None:
        """Create an empty active-connection registry."""
        self._active: dict[ModuleIdentity, RegisteredConnection] = {}

    def register(
        self,
        identity: ModuleIdentity,
        connection_id: ConnectionId,
    ) -> RegisteredConnection:
        """Register a module identity if no active connection owns it."""
        if identity in self._active:
            raise DuplicateModuleIdentityError(identity=identity)
        registered = RegisteredConnection(
            identity=identity,
            connection_id=connection_id,
        )
        self._active[identity] = registered
        return registered

    def active_identities(self) -> tuple[ModuleIdentity, ...]:
        """Return active identities in deterministic order."""
        return tuple(sorted(self._active))
