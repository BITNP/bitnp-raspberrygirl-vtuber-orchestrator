
from dataclasses import dataclass
from typing import override

from orchestrator.ids import ConnectionId


@dataclass(frozen=True, slots=True, order=True)
class ModuleIdentity:

    module_name: str

    instance_id: str

    @classmethod
    def parse(cls, *, module_name: str, instance_id: str) -> "ModuleIdentity":
        cleaned_module_name = module_name.strip()

        cleaned_instance_id = instance_id.strip()

        if cleaned_module_name == "":
            raise ModuleIdentityParseError(field_name="module_name")

        if cleaned_instance_id == "":
            raise ModuleIdentityParseError(field_name="instance_id")

        return cls(module_name=cleaned_module_name, instance_id=cleaned_instance_id)

    def label(self) -> str:
        return f"{self.module_name}/{self.instance_id}"


@dataclass(frozen=True, slots=True)
class RegisteredConnection:

    identity: ModuleIdentity

    connection_id: ConnectionId


@dataclass(frozen=True, slots=True)
class ModuleIdentityParseError(Exception):

    field_name: str

    @override
    def __str__(self) -> str:
        return f"module identity field is blank: {self.field_name}"


@dataclass(frozen=True, slots=True)
class DuplicateModuleIdentityError(Exception):

    identity: ModuleIdentity

    @override
    def __str__(self) -> str:
        return f"active module identity already registered: {self.identity.label()}"


class ConnectionRegistry:

    def __init__(self) -> None:
        self._active: dict[ModuleIdentity, RegisteredConnection] = {}

    def register(
        self,
        identity: ModuleIdentity,
        connection_id: ConnectionId,
    ) -> RegisteredConnection:
        if identity in self._active:
            raise DuplicateModuleIdentityError(identity=identity)

        registered = RegisteredConnection(
            identity=identity,
            connection_id=connection_id,
        )

        self._active[identity] = registered

        return registered

    def active_identities(self) -> tuple[ModuleIdentity, ...]:
        return tuple(sorted(self._active))
