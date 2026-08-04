"""Session-scoped migration mode for the response pipeline."""

from enum import StrEnum, unique


@unique
class ResponseExecutionMode(StrEnum):
    LEGACY_EXECUTE = "legacy_execute"
    NEW_SHADOW = "new_shadow"
    NEW_EXECUTE = "new_execute"
