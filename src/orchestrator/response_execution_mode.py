"""Session-scoped execution mode for the minimal response pipeline."""

from enum import StrEnum, unique


@unique
class ResponseExecutionMode(StrEnum):
    NEW_SHADOW = "new_shadow"
    NEW_EXECUTE = "new_execute"
