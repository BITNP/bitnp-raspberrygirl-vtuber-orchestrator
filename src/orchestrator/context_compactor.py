"""Maintenance-only context compaction proposal boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from orchestrator.transient_context import ContextComposition


class AsyncContextCompactor(Protocol):
    """Provider may propose one Chinese summary for an immutable composition."""

    async def compact(self, composition: ContextComposition) -> str | None: ...
