"""Trusted mapping from small model intents to bounded tool arguments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import final

from orchestrator.agent_pipeline import BrainStateSnapshot, ToolRequest

type ArgumentBuilder = Callable[[BrainStateSnapshot], dict[str, object] | None]


@dataclass(frozen=True, slots=True)
class IntentSpec:
    intent_id: str
    tool_kind: str
    tool_name: str
    required_capability: str
    build_arguments: ArgumentBuilder

    def available(self, snapshot: BrainStateSnapshot) -> bool:
        return self.required_capability in snapshot.capabilities


@final
class IntentRouter:
    def __init__(self, specs: tuple[IntentSpec, ...]) -> None:
        self._specs = {spec.intent_id: spec for spec in specs}
        if len(self._specs) != len(specs) or "answer" in self._specs:
            raise ValueError

    def allowed_intents(self, snapshot: BrainStateSnapshot) -> frozenset[str]:
        enabled = {
            spec.intent_id for spec in self._specs.values() if spec.available(snapshot)
        }
        return frozenset({"answer", *enabled})

    def request(
        self, intent_id: str, snapshot: BrainStateSnapshot
    ) -> ToolRequest | None:
        spec = self._specs.get(intent_id)
        if spec is None or not spec.available(snapshot):
            return None
        arguments = spec.build_arguments(snapshot)
        if arguments is None:
            return None
        return ToolRequest(spec.tool_kind, spec.tool_name, arguments)
