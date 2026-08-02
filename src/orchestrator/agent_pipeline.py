# ruff: noqa: C901, EM101, EM102, PLR0911, PLR2004, SIM102, TRY003
"""Reducer-validated, Brain-directed session agent pipeline.

This module deliberately keeps provider output and effect execution separate.
The model can *propose* a plan, but only :class:`AgentPlanReducer` can accept
it and expose effects to the runtime.  It is transport independent so voice
and comment ingress share exactly the same policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast, final

from orchestrator.json_boundary import JsonBoundaryError, JsonValue, parse_json_value


class AudienceSource(StrEnum):
    ASR = "asr"
    COMMENT = "comment"


class GateDecision(StrEnum):
    ACCEPT = "accept"
    DISCARD = "discard"


class PlanStage(StrEnum):
    INITIAL = "initial"
    FINAL = "final"


class PlanError(ValueError):
    """A model proposal was not a valid, bounded AgentPlan."""


@dataclass(frozen=True, slots=True)
class AudienceInput:
    session_id: str
    trace_id: str
    sequence: int
    source: AudienceSource
    received_at_ms: int
    text: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.trace_id or self.sequence < 0:
            raise ValueError("audience input correlation is invalid")
        if not self.text.strip() or len(self.text) > 4_000:
            raise ValueError("audience input text is invalid")


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    task_id: str
    kind: str
    lane: str
    status: str
    deadline_ms: int
    owner_turn_id: str
    cancellation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PlaybackSnapshot:
    status: str = "idle"
    position_ms: int = 0
    active_audio_id: str | None = None
    replacement_audio_id: str | None = None
    replacement_first_frame_ready: bool = False
    flush_accepted: bool = False


@dataclass(frozen=True, slots=True)
class BrainStateSnapshot:
    session_id: str
    turn_id: str
    revision: int
    cancellation_epoch: int
    input: AudienceInput
    context_summary: str
    recent_context: tuple[str, ...]
    memory_markdown: str
    capabilities: frozenset[str]
    tasks: tuple[TaskSnapshot, ...] = ()
    playback: PlaybackSnapshot = PlaybackSnapshot()
    frontend_caption: str = ""
    frontend_animation: str | None = None
    ppt_deck_id: str | None = None
    ppt_page: int | None = None
    context_revision: int = 0
    memory_revision: int = 0
    context_budget: int = 0
    compaction_required: bool = False
    knowledge_references: tuple[str, ...] = ()
    mcp_allowlist: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class StateOperation:
    kind: str
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class MediaOperation:
    kind: str
    audio_id: str | None = None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class FrontendOperation:
    kind: str
    value: str | int | None = None
    deck_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolRequest:
    kind: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolExecutionObservation:
    """A successful tool result, retained with its request provenance."""

    request: ToolRequest

    text: str


@dataclass(frozen=True, slots=True)
class AgentPlan:
    response_text: str
    expected_revision: int
    state_operations: tuple[StateOperation, ...] = ()
    media_operations: tuple[MediaOperation, ...] = ()
    frontend_operations: tuple[FrontendOperation, ...] = ()
    tool_requests: tuple[ToolRequest, ...] = ()
    citations: tuple[str, ...] = ()
    memory_patches: tuple[dict[str, object], ...] = ()

    @classmethod
    def from_json(cls, raw: str) -> AgentPlan:
        try:
            value = parse_json_value(raw)
        except JsonBoundaryError as error:
            raise PlanError("plan is not JSON") from error
        if not isinstance(value, dict):
            raise PlanError("plan must be an object")
        document = value
        allowed = {
            "response_text",
            "expected_revision",
            "state_operations",
            "media_operations",
            "frontend_operations",
            "tool_requests",
            "citations",
            "memory_patches",
        }
        if set(document) - allowed or {
            "response_text",
            "expected_revision",
        } - set(document):
            raise PlanError("plan fields are invalid")
        response = document["response_text"]
        revision = document["expected_revision"]
        if (
            not isinstance(response, str)
            or len(response) > 8_000
            or not isinstance(revision, int)
        ):
            raise PlanError("plan response or revision is invalid")
        return cls(
            response_text=response,
            expected_revision=revision,
            state_operations=_operations(
                document.get("state_operations", []), StateOperation
            ),
            media_operations=_media_operations(document.get("media_operations", [])),
            frontend_operations=_frontend_operations(
                document.get("frontend_operations", [])
            ),
            tool_requests=_tool_requests(document.get("tool_requests", [])),
            citations=_strings(document.get("citations", []), "citations"),
            memory_patches=tuple(
                dict(patch)
                for patch in _objects(
                    document.get("memory_patches", []), "memory_patches"
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanAccepted:
    plan: AgentPlan
    effects: tuple[object, ...]
    observations: tuple[ToolExecutionObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanRejected:
    reason: str


type PlanResult = PlanAccepted | PlanRejected


@final
class AgentPlanReducer:
    """Checks an untrusted plan before the runtime creates any effect."""

    _state_kinds = frozenset(
        {"create_task", "cancel_task", "context.compact", "memory.patch"}
    )
    _media_kinds = frozenset({"synthesize", "play", "stop", "replace_playback"})
    _frontend_kinds = frozenset({"caption", "animation", "ppt.load", "ppt.navigate"})
    _task_kinds = frozenset(
        {"tts", "playback", "retrieval", "mcp", "context_compaction", "memory_patch"}
    )

    def reduce(
        self, snapshot: BrainStateSnapshot, plan: AgentPlan, *, stage: PlanStage
    ) -> PlanResult:
        if plan.expected_revision != snapshot.revision:
            return PlanRejected("stale_revision")
        if stage is PlanStage.FINAL and plan.tool_requests:
            return PlanRejected("final_plan_requests_tools")
        if any(op.kind not in self._state_kinds for op in plan.state_operations):
            return PlanRejected("unsupported_state_operation")
        if any(op.kind not in self._media_kinds for op in plan.media_operations):
            return PlanRejected("unsupported_media_operation")
        if any(op.kind not in self._frontend_kinds for op in plan.frontend_operations):
            return PlanRejected("unsupported_frontend_operation")
        if any(
            not self._tool_allowed(snapshot, request) for request in plan.tool_requests
        ):
            return PlanRejected("tool_not_allowed")
        if not self._state_is_valid(snapshot, plan):
            return PlanRejected("invalid_state_operation")
        if not self._media_is_valid(snapshot, plan):
            return PlanRejected("unsafe_media_operation")
        if not self._frontend_is_valid(snapshot, plan):
            return PlanRejected("invalid_frontend_operation")
        if not self._memory_is_valid(snapshot, plan):
            return PlanRejected("invalid_memory_patch")
        return PlanAccepted(
            plan,
            (
                *plan.state_operations,
                *plan.media_operations,
                *plan.frontend_operations,
                *plan.tool_requests,
            ),
        )

    def _tool_allowed(self, snapshot: BrainStateSnapshot, request: ToolRequest) -> bool:
        if request.kind not in {"knowledge", "mcp"} or len(request.arguments) > 16:
            return False
        required = (
            "knowledge.lookup" if request.kind == "knowledge" else f"mcp:{request.name}"
        )
        return required in snapshot.capabilities and (
            request.kind == "knowledge" or request.name in snapshot.mcp_allowlist
        )

    def _state_is_valid(self, snapshot: BrainStateSnapshot, plan: AgentPlan) -> bool:
        for operation in plan.state_operations:
            if operation.kind == "create_task":
                kind = operation.payload.get("task_kind")
                if (
                    kind not in self._task_kinds
                    or f"task:{kind}" not in snapshot.capabilities
                ):
                    return False
            if operation.kind == "context.compact" and not snapshot.compaction_required:
                return False
            if operation.kind == "context.compact" and not isinstance(
                operation.payload.get("summary"), str
            ):
                return False
        return True

    @staticmethod
    def _media_is_valid(snapshot: BrainStateSnapshot, plan: AgentPlan) -> bool:
        for operation in plan.media_operations:
            if operation.kind == "replace_playback":
                # The reducer will not permit a flush/stop until ready.  It is
                # safe to queue a synthesis request while old audio continues.
                if not operation.audio_id and not operation.text:
                    return False
            if (
                operation.kind == "stop"
                and snapshot.playback.replacement_audio_id
                and not (
                    snapshot.playback.replacement_first_frame_ready
                    and snapshot.playback.flush_accepted
                )
            ):
                return False
        return True

    @staticmethod
    def _frontend_is_valid(snapshot: BrainStateSnapshot, plan: AgentPlan) -> bool:
        for operation in plan.frontend_operations:
            if operation.kind == "caption" and (
                not isinstance(operation.value, str) or not operation.value.strip()
            ):
                return False
            if operation.kind == "animation" and operation.value not in {
                "idle",
                "talk",
                "wave",
                "nod",
            }:
                return False
            if operation.kind == "ppt.load" and (
                not isinstance(operation.value, str) or not operation.value.strip()
            ):
                return False
            if operation.kind == "ppt.navigate" and (
                snapshot.ppt_deck_id is None
                or operation.deck_id != snapshot.ppt_deck_id
                or not isinstance(operation.value, int)
                or operation.value < 1
            ):
                return False
        return True

    @staticmethod
    def _memory_is_valid(snapshot: BrainStateSnapshot, plan: AgentPlan) -> bool:
        if len(plan.memory_patches) > 1:
            return False
        for patch in plan.memory_patches:
            if set(patch) - {
                "op",
                "id",
                "category",
                "value",
                "source_turn",
                "confidence",
                "base_revision",
            }:
                return False
            if patch.get("op") not in {"add", "update", "delete"} or not isinstance(
                patch.get("id"), str
            ):
                return False
            if patch.get("base_revision") != snapshot.memory_revision:
                return False
            if patch.get("source_turn") != snapshot.turn_id:
                return False
            if patch.get("op") != "delete" and (
                patch.get("category")
                not in {"preference", "session_fact", "voice_association"}
                or not isinstance(patch.get("value"), str)
            ):
                return False
        return True


class Gate(Protocol):
    def evaluate(
        self, audience_input: AudienceInput, *, active_summary: str
    ) -> GateDecision: ...


class Brain(Protocol):
    def plan(
        self, snapshot: BrainStateSnapshot, *, observations: tuple[str, ...] = ()
    ) -> str: ...

    def repair(self, snapshot: BrainStateSnapshot, invalid_plan: str) -> str: ...


class ToolExecutor(Protocol):
    def execute(
        self, request: ToolRequest, snapshot: BrainStateSnapshot
    ) -> str | None: ...


class AsyncGate(Protocol):
    async def evaluate(
        self, audience_input: AudienceInput, *, active_summary: str
    ) -> GateDecision: ...


class AsyncBrain(Protocol):
    async def plan(
        self, snapshot: BrainStateSnapshot, *, observations: tuple[str, ...] = ()
    ) -> str: ...

    async def repair(self, snapshot: BrainStateSnapshot, invalid_plan: str) -> str: ...


class AsyncToolExecutor(Protocol):
    async def execute(
        self, request: ToolRequest, snapshot: BrainStateSnapshot
    ) -> str | None: ...


@dataclass(slots=True)
class AgentPipeline:
    """One bounded, voice-priority scheduler for comments and ASR finals."""

    gate: Gate
    brain: Brain
    tools: ToolExecutor
    reducer: AgentPlanReducer = field(default_factory=AgentPlanReducer)
    comment_capacity: int = 16
    _voice: deque[AudienceInput] = field(default_factory=deque, init=False)
    _comments: deque[AudienceInput] = field(default_factory=deque, init=False)

    def submit(
        self, audience_input: AudienceInput, *, active_summary: str = ""
    ) -> GateDecision:
        decision = self.gate.evaluate(audience_input, active_summary=active_summary)
        if decision is GateDecision.DISCARD:
            return decision
        target = (
            self._voice
            if audience_input.source is AudienceSource.ASR
            else self._comments
        )
        if target is self._comments and len(target) >= self.comment_capacity:
            return GateDecision.DISCARD
        target.append(audience_input)
        return decision

    def next_input(self) -> AudienceInput | None:
        if self._voice:
            return self._voice.popleft()
        return self._comments.popleft() if self._comments else None

    def peek_input(self) -> AudienceInput | None:
        if self._voice:
            return self._voice[0]
        return self._comments[0] if self._comments else None

    def run(self, snapshot: BrainStateSnapshot) -> PlanResult:
        if snapshot.input != self.peek_input():
            return PlanRejected("input_not_scheduled")
        _ = self.next_input()
        raw = self.brain.plan(snapshot)
        result = self._reduce_or_repair(snapshot, raw, PlanStage.INITIAL)
        if not isinstance(result, PlanAccepted) or not result.plan.tool_requests:
            return result
        observations = tuple(
            ToolExecutionObservation(request, observation)
            for request in result.plan.tool_requests
            if (observation := self.tools.execute(request, snapshot)) is not None
        )
        observation_texts = tuple(observation.text for observation in observations)
        raw_final = self.brain.plan(snapshot, observations=observation_texts)
        final = self._reduce_or_repair(snapshot, raw_final, PlanStage.FINAL)
        if isinstance(final, PlanAccepted):
            return PlanAccepted(final.plan, final.effects, observations)
        return final

    def _reduce_or_repair(
        self, snapshot: BrainStateSnapshot, raw: str, stage: PlanStage
    ) -> PlanResult:
        try:
            plan = AgentPlan.from_json(raw)
        except PlanError:
            return self._repair_and_reduce(snapshot, raw, stage)
        result = self.reducer.reduce(snapshot, plan, stage=stage)
        if isinstance(result, PlanAccepted):
            return result
        return self._repair_and_reduce(snapshot, raw, stage)

    def _repair_and_reduce(
        self, snapshot: BrainStateSnapshot, raw: str, stage: PlanStage
    ) -> PlanResult:
        try:
            repaired = AgentPlan.from_json(self.brain.repair(snapshot, raw))
        except PlanError:
            return PlanRejected("plan_parse_failed")
        return self.reducer.reduce(snapshot, repaired, stage=stage)


@dataclass(slots=True)
class AsyncAgentPipeline:
    """Async equivalent used by the live OpenAI-compatible Brain runtime."""

    gate: AsyncGate
    brain: AsyncBrain
    tools: AsyncToolExecutor
    reducer: AgentPlanReducer = field(default_factory=AgentPlanReducer)
    comment_capacity: int = 16
    _voice: deque[AudienceInput] = field(default_factory=deque, init=False)
    _comments: deque[AudienceInput] = field(default_factory=deque, init=False)

    async def submit(
        self, audience_input: AudienceInput, *, active_summary: str = ""
    ) -> GateDecision:
        decision = await self.gate.evaluate(
            audience_input, active_summary=active_summary
        )
        if decision is GateDecision.DISCARD:
            return decision
        target = (
            self._voice
            if audience_input.source is AudienceSource.ASR
            else self._comments
        )
        if target is self._comments and len(target) >= self.comment_capacity:
            return GateDecision.DISCARD
        target.append(audience_input)
        return decision

    def peek_input(self) -> AudienceInput | None:
        if self._voice:
            return self._voice[0]
        return self._comments[0] if self._comments else None

    def next_input(self) -> AudienceInput | None:
        if self._voice:
            return self._voice.popleft()
        return self._comments.popleft() if self._comments else None

    async def run(self, snapshot: BrainStateSnapshot) -> PlanResult:
        if snapshot.input != self.peek_input():
            return PlanRejected("input_not_scheduled")
        _ = self.next_input()
        try:
            raw = await self.brain.plan(snapshot)
        except OSError:
            return PlanRejected("brain_unavailable")
        result = await self._reduce_or_repair(snapshot, raw, PlanStage.INITIAL)
        if not isinstance(result, PlanAccepted) or not result.plan.tool_requests:
            return result
        observations: list[ToolExecutionObservation] = []
        for request in result.plan.tool_requests:
            try:
                observation = await self.tools.execute(request, snapshot)
            except (OSError, TimeoutError, ValueError):
                observation = None
            if observation is not None:
                observations.append(ToolExecutionObservation(request, observation))
        try:
            raw_final = await self.brain.plan(
                snapshot, observations=tuple(item.text for item in observations)
            )
        except OSError:
            return PlanRejected("brain_unavailable")
        final = await self._reduce_or_repair(snapshot, raw_final, PlanStage.FINAL)
        if isinstance(final, PlanAccepted):
            return PlanAccepted(final.plan, final.effects, tuple(observations))
        return final

    async def _reduce_or_repair(
        self, snapshot: BrainStateSnapshot, raw: str, stage: PlanStage
    ) -> PlanResult:
        try:
            plan = AgentPlan.from_json(raw)
        except PlanError:
            return await self._repair_and_reduce(snapshot, raw, stage)
        result = self.reducer.reduce(snapshot, plan, stage=stage)
        if isinstance(result, PlanAccepted):
            return result
        return await self._repair_and_reduce(snapshot, raw, stage)

    async def _repair_and_reduce(
        self, snapshot: BrainStateSnapshot, raw: str, stage: PlanStage
    ) -> PlanResult:
        try:
            repaired = AgentPlan.from_json(await self.brain.repair(snapshot, raw))
        except (PlanError, OSError):
            return PlanRejected("plan_parse_failed")
        return self.reducer.reduce(snapshot, repaired, stage=stage)


def _strings(value: JsonValue, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PlanError(f"{field} is invalid")
    strings = tuple(item for item in value if isinstance(item, str))
    if len(strings) != len(value):
        raise PlanError(f"{field} is invalid")
    return strings


def _objects(value: JsonValue, field: str) -> tuple[dict[str, JsonValue], ...]:
    if not isinstance(value, list):
        raise PlanError(f"{field} is invalid")
    objects = tuple(item for item in value if isinstance(item, dict))
    if len(objects) != len(value):
        raise PlanError(f"{field} is invalid")
    return objects


def _operations(
    value: JsonValue, operation_type: type[StateOperation]
) -> tuple[StateOperation, ...]:
    objects = _objects(value, "operations")
    operations: list[StateOperation] = []
    for raw_item in objects:
        item = dict(raw_item)
        kind = item.pop("kind", None)
        payload = item.pop("payload", {})
        if not isinstance(kind, str) or not isinstance(payload, dict) or item:
            raise PlanError("state operation is invalid")
        operations.append(operation_type(kind, dict(payload)))
    return tuple(operations)


def _media_operations(value: JsonValue) -> tuple[MediaOperation, ...]:
    result: list[MediaOperation] = []
    for item in _objects(value, "media_operations"):
        if set(item) - {"kind", "audio_id", "text"} or not isinstance(
            item.get("kind"), str
        ):
            raise PlanError("media operation is invalid")
        audio_id, text = item.get("audio_id"), item.get("text")
        if (audio_id is not None and not isinstance(audio_id, str)) or (
            text is not None and not isinstance(text, str)
        ):
            raise PlanError("media operation payload is invalid")
        result.append(MediaOperation(cast("str", item["kind"]), audio_id, text))
    return tuple(result)


def _frontend_operations(value: JsonValue) -> tuple[FrontendOperation, ...]:
    result: list[FrontendOperation] = []
    for item in _objects(value, "frontend_operations"):
        if set(item) - {"kind", "value", "deck_id"} or not isinstance(
            item.get("kind"), str
        ):
            raise PlanError("frontend operation is invalid")
        item_value = item.get("value")
        deck_id = item.get("deck_id")
        if item_value is not None and (
            not isinstance(item_value, str) and type(item_value) is not int
        ):
            raise PlanError("frontend operation payload is invalid")
        if deck_id is not None and not isinstance(deck_id, str):
            raise PlanError("frontend operation deck id is invalid")
        result.append(
            FrontendOperation(cast("str", item["kind"]), item_value, deck_id)
        )
    return tuple(result)


def _tool_requests(value: JsonValue) -> tuple[ToolRequest, ...]:
    result: list[ToolRequest] = []
    for item in _objects(value, "tool_requests"):
        if (
            set(item) != {"kind", "name", "arguments"}
            or not isinstance(item["kind"], str)
            or not isinstance(item["name"], str)
            or not isinstance(item["arguments"], dict)
        ):
            raise PlanError("tool request is invalid")
        result.append(ToolRequest(item["kind"], item["name"], dict(item["arguments"])))
    return tuple(result)
