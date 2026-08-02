import json
from dataclasses import dataclass, replace

from orchestrator.agent_pipeline import (
    AgentPipeline,
    AgentPlan,
    AgentPlanReducer,
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
    GateDecision,
    PlanAccepted,
    PlanRejected,
    PlanStage,
    PlaybackSnapshot,
)


@dataclass
class _Gate:
    discarded: set[str]

    def evaluate(
        self, audience_input: AudienceInput, *, active_summary: str
    ) -> GateDecision:
        _ = active_summary
        return (
            GateDecision.DISCARD
            if audience_input.text in self.discarded
            else GateDecision.ACCEPT
        )


@dataclass
class _Brain:
    initial: str
    final: str | None = None
    repaired: str | None = None
    calls: int = 0

    def plan(
        self, snapshot: BrainStateSnapshot, *, observations: tuple[str, ...] = ()
    ) -> str:
        _ = snapshot
        self.calls += 1
        return self.initial if not observations or self.final is None else self.final

    def repair(self, snapshot: BrainStateSnapshot, invalid_plan: str) -> str:
        _ = snapshot, invalid_plan
        assert self.repaired is not None
        return self.repaired


class _Tools:
    def execute(self, request: object, snapshot: BrainStateSnapshot) -> str:
        _ = request, snapshot
        return "可信执行观察: 查到一条引用"


def _input(source: AudienceSource, text: str, sequence: int) -> AudienceInput:
    return AudienceInput("s-1", "trace-1", sequence, source, 1, text)


def _snapshot(
    audience_input: AudienceInput,
    playback: PlaybackSnapshot | None = None,
) -> BrainStateSnapshot:
    return BrainStateSnapshot(
        session_id="s-1",
        turn_id="turn-1",
        revision=3,
        cancellation_epoch=0,
        input=audience_input,
        context_summary="",
        recent_context=(),
        memory_markdown="# memory\n",
        capabilities=frozenset({"knowledge.lookup", "task:tts"}),
        playback=PlaybackSnapshot() if playback is None else playback,
    )


def _plan(**overrides: object) -> str:
    value: dict[str, object] = {
        "response_text": "您好",
        "expected_revision": 3,
        "state_operations": [],
        "media_operations": [],
        "frontend_operations": [],
        "tool_requests": [],
        "citations": [],
        "memory_patches": [],
    }
    value.update(overrides)
    return json.dumps(value)


def test_gate_discards_before_queue_and_voice_has_priority() -> None:
    brain = _Brain(_plan())
    pipeline = AgentPipeline(_Gate({"echo"}), brain, _Tools(), comment_capacity=1)
    assert (
        pipeline.submit(_input(AudienceSource.COMMENT, "echo", 0))
        is GateDecision.DISCARD
    )
    assert (
        pipeline.submit(_input(AudienceSource.COMMENT, "comment", 1))
        is GateDecision.ACCEPT
    )
    assert (
        pipeline.submit(_input(AudienceSource.ASR, "voice", 2)) is GateDecision.ACCEPT
    )
    voice = pipeline.next_input()
    comment = pipeline.next_input()
    assert voice is not None
    assert comment is not None
    assert voice.text == "voice"
    assert comment.text == "comment"


def test_invalid_json_is_repaired_once_without_effect_from_bad_plan() -> None:
    audience_input = _input(AudienceSource.ASR, "question", 0)
    pipeline = AgentPipeline(
        _Gate(set()), _Brain("not json", repaired=_plan()), _Tools()
    )
    assert pipeline.submit(audience_input) is GateDecision.ACCEPT
    result = pipeline.run(_snapshot(audience_input))
    assert isinstance(result, PlanAccepted)
    assert result.effects == ()


def test_task_capability_name_is_not_a_state_operation() -> None:
    audience_input = _input(AudienceSource.ASR, "question", 0)
    plan = _plan(
        state_operations=[{"kind": "task:tts", "payload": {"text": "您好"}}]
    )

    assert AgentPlanReducer().reduce(
        _snapshot(audience_input), AgentPlan.from_json(plan), stage=PlanStage.INITIAL
    ) == PlanRejected("unsupported_state_operation")


def test_semantically_invalid_plan_is_repaired_once_before_effects() -> None:
    audience_input = _input(AudienceSource.ASR, "question", 0)
    invalid = _plan(
        state_operations=[{"kind": "context.compact", "payload": {"summary": "x"}}]
    )
    repaired = _plan(
        state_operations=[{"kind": "create_task", "payload": {"task_kind": "tts"}}]
    )
    brain = _Brain(invalid, repaired=repaired)
    pipeline = AgentPipeline(_Gate(set()), brain, _Tools())

    assert pipeline.submit(audience_input) is GateDecision.ACCEPT
    assert isinstance(pipeline.run(_snapshot(audience_input)), PlanAccepted)
    assert brain.calls == 1


def test_tool_plan_requires_a_final_plan_with_no_new_tool_request() -> None:
    audience_input = _input(AudienceSource.ASR, "question", 0)
    initial = _plan(
        tool_requests=[
            {"kind": "knowledge", "name": "local", "arguments": {"query": "q"}}
        ]
    )
    pipeline = AgentPipeline(_Gate(set()), _Brain(initial, final=_plan()), _Tools())
    assert pipeline.submit(audience_input) is GateDecision.ACCEPT
    result = pipeline.run(_snapshot(audience_input))
    assert isinstance(result, PlanAccepted)
    assert tuple(observation.text for observation in result.observations) == (
        "可信执行观察: 查到一条引用",
    )


def test_final_plan_cannot_request_tools() -> None:
    audience_input = _input(AudienceSource.ASR, "question", 0)
    tool_request: dict[str, object] = {
        "kind": "knowledge",
        "name": "local",
        "arguments": {},
    }
    initial = _plan(
        tool_requests=[tool_request]
    )
    pipeline = AgentPipeline(
        _Gate(set()), _Brain(initial, final=initial, repaired=initial), _Tools()
    )
    assert pipeline.submit(audience_input) is GateDecision.ACCEPT
    result = pipeline.run(_snapshot(audience_input))
    assert result == PlanRejected("final_plan_requests_tools")


def test_stop_is_rejected_until_replacement_has_frame_and_flush_ack() -> None:
    audience_input = _input(AudienceSource.ASR, "replace", 0)
    reducer = AgentPlanReducer()
    plan = _plan(media_operations=[{"kind": "stop"}])
    parsed = AgentPlan.from_json(plan)
    result = reducer.reduce(
        _snapshot(
            audience_input,
            PlaybackSnapshot(
                status="playing",
                replacement_audio_id="next",
                replacement_first_frame_ready=False,
            ),
        ),
        parsed,
        stage=PlanStage.FINAL,
    )
    assert result == PlanRejected("unsafe_media_operation")


def test_ppt_navigation_requires_the_snapshot_deck_id() -> None:
    audience_input = _input(AudienceSource.COMMENT, "下一页", 0)
    snapshot = replace(_snapshot(audience_input), ppt_deck_id="deck-1", ppt_page=1)
    plan = _plan(
        frontend_operations=[
            {"kind": "ppt.navigate", "value": 2, "deck_id": "other-deck"}
        ]
    )

    assert AgentPlanReducer().reduce(
        snapshot, AgentPlan.from_json(plan), stage=PlanStage.INITIAL
    ) == PlanRejected("invalid_frontend_operation")
