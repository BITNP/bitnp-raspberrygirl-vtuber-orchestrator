import json

from orchestrator.agent_pipeline import (
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
)
from orchestrator.intent_router import IntentRouter, IntentSpec
from orchestrator.response_contracts import (
    CueKind,
    parse_inline_cues,
    parse_response_proposal,
)


def _snapshot(*, capabilities: frozenset[str] | None = None) -> BrainStateSnapshot:
    return BrainStateSnapshot(
        session_id="session-1",
        turn_id="turn-1",
        revision=1,
        cancellation_epoch=0,
        input=AudienceInput("session-1", "trace-1", 1, AudienceSource.ASR, 1, "查询"),
        context_summary="",
        recent_context=(),
        memory_markdown="",
        capabilities=frozenset() if capabilities is None else capabilities,
    )


def test_response_proposal_uses_plain_text_fallback_without_repair() -> None:
    proposal = parse_response_proposal(
        "不是 JSON", allowed_intents=frozenset({"answer"})
    )
    assert proposal.reply == "不是 JSON"
    assert proposal.intent == "answer"
    assert proposal.used_text_fallback


def test_response_proposal_only_accepts_two_allowlisted_fields() -> None:
    valid = parse_response_proposal(
        json.dumps({"reply": "您好", "intent": "knowledge"}),
        allowed_intents=frozenset({"answer", "knowledge"}),
    )
    invalid = parse_response_proposal(
        json.dumps({"reply": "您好", "intent": "unknown", "extra": True}),
        allowed_intents=frozenset({"answer", "knowledge"}),
    )
    assert valid.intent == "knowledge"
    assert not valid.used_text_fallback
    assert invalid.intent == "answer"
    assert invalid.used_text_fallback


def test_inline_cues_are_stripped_for_tts_and_keep_clean_offsets() -> None:
    result = parse_inline_cues(
        '欢迎<action name="wave"/>大家<expression name="happy"/>!',
        allowed_actions=frozenset({"wave"}),
        allowed_expressions=frozenset({"happy"}),
    )
    assert result.spoken_text == "欢迎大家!"
    assert result.marked_text == (
        '欢迎<action name="wave"/>大家<expression name="happy"/>!'
    )
    assert result.cues[0].kind is CueKind.ACTION
    assert result.cues[0].text_offset == 2
    assert result.cues[1].kind is CueKind.EXPRESSION
    assert result.cues[1].text_offset == 4


def test_invalid_or_unallowlisted_control_tags_never_reach_tts_or_timeline() -> None:
    result = parse_inline_cues(
        '好<action name="dance"/><action nope="x"/>。',
        allowed_actions=frozenset({"wave"}),
        allowed_expressions=frozenset(),
    )
    assert result.spoken_text == "好。"
    assert result.marked_text == "好。"
    assert result.cues == ()
    assert result.rejected_cues == 1


def test_intent_router_builds_arguments_from_trusted_snapshot_only() -> None:
    router = IntentRouter(
        (
            IntentSpec(
                "knowledge",
                "knowledge",
                "local",
                "knowledge.lookup",
                lambda snapshot: {"query": snapshot.input.text},
            ),
        )
    )
    snapshot = _snapshot(capabilities=frozenset({"knowledge.lookup"}))
    assert router.allowed_intents(snapshot) == frozenset({"answer", "knowledge"})
    request = router.request("knowledge", snapshot)
    assert request is not None
    assert request.arguments == {"query": "查询"}
