import json

import pytest

from orchestrator.brain_contracts import (
    AudienceInput,
    AudienceSource,
    BrainStateSnapshot,
)
from orchestrator.intent_router import IntentRouter, IntentSpec
from orchestrator.response_contracts import (
    BrainDecision,
    CueKind,
    OperationProposal,
    parse_final_speech_proposal,
    parse_inline_cues,
    parse_response_proposal,
)


def _snapshot() -> BrainStateSnapshot:
    return BrainStateSnapshot(
        "session-1",
        "candidate-1",
        1,
        0,
        AudienceInput("session-1", "trace-1", 1, AudienceSource.ASR, 1, "查询天气"),
        "",
        (),
        "",
        frozenset({"mcp:web/search"}),
    )


@pytest.mark.parametrize(
    ("payload", "decision", "has_operation"),
    [
        (
            {"decision": "discard", "speech": "", "operation": None},
            BrainDecision.DISCARD,
            False,
        ),
        (
            {"decision": "accept", "speech": "您好", "operation": None},
            BrainDecision.ACCEPT,
            False,
        ),
        (
            {
                "decision": "accept",
                "speech": "我来查询。",
                "operation": {
                    "intent": "mcp.web_search",
                    "arguments": {"query": "上海天气"},
                },
            },
            BrainDecision.ACCEPT,
            True,
        ),
    ],
)
def test_strict_brain_proposal_variants(
    payload: dict[str, object], decision: BrainDecision, has_operation: bool
) -> None:
    proposal = parse_response_proposal(json.dumps(payload, ensure_ascii=False))
    assert proposal is not None
    assert proposal.decision is decision
    assert (proposal.operation is not None) is has_operation


INVALID_PAYLOADS: list[object] = [
        "not-json",
        {"decision": "discard", "speech": "不该说", "operation": None},
        {
            "decision": "discard",
            "speech": "",
            "operation": {"intent": "x", "arguments": {}},
        },
        {"decision": "accept", "speech": "", "operation": None},
        {"decision": "accept", "speech": "好", "operation": []},
        {
            "decision": "accept",
            "speech": "好",
            "operation": [{"intent": "x", "arguments": {}}],
        },
        {
            "decision": "accept",
            "speech": "好",
            "operation": {"intent": "x", "arguments": {}, "extra": 1},
        },
]


@pytest.mark.parametrize("payload", INVALID_PAYLOADS)
def test_invalid_proposals_have_no_text_fallback(payload: object) -> None:
    raw = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )
    assert parse_response_proposal(raw) is None


def test_final_response_cannot_request_another_operation() -> None:
    raw = json.dumps(
        {
            "decision": "accept",
            "speech": "结果",
            "operation": {"intent": "again", "arguments": {}},
        }
    )
    assert parse_final_speech_proposal(raw) is None


def test_speech_cues_are_separated_from_tts_text() -> None:
    result = parse_inline_cues(
        '欢迎<action name="hello"/>大家<expression name="happy"/>!',
        allowed_actions=frozenset({"hello"}),
        allowed_expressions=frozenset({"happy"}),
    )
    assert result.spoken_text == "欢迎大家!"
    assert result.cues[0].kind is CueKind.ACTION
    assert result.cues[1].kind is CueKind.EXPRESSION


def test_router_keeps_speech_and_validated_query_separate() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 64}},
    }
    router = IntentRouter(
        (IntentSpec("mcp.web_search", "mcp", "web/search", "mcp:web/search", schema),)
    )
    request = router.request(
        OperationProposal("mcp.web_search", {"query": "上海天气"}), _snapshot()
    )
    assert request is not None
    assert request.arguments == {"query": "上海天气"}
    assert (
        router.request(
            OperationProposal("mcp.web_search", {"query": "x", "extra": "speech"}),
            _snapshot(),
        )
        is None
    )
