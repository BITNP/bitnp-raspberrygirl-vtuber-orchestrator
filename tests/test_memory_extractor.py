from orchestrator.memory_extractor import parse_memory_candidate


def test_memory_extractor_accepts_only_one_bounded_preference_candidate() -> None:
    candidate = parse_memory_candidate(
        '{"key":"preferred_name","value":"小莓","confidence":95}'
    )
    assert candidate is not None
    assert candidate.key == "preferred_name"
    assert candidate.confidence == 95


def test_memory_extractor_discards_malformed_or_extra_model_fields() -> None:
    assert parse_memory_candidate("not json") is None
    assert (
        parse_memory_candidate(
            '{"key":"name","value":"小莓","confidence":95,"action":"write"}'
        )
        is None
    )
