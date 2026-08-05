import pytest

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value


def test_json_boundary_accepts_standard_exponent_notation() -> None:
    assert parse_json_value('{"value":1.25e2}') == {"value": 125.0}


@pytest.mark.parametrize(
    "payload",
    [
        '{"key":1,"key":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '"unescaped\x01control"',
    ],
)
def test_json_boundary_rejects_ambiguous_or_noncanonical_values(payload: str) -> None:
    with pytest.raises(JsonBoundaryError):
        _ = parse_json_value(payload)


def test_json_boundary_rejects_excessive_nesting() -> None:
    payload = "[" * 34 + "null" + "]" * 34

    with pytest.raises(JsonBoundaryError, match="nesting"):
        _ = parse_json_value(payload)
