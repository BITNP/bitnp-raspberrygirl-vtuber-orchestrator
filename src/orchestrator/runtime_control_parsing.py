"""Pure parsing for frontend presentation-result control input."""

from orchestrator.ids import SessionId, TraceId
from orchestrator.interactions import CommandId, PresentationResult
from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.sessions import EventCorrelation, EventSequence


def parse_presentation_result_control(
    raw_message: str, expected_session_id: SessionId
) -> tuple[PresentationResult, EventCorrelation] | None:
    """Parse a frontend acknowledgement bound to the expected session."""
    try:
        value = parse_json_value(raw_message)
    except JsonBoundaryError:
        return None

    if (
        not isinstance(value, dict)
        or value.get("event_type") != "presentation.result"
    ):
        return None

    data = value.get("data")
    trace_id = value.get("trace_id")
    session_id = value.get("session_id")
    sequence = value.get("seq")
    command_id = data.get("command_id") if isinstance(data, dict) else None
    succeeded = data.get("succeeded") if isinstance(data, dict) else None

    if (
        value.get("source") != "frontend"
        or not isinstance(trace_id, str)
        or not isinstance(session_id, str)
        or session_id != expected_session_id
        or type(sequence) is not int
        or not isinstance(command_id, str)
        or type(succeeded) is not bool
    ):
        return None

    return (
        PresentationResult(CommandId(command_id), succeeded),
        EventCorrelation(
            TraceId(trace_id), SessionId(session_id), EventSequence(sequence)
        ),
    )
