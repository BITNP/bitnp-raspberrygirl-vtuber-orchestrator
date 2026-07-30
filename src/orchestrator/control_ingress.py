
from __future__ import annotations

from dataclasses import dataclass

from orchestrator.identity import (
    EncryptedVoiceTemplate,
    ProfileEnrollment,
    VoiceProfileId,
)
from orchestrator.ids import SessionId, TraceId
from orchestrator.interactions import (
    ActionProposal,
    CommandId,
    PresentationCommand,
    PresentationCommandKind,
    PresentationResult,
)
from orchestrator.json_boundary import JsonBoundaryError, JsonValue, parse_json_value
from orchestrator.sessions import EventCorrelation, EventSequence


@dataclass(frozen=True, slots=True)
class ProfileEnrollmentControl:

    enrollment: ProfileEnrollment

    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class ProfileRevocationControl:

    profile_id: VoiceProfileId

    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class ActionControl:

    proposal: ActionProposal

    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class PresentationControl:

    proposal: PresentationCommand

    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class PresentationResultControl:

    result: PresentationResult

    correlation: EventCorrelation


type SessionControl = (
    ProfileEnrollmentControl
    | ProfileRevocationControl
    | ActionControl
    | PresentationControl
    | PresentationResultControl
)


def parse_session_control(  # noqa: C901, PLR0911, PLR0912
    raw_message: str,
) -> SessionControl | None:
    try:
        value = parse_json_value(raw_message)

    except JsonBoundaryError:
        return None

    if not isinstance(value, dict):
        return None

    if not _canonical_envelope(value):
        return None

    correlation = _correlation(value)

    if correlation is None:
        return None

    data = value.get("data")

    if not isinstance(data, dict):
        return None

    event_type = value.get("event_type")

    source = value.get("source")

    match event_type, source:
        case "profile.enroll.command", "orchestrator":
            return _profile_enrollment(data, correlation)

        case "profile.revoke.command", "orchestrator":
            return _profile_revocation(data, correlation)

        case "action.command", "orchestrator":
            return _action(data, correlation)

        case "presentation.load.command", "orchestrator":
            return _presentation(PresentationCommandKind.LOAD, data, correlation)

        case "presentation.play.command", "orchestrator":
            return _presentation(PresentationCommandKind.PLAY, data, correlation)

        case "presentation.navigate.command", "orchestrator":
            return _presentation(PresentationCommandKind.NAVIGATE, data, correlation)

        case "presentation.result", "frontend":
            return _presentation_result(data, correlation)

        case _:
            return None


def _correlation(value: dict[str, JsonValue]) -> EventCorrelation | None:
    trace_id = value.get("trace_id")

    session_id = value.get("session_id")

    sequence = value.get("seq")

    if (
        not isinstance(trace_id, str)
        or trace_id.strip() == ""
        or not isinstance(session_id, str)
        or session_id.strip() == ""
        or type(sequence) is not int
        or sequence < 0
    ):
        return None

    return EventCorrelation(
        TraceId(trace_id), SessionId(session_id), EventSequence(sequence)
    )


def _profile_enrollment(
    data: dict[str, JsonValue], correlation: EventCorrelation
) -> ProfileEnrollmentControl | None:
    profile_id = _text(data, "profile_id")

    preferred_name = _text(data, "preferred_name")

    encrypted_template = _text(data, "encrypted_template")

    consented = data.get("consented")

    if (
        profile_id is None
        or preferred_name is None
        or encrypted_template is None
        or consented is not True
    ):
        return None

    return ProfileEnrollmentControl(
        ProfileEnrollment(
            VoiceProfileId(profile_id),
            preferred_name,
            EncryptedVoiceTemplate(encrypted_template.encode()),
            consented=True,
        ),
        correlation,
    )


def _profile_revocation(
    data: dict[str, JsonValue], correlation: EventCorrelation
) -> ProfileRevocationControl | None:
    profile_id = _text(data, "profile_id")

    if profile_id is None:
        return None

    return ProfileRevocationControl(VoiceProfileId(profile_id), correlation)


def _action(
    data: dict[str, JsonValue], correlation: EventCorrelation
) -> ActionControl | None:
    command_id = _text(data, "command_id")

    action = _text(data, "action")

    if command_id is None or action is None:
        return None

    return ActionControl(ActionProposal(action, CommandId(command_id)), correlation)


def _presentation(
    kind: PresentationCommandKind,
    data: dict[str, JsonValue],
    correlation: EventCorrelation,
) -> PresentationControl | None:
    command_id = _text(data, "command_id")

    deck_id = _text(data, "deck_id")

    deck_version = _text(data, "deck_version")

    page = data.get("page")

    if command_id is None or deck_id is None or type(page) is not int:
        return None

    return PresentationControl(
        PresentationCommand(
            kind,
            deck_id,
            page,
            CommandId(command_id),
            "v1" if deck_version is None else deck_version,
        ),
        correlation,
    )


def _presentation_result(
    data: dict[str, JsonValue], correlation: EventCorrelation
) -> PresentationResultControl | None:
    command_id = _text(data, "command_id")

    succeeded = data.get("succeeded")

    if command_id is None or type(succeeded) is not bool:
        return None

    return PresentationResultControl(
        PresentationResult(CommandId(command_id), succeeded), correlation
    )


def _text(data: dict[str, JsonValue], field_name: str) -> str | None:
    value = data.get(field_name)

    if not isinstance(value, str) or value.strip() == "":
        return None

    return value


def _canonical_envelope(value: dict[str, JsonValue]) -> bool:
    required = {
        "schema_version",
        "event_type",
        "event_id",
        "source",
        "time",
        "trace_id",
        "session_id",
        "seq",
        "data",
    }

    allowed = required | {"turn_id", "segment_id", "traceparent"}

    if set(value).difference(allowed) or not required.issubset(value):
        return False

    schema_version = value["schema_version"]

    event_id = value["event_id"]

    event_time = value["time"]

    return (
        schema_version in {"1.0.0", "1.1.0"}
        and isinstance(event_id, str)
        and event_id.strip() != ""
        and isinstance(event_time, str)
        and event_time.strip() != ""
    )
