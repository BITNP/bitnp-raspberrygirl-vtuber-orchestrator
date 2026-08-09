
from base64 import b64decode
from binascii import Error as Base64Error
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from orchestrator.config import ConfigParseError, TrustedLanToken
from orchestrator.control_roles import RoleTokens

CONTROL_BIND_HOST_KEY: Final = "ORCHESTRATOR_CONTROL_BIND_HOST"

CONTROL_BIND_PORT_KEY: Final = "ORCHESTRATOR_CONTROL_BIND_PORT"

RTP_BIND_HOST_KEY: Final = "ORCHESTRATOR_RTP_BIND_HOST"

RTP_BIND_PORT_KEY: Final = "ORCHESTRATOR_RTP_BIND_PORT"

ADVERTISED_HOST_KEY: Final = "ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST"

ADVERTISED_CONTROL_PORT_KEY: Final = "ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT"

ADVERTISED_RTP_PORT_KEY: Final = "ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT"

TLS_CERT_PATH_KEY: Final = "ORCHESTRATOR_CONTROL_TLS_CERT_PATH"

TLS_KEY_PATH_KEY: Final = "ORCHESTRATOR_CONTROL_TLS_KEY_PATH"

LOOPBACK_WS_KEY: Final = "ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS"

MIC_TOKEN_KEY: Final = "ORCHESTRATOR_MIC_CONTROL_TOKEN"  # noqa: S105
SOUND_TOKEN_KEY: Final = "ORCHESTRATOR_SOUND_CONTROL_TOKEN"  # noqa: S105
COMMENTS_TOKEN_KEY: Final = "ORCHESTRATOR_COMMENTS_CONTROL_TOKEN"  # noqa: S105
FRONTEND_TOKEN_KEY: Final = "ORCHESTRATOR_FRONTEND_CONTROL_TOKEN"  # noqa: S105
OPERATOR_TOKEN_KEY: Final = "ORCHESTRATOR_OPERATOR_CONTROL_TOKEN"  # noqa: S105
MAX_SESSIONS_KEY: Final = "ORCHESTRATOR_MAX_SESSIONS"
SESSION_IDLE_TTL_SECONDS_KEY: Final = "ORCHESTRATOR_SESSION_IDLE_TTL_SECONDS"
SESSION_SWEEP_SECONDS_KEY: Final = "ORCHESTRATOR_SESSION_SWEEP_SECONDS"
VOICE_TEMPLATE_KEY: Final = "ORCHESTRATOR_VOICE_TEMPLATE_KEY"
VOICE_MATCH_THRESHOLD_KEY: Final = "ORCHESTRATOR_VOICE_MATCH_THRESHOLD"
VOICE_AMBIGUITY_MARGIN_KEY: Final = "ORCHESTRATOR_VOICE_AMBIGUITY_MARGIN"
VOICE_EVIDENCE_TTL_SECONDS_KEY: Final = "ORCHESTRATOR_VOICE_EVIDENCE_TTL_SECONDS"

DEFAULT_CONTROL_BIND_HOST: Final = "127.0.0.1"

DEFAULT_CONTROL_BIND_PORT: Final = "8443"

DEFAULT_RTP_BIND_HOST: Final = "127.0.0.1"

DEFAULT_RTP_BIND_PORT: Final = "5004"

DEFAULT_ADVERTISED_CONTROL_PORT: Final = "443"

DEFAULT_ADVERTISED_RTP_PORT: Final = "5004"

MAX_UDP_PORT: Final = 65_535
AES_256_KEY_BYTES: Final = 32

ControlScheme = Literal["ws", "wss"]


@dataclass(frozen=True, slots=True)
class TransportConfig:

    control_bind_host: str

    control_bind_port: int

    udp_bind_host: str

    udp_bind_port: int

    advertised_host: str

    advertised_control_port: int

    advertised_udp_port: int

    control_scheme: ControlScheme

    control_token: TrustedLanToken | None

    tls_cert_path: Path | None

    tls_key_path: Path | None

    role_tokens: RoleTokens = field(default_factory=RoleTokens)

    max_sessions: int = 16

    session_idle_ttl_seconds: int = 1800

    session_sweep_seconds: int = 30

    voice_template_key: bytes | None = None

    voice_match_threshold: float = 0.90

    voice_ambiguity_margin: float = 0.05

    voice_evidence_ttl_seconds: int = 120


def load_transport_config_from_env(env: Mapping[str, str]) -> TransportConfig:
    insecure_ws = _parse_insecure_ws(env.get(LOOPBACK_WS_KEY))

    control_bind_host = _require_text(
        env.get(CONTROL_BIND_HOST_KEY, DEFAULT_CONTROL_BIND_HOST), CONTROL_BIND_HOST_KEY
    )

    udp_bind_host = _require_text(
        env.get(RTP_BIND_HOST_KEY, DEFAULT_RTP_BIND_HOST), RTP_BIND_HOST_KEY
    )

    advertised_host = _require_text(env.get(ADVERTISED_HOST_KEY), ADVERTISED_HOST_KEY)

    role_tokens = _parse_role_tokens(env)

    return TransportConfig(
        control_bind_host=control_bind_host,
        control_bind_port=_parse_port(
            env.get(CONTROL_BIND_PORT_KEY, DEFAULT_CONTROL_BIND_PORT),
            CONTROL_BIND_PORT_KEY,
        ),
        udp_bind_host=udp_bind_host,
        udp_bind_port=_parse_port(
            env.get(RTP_BIND_PORT_KEY, DEFAULT_RTP_BIND_PORT), RTP_BIND_PORT_KEY
        ),
        advertised_host=advertised_host,
        advertised_control_port=_parse_port(
            env.get(ADVERTISED_CONTROL_PORT_KEY, DEFAULT_ADVERTISED_CONTROL_PORT),
            ADVERTISED_CONTROL_PORT_KEY,
        ),
        advertised_udp_port=_parse_port(
            env.get(ADVERTISED_RTP_PORT_KEY, DEFAULT_ADVERTISED_RTP_PORT),
            ADVERTISED_RTP_PORT_KEY,
        ),
        control_scheme="ws" if insecure_ws else "wss",
        control_token=None,
        tls_cert_path=_parse_tls_path(
            env.get(TLS_CERT_PATH_KEY), TLS_CERT_PATH_KEY, insecure_ws
        ),
        tls_key_path=_parse_tls_path(
            env.get(TLS_KEY_PATH_KEY), TLS_KEY_PATH_KEY, insecure_ws
        ),
        role_tokens=role_tokens,
        max_sessions=_parse_positive_int(
            env.get(MAX_SESSIONS_KEY), MAX_SESSIONS_KEY, 16
        ),
        session_idle_ttl_seconds=_parse_positive_int(
            env.get(SESSION_IDLE_TTL_SECONDS_KEY), SESSION_IDLE_TTL_SECONDS_KEY, 1800
        ),
        session_sweep_seconds=_parse_positive_int(
            env.get(SESSION_SWEEP_SECONDS_KEY), SESSION_SWEEP_SECONDS_KEY, 30
        ),
        voice_template_key=_parse_voice_key(env.get(VOICE_TEMPLATE_KEY)),
        voice_match_threshold=_parse_probability(
            env.get(VOICE_MATCH_THRESHOLD_KEY), VOICE_MATCH_THRESHOLD_KEY, 0.90
        ),
        voice_ambiguity_margin=_parse_probability(
            env.get(VOICE_AMBIGUITY_MARGIN_KEY),
            VOICE_AMBIGUITY_MARGIN_KEY,
            0.05,
        ),
        voice_evidence_ttl_seconds=_parse_positive_int(
            env.get(VOICE_EVIDENCE_TTL_SECONDS_KEY),
            VOICE_EVIDENCE_TTL_SECONDS_KEY,
            120,
        ),
    )


def _parse_voice_key(value: str | None) -> bytes | None:
    if value is None or value.strip() == "":
        return None
    try:
        key = b64decode(value.strip(), validate=True)
    except (Base64Error, ValueError) as error:
        raise ConfigParseError(field_name=VOICE_TEMPLATE_KEY) from error
    if len(key) != AES_256_KEY_BYTES:
        raise ConfigParseError(field_name=VOICE_TEMPLATE_KEY)
    return key


def _parse_probability(value: str | None, field_name: str, default: float) -> float:
    try:
        parsed = default if value is None else float(value)
    except ValueError as error:
        raise ConfigParseError(field_name=field_name) from error
    if not 0 <= parsed <= 1:
        raise ConfigParseError(field_name=field_name)
    return parsed


def _parse_insecure_ws(value: str | None) -> bool:
    match "false" if value is None else value.strip().lower():
        case "false":
            return False

        case "true":
            return True

        case _:
            raise ConfigParseError(field_name=LOOPBACK_WS_KEY)


def _require_text(value: str | None, field_name: str) -> str:
    if value is None or value.strip() == "":
        raise ConfigParseError(field_name=field_name)

    return value.strip()


def _parse_port(value: str | None, field_name: str) -> int:
    parsed = _require_text(value, field_name)

    if not parsed.isdecimal():
        raise ConfigParseError(field_name=field_name)

    port = int(parsed)

    if port < 1 or port > MAX_UDP_PORT:
        raise ConfigParseError(field_name=field_name)

    return port


def _parse_role_tokens(env: Mapping[str, str]) -> RoleTokens:
    values = {
        key: TrustedLanToken(_require_text(env.get(key), key))
        for key in (
            MIC_TOKEN_KEY,
            SOUND_TOKEN_KEY,
            COMMENTS_TOKEN_KEY,
            FRONTEND_TOKEN_KEY,
            OPERATOR_TOKEN_KEY,
        )
    }
    tokens = RoleTokens(
        mic=values[MIC_TOKEN_KEY],
        sound=values[SOUND_TOKEN_KEY],
        comments=values[COMMENTS_TOKEN_KEY],
        frontend=values[FRONTEND_TOKEN_KEY],
        operator=values[OPERATOR_TOKEN_KEY],
    )
    if not tokens.validate_unique():
        raise ConfigParseError(field_name="ORCHESTRATOR_*_CONTROL_TOKEN")
    return tokens


def _parse_tls_path(
    value: str | None, field_name: str, insecure_ws: bool
) -> Path | None:
    if insecure_ws:
        return None

    return Path(_require_text(value, field_name))


def _parse_positive_int(value: str | None, field_name: str, default: int) -> int:
    parsed = str(default) if value is None or value.strip() == "" else value.strip()
    if not parsed.isdecimal() or int(parsed) < 1:
        raise ConfigParseError(field_name=field_name)
    return int(parsed)
