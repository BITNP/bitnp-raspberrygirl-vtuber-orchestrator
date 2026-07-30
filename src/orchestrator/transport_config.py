"""模块契约说明.

职责: 提供 orchestrator.transport_config
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from orchestrator.config import ConfigParseError, TrustedLanToken

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

TOKEN_KEY: Final = "TRUSTED_LAN_TOKEN"  # noqa: S105 - environment key name only.

DEFAULT_CONTROL_BIND_HOST: Final = "127.0.0.1"

DEFAULT_CONTROL_BIND_PORT: Final = "8443"

DEFAULT_RTP_BIND_HOST: Final = "127.0.0.1"

DEFAULT_RTP_BIND_PORT: Final = "5004"

DEFAULT_ADVERTISED_CONTROL_PORT: Final = "443"

DEFAULT_ADVERTISED_RTP_PORT: Final = "5004"

LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "localhost"})

MAX_UDP_PORT: Final = 65_535

ControlScheme = Literal["ws", "wss"]


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """类契约说明.

    职责: 保存 TransportConfig
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: control_bind_host、control_bi
    nd_port、udp_bind_host、udp_bind_port、
    advertised_host、advertised_control_p
    ort。
    """

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


def load_transport_config_from_env(env: Mapping[str, str]) -> TransportConfig:
    """函数契约说明.

    功能: 执行
    load_transport_config_from_env
    的同步逻辑,并协调 _parse_loopback_ws,
    _require_text, TransportConfig, get。
    参数: env: Mapping[str, str]。 必填。
    契约: 同步调用。 返回 `TransportConfig`。
    """
    loopback_ws = _parse_loopback_ws(env.get(LOOPBACK_WS_KEY))

    control_bind_host = _require_text(
        env.get(CONTROL_BIND_HOST_KEY, DEFAULT_CONTROL_BIND_HOST), CONTROL_BIND_HOST_KEY
    )

    udp_bind_host = _require_text(
        env.get(RTP_BIND_HOST_KEY, DEFAULT_RTP_BIND_HOST), RTP_BIND_HOST_KEY
    )

    advertised_host = _require_text(env.get(ADVERTISED_HOST_KEY), ADVERTISED_HOST_KEY)

    if loopback_ws:
        _require_loopback_host(control_bind_host, CONTROL_BIND_HOST_KEY)

        _require_loopback_host(udp_bind_host, RTP_BIND_HOST_KEY)

        _require_loopback_host(advertised_host, ADVERTISED_HOST_KEY)

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
        control_scheme="ws" if loopback_ws else "wss",
        control_token=_parse_token(env.get(TOKEN_KEY), loopback_ws),
        tls_cert_path=_parse_tls_path(
            env.get(TLS_CERT_PATH_KEY), TLS_CERT_PATH_KEY, loopback_ws
        ),
        tls_key_path=_parse_tls_path(
            env.get(TLS_KEY_PATH_KEY), TLS_KEY_PATH_KEY, loopback_ws
        ),
    )


def _parse_loopback_ws(value: str | None) -> bool:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: value: str | None。 必填。
    契约: 同步调用。 返回 `bool`。 可能抛出
    ConfigParseError。
    """
    match "false" if value is None else value.strip().lower():
        case "false":
            return False

        case "true":
            return True

        case _:
            raise ConfigParseError(field_name=LOOPBACK_WS_KEY)


def _require_text(value: str | None, field_name: str) -> str:
    """函数契约说明.

    功能: 执行 _require_text 的同步逻辑,并协调
    strip, ConfigParseError。
    参数: value: str | None。 必填。
    field_name: str。 必填。
    契约: 同步调用。 返回 `str`。 可能抛出
    ConfigParseError。
    """
    if value is None or value.strip() == "":
        raise ConfigParseError(field_name=field_name)

    return value.strip()


def _parse_port(value: str | None, field_name: str) -> int:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: value: str | None。 必填。
    field_name: str。 必填。
    契约: 同步调用。 返回 `int`。 可能抛出
    ConfigParseError。
    """
    parsed = _require_text(value, field_name)

    if not parsed.isdecimal():
        raise ConfigParseError(field_name=field_name)

    port = int(parsed)

    if port < 1 or port > MAX_UDP_PORT:
        raise ConfigParseError(field_name=field_name)

    return port


def _require_loopback_host(host: str, field_name: str) -> None:
    """函数契约说明.

    功能: 执行 _require_loopback_host
    的同步逻辑,并协调 lower, ConfigParseError。
    参数: host: str。 必填。 field_name: str。
    必填。
    契约: 同步调用。 返回 `None`。 可能抛出
    ConfigParseError。
    """
    if host.lower() not in LOOPBACK_HOSTS:
        raise ConfigParseError(field_name=field_name)


def _parse_token(value: str | None, loopback_ws: bool) -> TrustedLanToken | None:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: value: str | None。 必填。
    loopback_ws: bool。 必填。
    契约: 同步调用。 返回 `TrustedLanToken |
    None`。
    """
    if loopback_ws:
        return None

    return TrustedLanToken(_require_text(value, TOKEN_KEY))


def _parse_tls_path(
    value: str | None, field_name: str, loopback_ws: bool
) -> Path | None:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: value: str | None。 必填。
    field_name: str。 必填。 loopback_ws:
    bool。 必填。
    契约: 同步调用。 返回 `Path | None`。
    """
    if loopback_ws:
        return None

    return Path(_require_text(value, field_name))
