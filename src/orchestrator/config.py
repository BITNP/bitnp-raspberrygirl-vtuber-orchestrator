"""Typed Orchestrator runtime configuration."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, NewType, override

DEFAULT_SERVICE_NAME: Final = "orchestrator"
DEFAULT_SERVICE_VERSION: Final = "0.1.0"
DEFAULT_FAKE_SESSION_PREFIX: Final = "session-fake"
DEFAULT_LLM_PROVIDER: Final = "mock"
DEFAULT_ASR_PROVIDER: Final = "mock"
DEFAULT_TTS_PROVIDER: Final = "mock"
DEFAULT_TRUSTED_LAN_TOKEN_MIN_LENGTH: Final = 12
LLM_PROVIDER_KEY: Final = "ORCHESTRATOR_LLM_PROVIDER"
LLM_API_KEY_KEY: Final = "ORCHESTRATOR_LLM_API_KEY"
ASR_PROVIDER_KEY: Final = "ORCHESTRATOR_ASR_PROVIDER"
ASR_ENDPOINT_KEY: Final = "ORCHESTRATOR_ASR_ENDPOINT"
ASR_MODEL_KEY: Final = "ORCHESTRATOR_ASR_MODEL"
ASR_API_KEY_KEY: Final = "ORCHESTRATOR_ASR_API_KEY"
TTS_PROVIDER_KEY: Final = "ORCHESTRATOR_TTS_PROVIDER"
TTS_ENDPOINT_KEY: Final = "ORCHESTRATOR_TTS_ENDPOINT"
TTS_MODEL_KEY: Final = "ORCHESTRATOR_TTS_MODEL"
TTS_API_KEY_KEY: Final = "ORCHESTRATOR_TTS_API_KEY"
TRUSTED_LAN_TOKEN_KEY: Final = "TRUSTED_LAN_TOKEN"  # noqa: S105 - env key name only.
SERVICE_NAME_KEY: Final = "ORCHESTRATOR_SERVICE_NAME"
SERVICE_VERSION_KEY: Final = "ORCHESTRATOR_SERVICE_VERSION"
SESSION_ID_PREFIX_KEY: Final = "ORCHESTRATOR_SESSION_ID_PREFIX"
LlmProvider = Literal["mock", "openai_compatible"]
AsrProvider = Literal["mock", "openai_compatible"]
TtsProvider = Literal["mock", "vllm_omni"]
TrustedLanToken = NewType("TrustedLanToken", str)
LlmApiKey = NewType("LlmApiKey", str)


@dataclass(frozen=True, slots=True)
class ConfigParseError(Exception):
    """Raised when a config field cannot be parsed."""

    field_name: str

    @override
    def __str__(self) -> str:
        return f"config field is blank: {self.field_name}"


@dataclass(frozen=True, slots=True)
class OrchestratorConfigInput:
    """Raw parsed values before whitespace normalization."""

    service_name: str
    service_version: str
    session_id_prefix: str
    fake: bool
    llm_provider: LlmProvider = DEFAULT_LLM_PROVIDER
    llm_api_key: LlmApiKey | None = None
    asr_provider: AsrProvider = DEFAULT_ASR_PROVIDER
    asr_endpoint: str | None = None
    asr_model: str | None = None
    asr_api_key: str | None = None
    tts_provider: TtsProvider = DEFAULT_TTS_PROVIDER
    tts_endpoint: str | None = None
    tts_model: str | None = None
    tts_api_key: str | None = None
    trusted_lan_token: TrustedLanToken | None = None


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Runtime config consumed by the central Orchestrator shell."""

    service_name: str
    service_version: str
    session_id_prefix: str
    fake: bool
    llm_provider: LlmProvider = DEFAULT_LLM_PROVIDER
    llm_api_key: LlmApiKey | None = None
    asr_provider: AsrProvider = DEFAULT_ASR_PROVIDER
    asr_endpoint: str | None = None
    asr_model: str | None = None
    asr_api_key: str | None = None
    tts_provider: TtsProvider = DEFAULT_TTS_PROVIDER
    tts_endpoint: str | None = None
    tts_model: str | None = None
    tts_api_key: str | None = None
    trusted_lan_token: TrustedLanToken | None = None

    @classmethod
    def parse(cls, config: OrchestratorConfigInput) -> "OrchestratorConfig":
        """Normalize trusted config input into an immutable config value."""
        for field_name, raw_value in (
            ("service_name", config.service_name),
            ("service_version", config.service_version),
            ("session_id_prefix", config.session_id_prefix),
        ):
            if raw_value.strip() == "":
                raise ConfigParseError(field_name=field_name)
        _require_provider_fields(
            config.asr_provider,
            config.asr_endpoint,
            config.asr_model,
            ASR_ENDPOINT_KEY,
            ASR_MODEL_KEY,
        )
        _require_provider_fields(
            config.tts_provider,
            config.tts_endpoint,
            config.tts_model,
            TTS_ENDPOINT_KEY,
            TTS_MODEL_KEY,
        )
        return cls(
            service_name=config.service_name.strip(),
            service_version=config.service_version.strip(),
            session_id_prefix=config.session_id_prefix.strip(),
            fake=config.fake,
            llm_provider=config.llm_provider,
            llm_api_key=config.llm_api_key,
            asr_provider=config.asr_provider,
            asr_endpoint=_normalize_optional(config.asr_endpoint),
            asr_model=_normalize_optional(config.asr_model),
            asr_api_key=_normalize_optional(config.asr_api_key),
            tts_provider=config.tts_provider,
            tts_endpoint=_normalize_optional(config.tts_endpoint),
            tts_model=_normalize_optional(config.tts_model),
            tts_api_key=_normalize_optional(config.tts_api_key),
            trusted_lan_token=config.trusted_lan_token,
        )


def load_fake_config() -> OrchestratorConfig:
    """Return deterministic fake config for local tests and readiness checks."""
    return OrchestratorConfig.parse(OrchestratorConfigInput(
        service_name=DEFAULT_SERVICE_NAME,
        service_version=DEFAULT_SERVICE_VERSION,
        session_id_prefix=DEFAULT_FAKE_SESSION_PREFIX,
        fake=True,
    ))


def load_config_from_env(env: Mapping[str, str] | None = None) -> OrchestratorConfig:
    """Parse Orchestrator config from environment-style key/value input."""
    source = os.environ if env is None else env
    return OrchestratorConfig.parse(OrchestratorConfigInput(
        service_name=source.get(SERVICE_NAME_KEY, DEFAULT_SERVICE_NAME),
        service_version=source.get(SERVICE_VERSION_KEY, DEFAULT_SERVICE_VERSION),
        session_id_prefix=source.get(
            SESSION_ID_PREFIX_KEY,
            DEFAULT_FAKE_SESSION_PREFIX,
        ),
        fake=_parse_fake(source.get(LLM_PROVIDER_KEY)),
        llm_provider=_parse_llm_provider(source.get(LLM_PROVIDER_KEY)),
        llm_api_key=_parse_optional_secret(source.get(LLM_API_KEY_KEY)),
        asr_provider=_parse_asr_provider(source.get(ASR_PROVIDER_KEY)),
        asr_endpoint=source.get(ASR_ENDPOINT_KEY),
        asr_model=source.get(ASR_MODEL_KEY),
        asr_api_key=source.get(ASR_API_KEY_KEY),
        tts_provider=_parse_tts_provider(source.get(TTS_PROVIDER_KEY)),
        tts_endpoint=source.get(TTS_ENDPOINT_KEY),
        tts_model=source.get(TTS_MODEL_KEY),
        tts_api_key=source.get(TTS_API_KEY_KEY),
        trusted_lan_token=_parse_optional_token(source.get(TRUSTED_LAN_TOKEN_KEY)),
    ))


def _parse_fake(raw_provider: str | None) -> bool:
    return _parse_llm_provider(raw_provider) == "mock"


def _parse_llm_provider(raw_provider: str | None) -> LlmProvider:
    provider = DEFAULT_LLM_PROVIDER if raw_provider is None else raw_provider.strip()
    match provider:
        case "mock":
            return "mock"
        case "openai_compatible":
            return "openai_compatible"
        case _:
            raise ConfigParseError(field_name=LLM_PROVIDER_KEY)


def _parse_asr_provider(raw_provider: str | None) -> AsrProvider:
    provider = DEFAULT_ASR_PROVIDER if raw_provider is None else raw_provider.strip()
    match provider:
        case "mock" | "openai_compatible":
            return provider
        case _:
            raise ConfigParseError(field_name=ASR_PROVIDER_KEY)


def _parse_tts_provider(raw_provider: str | None) -> TtsProvider:
    provider = DEFAULT_TTS_PROVIDER if raw_provider is None else raw_provider.strip()
    match provider:
        case "mock" | "vllm_omni":
            return provider
        case _:
            raise ConfigParseError(field_name=TTS_PROVIDER_KEY)


def _require_provider_fields(
    provider: AsrProvider | TtsProvider,
    endpoint: str | None,
    model: str | None,
    endpoint_field: str,
    model_field: str,
) -> None:
    if provider == "mock":
        return
    if endpoint is None or endpoint.strip() == "":
        raise ConfigParseError(field_name=endpoint_field)
    if model is None or model.strip() == "":
        raise ConfigParseError(field_name=model_field)


def _normalize_optional(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _parse_optional_secret(raw_secret: str | None) -> LlmApiKey | None:
    if raw_secret is None or raw_secret.strip() == "":
        return None
    return LlmApiKey(raw_secret.strip())


def _parse_optional_token(raw_token: str | None) -> TrustedLanToken | None:
    if raw_token is None or raw_token.strip() == "":
        return None
    return TrustedLanToken(raw_token.strip())
