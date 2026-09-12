"""Validated deployment overrides for workload-specific LLM generation."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

type ReasoningDialect = Literal["deepseek", "openai", "none"]
type OptionalSampling = float | Literal["omit"] | None
type ReasoningSetting = Literal["enabled", "disabled", "omit"]
type TokenParameter = Literal["auto", "max_tokens", "max_completion_tokens"]


@dataclass(frozen=True, slots=True)
class LLMGenerationSettings:
    temperature: OptionalSampling = None
    top_p: OptionalSampling = None
    frequency_penalty: OptionalSampling = None
    presence_penalty: OptionalSampling = None
    reasoning: ReasoningSetting | None = None
    reasoning_effort: str = "medium"
    max_completion_tokens: int | None = None
    budget_parameter: TokenParameter = "auto"
    reasoning_dialect: ReasoningDialect | None = None

    def parameters(
        self,
        *,
        temperature: float,
        reasoning: str,
        max_completion_tokens: int,
        dialect: ReasoningDialect,
    ) -> dict[str, object]:
        """Use request defaults only where deployment settings are absent."""
        sampling = {
            "temperature": temperature
            if self.temperature is None
            else self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        body: dict[str, object] = {
            name: value
            for name, value in sampling.items()
            if value is not None and value != "omit"
        }
        selected_dialect = self.reasoning_dialect or dialect
        mode = self.reasoning or reasoning
        if mode != "omit":
            if selected_dialect == "deepseek":
                body["thinking"] = {"type": mode}
            elif selected_dialect == "openai":
                body["reasoning_effort"] = (
                    self.reasoning_effort if mode == "enabled" else "none"
                )
        budget_parameter = self.budget_parameter
        if budget_parameter == "auto":
            budget_parameter = (
                "max_completion_tokens"
                if selected_dialect == "openai"
                else "max_tokens"
            )
        body[budget_parameter] = self.max_completion_tokens or max_completion_tokens
        return body


@dataclass(frozen=True, slots=True)
class LLMGenerationConfig:
    brain: LLMGenerationSettings = field(default_factory=LLMGenerationSettings)
    maintenance: LLMGenerationSettings = field(default_factory=LLMGenerationSettings)


def _choice(value: str | None, choices: tuple[str, ...], key: str) -> str | None:
    if value is not None and value not in choices:
        raise ValueError(key)
    return value


def _sampling(
    value: str | None, minimum: float, maximum: float, key: str
) -> OptionalSampling:
    if value is None or value == "omit":
        return value
    try:
        number = float(value)
    except ValueError:
        raise ValueError(key) from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(key)
    return number


def _settings(env: Mapping[str, str], workload: str | None) -> LLMGenerationSettings:
    def selected(suffix: str) -> tuple[str | None, str]:
        prefixes = ("ORCHESTRATOR_LLM_",)
        if workload is not None:
            prefixes = (f"ORCHESTRATOR_LLM_{workload}_", *prefixes)
        for prefix in prefixes:
            key = prefix + suffix
            value = env.get(key, "").strip()
            if value:
                return value, key
        return None, "ORCHESTRATOR_LLM_" + suffix

    def sampling(suffix: str, minimum: float, maximum: float) -> OptionalSampling:
        value, key = selected(suffix)
        return _sampling(value, minimum, maximum, key)

    raw_tokens, token_key = selected("MAX_COMPLETION_TOKENS")
    tokens = None
    if raw_tokens is not None:
        try:
            tokens = int(raw_tokens)
        except ValueError:
            raise ValueError(token_key) from None
        if tokens <= 0:
            raise ValueError(token_key)
    reasoning, reasoning_key = selected("REASONING")
    _ = _choice(reasoning, ("enabled", "disabled", "omit"), reasoning_key)
    effort, effort_key = selected("REASONING_EFFORT")
    _ = _choice(effort, ("minimal", "low", "medium", "high", "xhigh"), effort_key)
    parameter, parameter_key = selected("TOKEN_PARAMETER")
    _ = _choice(
        parameter, ("auto", "max_tokens", "max_completion_tokens"), parameter_key
    )
    dialect, dialect_key = selected("REASONING_DIALECT")
    _ = _choice(dialect, ("deepseek", "openai", "none"), dialect_key)
    return LLMGenerationSettings(
        temperature=sampling("TEMPERATURE", 0.0, 2.0),
        top_p=sampling("TOP_P", 0.0, 1.0),
        frequency_penalty=sampling("FREQUENCY_PENALTY", -2.0, 2.0),
        presence_penalty=sampling("PRESENCE_PENALTY", -2.0, 2.0),
        reasoning=cast("ReasoningSetting | None", reasoning),
        reasoning_effort=effort or "medium",
        max_completion_tokens=tokens,
        budget_parameter=cast("TokenParameter", parameter or "auto"),
        reasoning_dialect=cast("ReasoningDialect | None", dialect),
    )


def load_llm_generation_config(env: Mapping[str, str]) -> LLMGenerationConfig:
    # Validate global values even if both workloads override them.
    _ = _settings(env, None)
    return LLMGenerationConfig(_settings(env, "BRAIN"), _settings(env, "MAINTENANCE"))
