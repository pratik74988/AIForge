from __future__ import annotations

import os

from .exceptions import LLMConfigurationError


class PlannerConfig:
    """
    Runtime configuration for the AIForge Planner.

    Configuration is read from environment variables so API keys and
    provider-specific settings are not stored in source code.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or os.getenv("AIFORGE_LLM_MODEL")
        self.api_key = api_key or os.getenv("AIFORGE_LLM_API_KEY")

        temperature_value = (
            temperature
            if temperature is not None
            else os.getenv("AIFORGE_LLM_TEMPERATURE", "0.0")
        )

        max_tokens_value = (
            max_tokens
            if max_tokens is not None
            else os.getenv("AIFORGE_LLM_MAX_TOKENS")
        )

        try:
            self.temperature = float(temperature_value)
        except (TypeError, ValueError) as exc:
            raise LLMConfigurationError(
                "AIFORGE_LLM_TEMPERATURE must be a valid number."
            ) from exc

        if max_tokens_value is None:
            self.max_tokens = None
        else:
            try:
                self.max_tokens = int(max_tokens_value)
            except (TypeError, ValueError) as exc:
                raise LLMConfigurationError(
                    "AIFORGE_LLM_MAX_TOKENS must be a valid integer."
                ) from exc

        self._validate()

    def _validate(self) -> None:
        if not self.model or not self.model.strip():
            raise LLMConfigurationError(
                "AIFORGE_LLM_MODEL must be configured."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise LLMConfigurationError(
                "AIFORGE_LLM_TEMPERATURE must be between 0.0 and 2.0."
            )

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise LLMConfigurationError(
                "AIFORGE_LLM_MAX_TOKENS must be greater than zero."
            )