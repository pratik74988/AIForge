from __future__ import annotations

from ..config import PlannerConfig
from .litellm_provider import LiteLLMProvider


def create_litellm_provider(
    config: PlannerConfig,
) -> LiteLLMProvider:
    """
    Create a LiteLLM provider from planner configuration.
    """

    return LiteLLMProvider(
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )