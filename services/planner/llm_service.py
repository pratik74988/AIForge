from __future__ import annotations

from typing import Protocol

from .exceptions import LLMError
from .models.planner_result import PlannerResult
from .prompt import PlannerPromptBuilder
from .response_parser import PlannerResponseParser


class LLMProvider(Protocol):
    """Interface required by LLMService from an LLM provider."""

    def generate(self, prompt) -> str:
        ...


class LLMService:
    """
    Coordinates prompt construction, LLM invocation, and response parsing.

    The service does not know which LLM provider is being used.
    """

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: PlannerPromptBuilder | None = None,
        response_parser: PlannerResponseParser | None = None,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder or PlannerPromptBuilder()
        self._response_parser = response_parser or PlannerResponseParser()

    def generate(self, context) -> PlannerResult:
        prompt = self._prompt_builder.build(context)

        try:
            response = self._provider.generate(prompt)
        except Exception as exc:
            raise LLMError(
                f"LLM provider request failed: {exc}"
            ) from exc

        return self._response_parser.parse(response)