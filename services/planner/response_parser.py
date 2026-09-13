"""
Structured response parsing for the AIForge Planner.

LLM responses are untrusted input. This module converts the provider's
raw structured response into a validated PlannerResult.

Provider-specific SDK objects should be converted to plain Python data
before reaching this parser.
"""

from __future__ import annotations
import re
import json
from typing import Any
from urllib import response

from pydantic import ValidationError

from .exceptions import LLMResponseError, PlannerValidationError
from .models.planner_result import PlannerResult


class PlannerResponseParser:
    """Parse and validate structured LLM Planner responses."""

    _CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

    def parse(self, response: Any) -> PlannerResult:
        """
        Parse a provider response into a validated PlannerResult.

        Supported input:
            - PlannerResult
            - dictionary
            - JSON string

        Args:
            response: Raw structured response returned by an LLM adapter.

        Returns:
            Validated PlannerResult.

        Raises:
            LLMResponseError:
                If the response cannot be interpreted as structured data.

            PlannerValidationError:
                If the structured response does not satisfy the
                PlannerResult contract.
        """
        data = self._parse_structure(response)

        try:
            return PlannerResult.model_validate(data)
        except ValidationError as exc:
            raise PlannerValidationError(
                f"LLM output failed PlannerResult validation: {exc}"
            ) from exc

    @staticmethod
    def _parse_structure(response: Any) -> dict[str, Any]:
        """Convert supported response formats into a dictionary."""
        if isinstance(response, PlannerResult):
            return response.model_dump(mode="python")

        if isinstance(response, dict):
            return response

        if isinstance(response, str):
            text = response.strip()

            if not text:
                raise LLMResponseError(
                    "LLM response must not be empty."
                )

            fence_match = PlannerResponseParser._CODE_FENCE_PATTERN.match(text)
            if fence_match:
                text = fence_match.group(1).strip()

            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMResponseError(
                    f"LLM response is not valid JSON: {exc}"
                ) from exc

            if not isinstance(data, dict):
                raise LLMResponseError(
                    "LLM JSON response must contain an object at the root."
                )

            return data

        raise LLMResponseError(
            "Unsupported LLM response type. Expected PlannerResult, "
            "dictionary, or JSON string."
        )