"""
Exceptions used by the AIForge Planner.

Planner-specific exceptions prevent provider SDK errors from leaking
into the rest of the application.
"""

from __future__ import annotations


class PlannerError(Exception):
    """Base exception for Planner failures."""


class LLMError(PlannerError):
    """Base exception for LLM-related failures."""


class LLMRequestError(LLMError):
    """Raised when an LLM request fails."""


class LLMResponseError(LLMError):
    """Raised when an LLM response cannot be interpreted."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM client is incorrectly configured."""


class PlannerValidationError(PlannerError):
    """Raised when an LLM result violates the Planner contract."""