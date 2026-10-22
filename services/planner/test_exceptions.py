"""Tests for Planner exceptions."""

from .exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMRequestError,
    LLMResponseError,
    PlannerError,
    PlannerValidationError,
)


def test_planner_error_is_base_exception() -> None:
    """PlannerError should derive directly from Exception."""
    error = PlannerError("planner failed")

    assert isinstance(error, Exception)


def test_llm_errors_inherit_from_planner_error() -> None:
    """All LLM failures should be catchable as PlannerError."""
    errors = (
        LLMError("llm error"),
        LLMRequestError("request failed"),
        LLMResponseError("invalid response"),
        LLMConfigurationError("invalid configuration"),
    )

    for error in errors:
        assert isinstance(error, LLMError)
        assert isinstance(error, PlannerError)


def test_validation_error_inherits_from_planner_error() -> None:
    """Planner validation failures should be PlannerError instances."""
    error = PlannerValidationError("invalid planner output")

    assert isinstance(error, PlannerError)


def test_exception_messages_are_preserved() -> None:
    """Useful error messages should survive exception construction."""
    error = LLMResponseError("response was not valid JSON")

    assert str(error) == "response was not valid JSON"
    