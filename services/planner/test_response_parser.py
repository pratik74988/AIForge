"""Tests for structured Planner response parsing."""

import json

import pytest

from .exceptions import LLMResponseError, PlannerValidationError
from .models.planner_result import PlannerResult
from .models.test_training_spec import make_valid_training_spec
from .response_parser import PlannerResponseParser


def make_valid_response() -> dict:
    """Create a valid structured Planner response."""
    return {
        "recommendations": [],
        "proposed_spec": make_valid_training_spec().model_dump(
            mode="python"
        ),
        "requires_approval": True,
    }


def test_parse_dictionary() -> None:
    """A valid dictionary should produce a PlannerResult."""
    parser = PlannerResponseParser()

    result = parser.parse(make_valid_response())

    assert isinstance(result, PlannerResult)
    assert result.proposed_spec.task.target == "churn"


def test_parse_json_string() -> None:
    """A valid JSON string should produce a PlannerResult."""
    parser = PlannerResponseParser()

    response = json.dumps(make_valid_response())

    result = parser.parse(response)

    assert isinstance(result, PlannerResult)
    assert result.proposed_spec.model.template == "sklearn"


def test_parse_existing_planner_result() -> None:
    """An existing PlannerResult should be accepted."""
    parser = PlannerResponseParser()

    original = PlannerResult(
        proposed_spec=make_valid_training_spec(),
    )

    result = parser.parse(original)

    assert result == original


def test_empty_string_is_rejected() -> None:
    """Empty LLM responses should raise LLMResponseError."""
    parser = PlannerResponseParser()

    with pytest.raises(LLMResponseError, match="must not be empty"):
        parser.parse("")


def test_whitespace_string_is_rejected() -> None:
    """Whitespace-only responses should be rejected."""
    parser = PlannerResponseParser()

    with pytest.raises(LLMResponseError, match="must not be empty"):
        parser.parse("   \n")


def test_invalid_json_is_rejected() -> None:
    """Malformed JSON should raise LLMResponseError."""
    parser = PlannerResponseParser()

    with pytest.raises(
        LLMResponseError,
        match="not valid JSON",
    ):
        parser.parse("{invalid json")


def test_non_object_json_is_rejected() -> None:
    """The JSON root must be an object."""
    parser = PlannerResponseParser()

    with pytest.raises(
        LLMResponseError,
        match="must contain an object",
    ):
        parser.parse('["classification", "churn"]')


def test_invalid_training_spec_is_rejected() -> None:
    """Schema-invalid Planner output must be rejected."""
    parser = PlannerResponseParser()

    response = make_valid_response()
    del response["proposed_spec"]["model"]

    with pytest.raises(
        PlannerValidationError,
        match="failed PlannerResult validation",
    ):
        parser.parse(response)


def test_unknown_fields_are_rejected() -> None:
    """Unexpected LLM fields must not enter PlannerResult."""
    parser = PlannerResponseParser()

    response = make_valid_response()
    response["confidence"] = 0.99

    with pytest.raises(
        PlannerValidationError,
        match="failed PlannerResult validation",
    ):
        parser.parse(response)


def test_unsupported_response_type_is_rejected() -> None:
    """Unsupported provider response objects should be rejected."""
    parser = PlannerResponseParser()

    with pytest.raises(
        LLMResponseError,
        match="Unsupported LLM response type",
    ):
        parser.parse(12345)