"""Tests for PlannerResult."""

import pytest
from pydantic import ValidationError

from .planner_result import PlannerResult
from .recommendation import Recommendation, RecommendationType
from .test_training_spec import make_valid_training_spec


def make_valid_planner_result() -> PlannerResult:
    """Build a valid PlannerResult for testing."""
    return PlannerResult(
        recommendations=[
            Recommendation(
                type=RecommendationType.MODEL,
                title="Use class weighting",
                reasoning="The target classes are imbalanced.",
                decision="Use balanced class weights.",
            )
        ],
        proposed_spec=make_valid_training_spec(),
    )


def test_valid_planner_result() -> None:
    """A valid PlannerResult should be accepted."""
    result = make_valid_planner_result()

    assert len(result.recommendations) == 1
    assert result.proposed_spec.task.target == "churn"
    assert result.requires_approval is True


def test_planner_result_can_have_no_recommendations() -> None:
    """A PlannerResult may contain only a proposed specification."""
    result = PlannerResult(
        proposed_spec=make_valid_training_spec(),
    )

    assert result.recommendations == []


def test_approval_can_be_disabled() -> None:
    """The Planner can mark a result as already approved."""
    result = PlannerResult(
        proposed_spec=make_valid_training_spec(),
        requires_approval=False,
    )

    assert result.requires_approval is False


def test_proposed_spec_is_required() -> None:
    """A PlannerResult must contain a proposed TrainingSpec."""
    with pytest.raises(ValidationError):
        PlannerResult(
            recommendations=[],
        )


def test_invalid_recommendation_is_rejected() -> None:
    """Invalid nested recommendations must not be accepted."""
    with pytest.raises(ValidationError):
        PlannerResult(
            recommendations=[
                {
                    "type": "model",
                    "title": "",
                    "reasoning": "The target is imbalanced.",
                    "decision": "Use balanced class weights.",
                }
            ],
            proposed_spec=make_valid_training_spec(),
        )


def test_invalid_training_spec_is_rejected() -> None:
    """Invalid nested TrainingSpec data must not be accepted."""
    with pytest.raises(ValidationError):
        PlannerResult(
            recommendations=[],
            proposed_spec={
                "task": {
                    "type": "classification",
                    "target": "churn",
                }
            },
        )


def test_unknown_fields_are_rejected() -> None:
    """Unexpected LLM-generated fields must not enter the result."""
    with pytest.raises(ValidationError):
        PlannerResult(
            proposed_spec=make_valid_training_spec(),
            confidence=0.95,
        )