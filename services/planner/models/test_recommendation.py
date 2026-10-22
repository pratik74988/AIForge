"""Tests for Planner recommendation models."""

import pytest
from pydantic import ValidationError

from .recommendation import Recommendation, RecommendationType


def test_valid_recommendation() -> None:
    """A valid recommendation should be accepted."""
    recommendation = Recommendation(
        type=RecommendationType.MODEL,
        title="Use class weighting",
        reasoning="The target classes are imbalanced.",
        decision="Set class_weight to balanced.",
    )

    assert recommendation.type == RecommendationType.MODEL
    assert recommendation.title == "Use class weighting"
    assert recommendation.requires_approval is True


def test_recommendation_approval_can_be_disabled() -> None:
    """Recommendations can be marked as not requiring approval."""
    recommendation = Recommendation(
        type=RecommendationType.PREPROCESSING,
        title="Impute missing values",
        reasoning="Numeric columns contain missing values.",
        decision="Use median imputation.",
        requires_approval=False,
    )

    assert recommendation.requires_approval is False


def test_empty_title_is_rejected() -> None:
    """Recommendation titles must not be empty."""
    with pytest.raises(ValidationError):
        Recommendation(
            type=RecommendationType.MODEL,
            title="",
            reasoning="The target is imbalanced.",
            decision="Use class weighting.",
        )


def test_empty_reasoning_is_rejected() -> None:
    """Recommendation reasoning must not be empty."""
    with pytest.raises(ValidationError):
        Recommendation(
            type=RecommendationType.MODEL,
            title="Use class weighting",
            reasoning="",
            decision="Use class weighting.",
        )


def test_empty_decision_is_rejected() -> None:
    """Recommendation decisions must not be empty."""
    with pytest.raises(ValidationError):
        Recommendation(
            type=RecommendationType.MODEL,
            title="Use class weighting",
            reasoning="The target is imbalanced.",
            decision="",
        )


def test_unknown_fields_are_rejected() -> None:
    """Unexpected LLM-generated fields must not enter the model."""
    with pytest.raises(ValidationError):
        Recommendation(
            type=RecommendationType.MODEL,
            title="Use class weighting",
            reasoning="The target is imbalanced.",
            decision="Use class weighting.",
            confidence=0.95,
        )


def test_all_recommendation_types_are_supported() -> None:
    """All defined recommendation categories should be valid."""
    for recommendation_type in RecommendationType:
        recommendation = Recommendation(
            type=recommendation_type,
            title="Example recommendation",
            reasoning="Example reasoning.",
            decision="Example decision.",
        )

        assert recommendation.type == recommendation_type