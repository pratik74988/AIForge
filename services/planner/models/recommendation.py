"""
Models for Planner recommendations.

Recommendations represent decisions or suggestions made during the
planning process. They are intentionally separate from TrainingSpec.

A recommendation may be shown to the user before a final TrainingSpec
is produced and approved.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RecommendationType(str, Enum):
    """Categories of decisions the Planner can recommend."""

    PREPROCESSING = "preprocessing"
    MODEL = "model"
    EVALUATION = "evaluation"
    SPLIT = "split"
    FEATURES = "features"
    RESOURCES = "resources"


class Recommendation(BaseModel):
    """
    A single Planner recommendation.

    Args:
        type:
            Category of the recommendation.

        title:
            Short human-readable description.

        reasoning:
            Explanation of why the Planner is making this recommendation.

        decision:
            Concrete decision proposed by the Planner.

        requires_approval:
            Whether the Planner should wait for user approval before
            treating this recommendation as final.
    """

    model_config = ConfigDict(extra="forbid")

    type: RecommendationType
    title: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    requires_approval: bool = True