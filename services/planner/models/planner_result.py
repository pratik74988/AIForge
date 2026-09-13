"""
Structured output models for the AIForge Planner.

PlannerResult represents the structured result produced after the LLM
has reasoned about a planning request.

It is deliberately separate from TrainingSpec because the Planner may
produce recommendations that require user approval before a final
TrainingSpec is accepted.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .recommendation import Recommendation
from .training_spec import TrainingSpec


class PlannerResult(BaseModel):
    """
    Structured result produced by the Planner.

    Args:
        recommendations:
            Human-readable recommendations explaining the Planner's
            proposed decisions.

        proposed_spec:
            TrainingSpec proposed by the Planner.

        requires_approval:
            Whether the proposed plan should be reviewed before being
            treated as final.
    """

    model_config = ConfigDict(extra="forbid")

    recommendations: list[Recommendation] = Field(default_factory=list)

    proposed_spec: TrainingSpec

    requires_approval: bool = True