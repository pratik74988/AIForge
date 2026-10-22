from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CompiledPipeline(BaseModel):
    """Compiled representation of a TrainingSpec."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    template: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)

    preprocessing_steps: list[str] = Field(
        default_factory=list
    )

    hyperparameters: dict[str, Any] = Field(
        default_factory=dict
    )

    evaluation_metrics: list[str] = Field(
        default_factory=list
    )

    primary_metric: str = Field(min_length=1)

    template_class: type = Field(
        description="Resolved template implementation class."
    )

    pipeline: Any = Field(
        default=None,
        description="Constructed sklearn pipeline.",
    )