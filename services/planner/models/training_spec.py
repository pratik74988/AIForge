"""
Pydantic models for the AIForge TrainingSpec contract.

TrainingSpec is the machine-readable contract produced by the Planner
and consumed by Governance and the Pipeline Compiler.

The models intentionally describe ML intent rather than framework-specific
implementation details. Template-specific interpretation belongs to the
Template Registry and Pipeline Compiler.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    """Supported machine-learning task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class SplitStrategy(str, Enum):
    """Strategies available for splitting a dataset."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    TIME = "time"


class TaskSpec(BaseModel):
    """Defines the machine-learning problem to solve."""

    model_config = ConfigDict(extra="forbid")

    type: TaskType
    target: str = Field(min_length=1)

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        """Reject empty or whitespace-only target names."""
        value = value.strip()

        if not value:
            raise ValueError("Target name must not be empty.")

        return value

class FeatureSpec(BaseModel):
    """
    Defines which dataset features should be available to the pipeline.

    If ``include`` is None, all eligible features are considered unless
    explicitly excluded.
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str] | None = None
    exclude: list[str] = Field(default_factory=list)

    @field_validator("include", "exclude")
    @classmethod
    def validate_feature_names(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        """Reject empty or whitespace-only feature names."""
        if value is None:
            return None

        cleaned = [name.strip() for name in value]

        if any(not name for name in cleaned):
            raise ValueError("Feature names must not be empty.")

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Feature names must be unique.")

        return cleaned


class PreprocessingStep(BaseModel):
    """
    Describes one semantic preprocessing operation.

    ``operation`` is intentionally framework-agnostic. For example,
    ``scale`` with ``method=standard`` can be compiled into a
    ``StandardScaler`` by the sklearn template.

    The ``columns`` field accepts either:
    - ``"numeric"``
    - ``"categorical"``
    - ``"all"``
    - an explicit list of column names
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    columns: str | list[str] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "operation")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Reject empty or whitespace-only values."""
        value = value.strip()

        if not value:
            raise ValueError("Value must not be empty.")

        return value

    @field_validator("columns")
    @classmethod
    def validate_columns(
        cls,
        value: str | list[str] | None,
    ) -> str | list[str] | None:
        """Validate semantic column groups and explicit column names."""
        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()

            if not value:
                raise ValueError("Column group must not be empty.")

            allowed_groups = {"numeric", "categorical", "all"}

            if value not in allowed_groups:
                raise ValueError(
                    "String columns must be one of: "
                    "'numeric', 'categorical', or 'all'. "
                    "Use a list for explicit column names."
                )

            return value

        cleaned = [column.strip() for column in value]

        if any(not column for column in cleaned):
            raise ValueError("Column names must not be empty.")

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Column names must be unique.")

        return cleaned


class PreprocessingSpec(BaseModel):
    """Defines the ordered preprocessing pipeline."""

    model_config = ConfigDict(extra="forbid")

    steps: list[PreprocessingStep] = Field(default_factory=list)


class SplitSpec(BaseModel):
    """Defines how the dataset should be divided for training and evaluation."""

    model_config = ConfigDict(extra="forbid")

    strategy: SplitStrategy
    test_size: float = Field(gt=0, lt=1)
    validation_size: float = Field(default=0.0, ge=0, lt=1)

    @field_validator("validation_size")
    @classmethod
    def validate_total_split_size(cls, value: float, info: Any) -> float:
        """
        Ensure validation and test sets do not consume the entire dataset.

        The training portion is the remaining fraction:

            train = 1 - test_size - validation_size
        """
        test_size = info.data.get("test_size")

        if test_size is not None and test_size + value >= 1:
            raise ValueError(
                "test_size + validation_size must be less than 1."
            )

        return value


class ModelSpec(BaseModel):
    """
    Defines the model implementation selected by the Planner.

    ``template`` identifies the implementation family registered in the
    Template Registry, while ``algorithm`` identifies the model within
    that template.
    """

    model_config = ConfigDict(extra="forbid")

    template: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("template", "algorithm")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Reject empty or whitespace-only identifiers."""
        value = value.strip()

        if not value:
            raise ValueError("Model identifier must not be empty.")

        return value


class EvaluationSpec(BaseModel):
    """Defines how model performance should be evaluated."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str = Field(min_length=1)
    metrics: list[str] = Field(min_length=1)

    @field_validator("primary_metric")
    @classmethod
    def validate_primary_metric(cls, value: str) -> str:
        """Normalize and validate the primary metric name."""
        value = value.strip()

        if not value:
            raise ValueError("Primary metric must not be empty.")

        return value

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        """Normalize metric names and remove accidental duplicates."""
        cleaned = [metric.strip() for metric in value]

        if any(not metric for metric in cleaned):
            raise ValueError("Metric names must not be empty.")

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Metric names must be unique.")

        return cleaned

    @field_validator("metrics")
    @classmethod
    def ensure_primary_metric_is_in_metrics(
        cls,
        value: list[str],
        info: Any,
    ) -> list[str]:
        """Ensure the primary metric is also part of the evaluated metrics."""
        primary_metric = info.data.get("primary_metric")

        if primary_metric is not None and primary_metric not in value:
            raise ValueError(
                "primary_metric must also be included in metrics."
            )

        return value


class ResourceSpec(BaseModel):
    """Describes optional resource requirements for training."""

    model_config = ConfigDict(extra="forbid")

    cpu: float | None = Field(default=None, gt=0)
    memory: str | None = None
    gpu: bool = False
    max_runtime_minutes: int | None = Field(default=None, gt=0)

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, value: str | None) -> str | None:
        """Reject empty memory specifications."""
        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError("Memory specification must not be empty.")

        return value


class ReproducibilitySpec(BaseModel):
    """Defines reproducibility-related training settings."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int = Field(default=42, ge=0)


class TrainingSpec(BaseModel):
    """
    Machine-readable contract produced by the AIForge Planner.

    TrainingSpec records the Planner's decisions about what should be
    trained and how it should be evaluated. It does not execute training,
    perform governance checks, or contain framework-specific pipeline code.

    The downstream flow is:

        TrainingSpec
            -> Governance / Validation
            -> Pipeline Compiler
            -> Template Registry
            -> Training execution
    """

    model_config = ConfigDict(extra="forbid")

    spec_version: str = Field(default="1.0", min_length=1)

    task: TaskSpec
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    preprocessing: PreprocessingSpec = Field(
        default_factory=PreprocessingSpec
    )
    split: SplitSpec
    model: ModelSpec
    evaluation: EvaluationSpec
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    reproducibility: ReproducibilitySpec = Field(
        default_factory=ReproducibilitySpec
    )

    @field_validator("spec_version")
    @classmethod
    def validate_spec_version(cls, value: str) -> str:
        """Reject empty or whitespace-only schema versions."""
        value = value.strip()

        if not value:
            raise ValueError("spec_version must not be empty.")

        return value