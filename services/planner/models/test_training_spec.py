"""Tests for the AIForge TrainingSpec Pydantic models."""

import pytest
from pydantic import ValidationError

from .training_spec import (
    EvaluationSpec,
    FeatureSpec,
    ModelSpec,
    PreprocessingSpec,
    PreprocessingStep,
    ReproducibilitySpec,
    ResourceSpec,
    SplitSpec,
    SplitStrategy,
    TaskSpec,
    TaskType,
    TrainingSpec,
)


def make_valid_training_spec() -> TrainingSpec:
    """Create a minimal valid TrainingSpec for testing."""
    return TrainingSpec(
        task=TaskSpec(
            type=TaskType.CLASSIFICATION,
            target="churn",
        ),
        features=FeatureSpec(
            exclude=["customer_id"],
        ),
        preprocessing=PreprocessingSpec(
            steps=[
                PreprocessingStep(
                    name="scale_numeric",
                    operation="scale",
                    columns="numeric",
                    parameters={"method": "standard"},
                )
            ]
        ),
        split=SplitSpec(
            strategy=SplitStrategy.STRATIFIED,
            test_size=0.2,
            validation_size=0.1,
        ),
        model=ModelSpec(
            template="sklearn",
            algorithm="logistic_regression",
            hyperparameters={
                "C": 1.0,
                "penalty": "l2",
                "class_weight": "balanced",
            },
        ),
        evaluation=EvaluationSpec(
            primary_metric="recall",
            metrics=["recall", "precision", "f1"],
        ),
    )


def test_minimal_valid_training_spec() -> None:
    """A correctly constructed TrainingSpec should validate."""
    spec = make_valid_training_spec()

    assert spec.spec_version == "1.0"
    assert spec.task.type == TaskType.CLASSIFICATION
    assert spec.task.target == "churn"
    assert spec.model.template == "sklearn"
    assert spec.model.algorithm == "logistic_regression"
    assert spec.evaluation.primary_metric == "recall"


def test_training_spec_serializes_to_dict() -> None:
    """TrainingSpec should serialize into a plain dictionary."""
    spec = make_valid_training_spec()

    data = spec.model_dump()

    assert data["spec_version"] == "1.0"
    assert data["task"]["type"] == "classification"
    assert data["model"]["template"] == "sklearn"
    assert data["model"]["algorithm"] == "logistic_regression"


def test_default_values() -> None:
    """Optional sections should receive their defined defaults."""
    spec = make_valid_training_spec()

    assert spec.features.include is None
    assert spec.preprocessing.steps
    assert spec.resources.cpu is None
    assert spec.resources.memory is None
    assert spec.resources.gpu is False
    assert spec.resources.max_runtime_minutes is None
    assert spec.reproducibility.random_seed == 42


def test_empty_target_is_rejected() -> None:
    """A target name must not be empty."""
    with pytest.raises(ValidationError):
        TaskSpec(
            type=TaskType.CLASSIFICATION,
            target="   ",
        )


def test_duplicate_feature_names_are_rejected() -> None:
    """Feature lists must not contain duplicate names."""
    with pytest.raises(ValidationError):
        FeatureSpec(
            exclude=["age", "age"],
        )


def test_empty_feature_name_is_rejected() -> None:
    """Feature names must not be empty."""
    with pytest.raises(ValidationError):
        FeatureSpec(
            exclude=["age", "  "],
        )


def test_semantic_column_groups_are_accepted() -> None:
    """Supported semantic column groups should validate."""
    for group in ("numeric", "categorical", "all"):
        step = PreprocessingStep(
            name="test",
            operation="impute",
            columns=group,
        )

        assert step.columns == group


def test_invalid_semantic_column_group_is_rejected() -> None:
    """Unknown string column groups should be rejected."""
    with pytest.raises(ValidationError):
        PreprocessingStep(
            name="test",
            operation="impute",
            columns="text",
        )


def test_explicit_column_names_are_accepted() -> None:
    """Preprocessing steps should support explicit feature names."""
    step = PreprocessingStep(
        name="impute_income",
        operation="impute",
        columns=["income", "salary"],
        parameters={"strategy": "median"},
    )

    assert step.columns == ["income", "salary"]


def test_invalid_split_sizes_are_rejected() -> None:
    """Test and validation portions must leave data for training."""
    with pytest.raises(ValidationError):
        SplitSpec(
            strategy=SplitStrategy.RANDOM,
            test_size=0.7,
            validation_size=0.3,
        )


def test_invalid_individual_split_size_is_rejected() -> None:
    """Individual split proportions must be within valid bounds."""
    with pytest.raises(ValidationError):
        SplitSpec(
            strategy=SplitStrategy.RANDOM,
            test_size=1.0,
            validation_size=0.0,
        )


def test_primary_metric_must_be_evaluated() -> None:
    """The primary metric must also appear in the metrics list."""
    with pytest.raises(ValidationError):
        EvaluationSpec(
            primary_metric="recall",
            metrics=["precision", "f1"],
        )


def test_duplicate_metrics_are_rejected() -> None:
    """Evaluation metrics should not be duplicated."""
    with pytest.raises(ValidationError):
        EvaluationSpec(
            primary_metric="recall",
            metrics=["recall", "f1", "recall"],
        )


def test_empty_model_identifiers_are_rejected() -> None:
    """Template and algorithm identifiers must not be empty."""
    with pytest.raises(ValidationError):
        ModelSpec(
            template="   ",
            algorithm="logistic_regression",
        )

    with pytest.raises(ValidationError):
        ModelSpec(
            template="sklearn",
            algorithm="   ",
        )


def test_negative_random_seed_is_rejected() -> None:
    """Random seeds must be non-negative."""
    with pytest.raises(ValidationError):
        ReproducibilitySpec(random_seed=-1)


def test_invalid_cpu_resource_is_rejected() -> None:
    """CPU requirements must be positive when specified."""
    with pytest.raises(ValidationError):
        ResourceSpec(cpu=0)


def test_invalid_runtime_is_rejected() -> None:
    """Maximum runtime must be positive when specified."""
    with pytest.raises(ValidationError):
        ResourceSpec(max_runtime_minutes=0)


def test_unknown_training_spec_fields_are_rejected() -> None:
    """The contract must reject fields that are not part of the schema."""
    with pytest.raises(ValidationError):
        TrainingSpec(
            task=TaskSpec(
                type=TaskType.CLASSIFICATION,
                target="churn",
            ),
            split=SplitSpec(
                strategy=SplitStrategy.STRATIFIED,
                test_size=0.2,
            ),
            model=ModelSpec(
                template="sklearn",
                algorithm="logistic_regression",
            ),
            evaluation=EvaluationSpec(
                primary_metric="recall",
                metrics=["recall"],
            ),
            unexpected_field="not_allowed",
        )


def test_training_spec_round_trip() -> None:
    """A serialized TrainingSpec should reconstruct successfully."""
    spec = make_valid_training_spec()

    data = spec.model_dump()
    reconstructed = TrainingSpec.model_validate(data)

    assert reconstructed == spec