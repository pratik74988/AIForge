import pytest
import pandas as pd

from compiler.compiler import PipelineCompiler
from planner.models.training_spec import TrainingSpec, FeatureSpec
from policy.compliance import check_compliance


def make_valid_spec() -> TrainingSpec:
    return TrainingSpec(
        spec_version="1.0",
        task={
            "type": "classification",
            "target": "target",
        },
        features=FeatureSpec(
            numerical=["age", "income"],
            categorical=["city"],
        ),
        preprocessing={
            "steps": [],
        },
        split={
            "strategy": "random",
            "test_size": 0.2,
        },
        model={
            "template": "sklearn",
            "algorithm": "random_forest",
            "hyperparameters": {
                "n_estimators": 100,
            },
        },
        evaluation={
            "metrics": ["accuracy"],
            "primary_metric": "accuracy",
        },
        resources={},
        reproducibility={},
    )


def test_compiler_resolves_template_from_registry():
    spec = make_valid_spec()
    compliance_result = check_compliance(spec)

    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    assert compiled.template == "sklearn"
    assert compiled.algorithm == "random_forest"
    assert compiled.hyperparameters["n_estimators"] == 100
    assert compiled.template_class.__name__ == "RandomForestTemplate"


def test_compiler_rejects_non_compliant_spec():
    spec = make_valid_spec()

    spec.model.template = "unsupported_framework"

    compliance_result = check_compliance(spec)

    assert compliance_result.approved is False

    with pytest.raises(
        ValueError,
        match="compliance approval was not granted",
    ):
        PipelineCompiler().compile(
            spec,
            compliance_result,
        )


def test_compiler_rejects_spec_requiring_human_approval():
    spec = make_valid_spec()

    spec.resources.cpu = 12

    compliance_result = check_compliance(spec)

    assert compliance_result.approved is True
    assert compliance_result.requires_human_approval is True

    with pytest.raises(
        ValueError,
        match="human approval is required",
    ):
        PipelineCompiler().compile(
            spec,
            compliance_result,
        )


def test_compiler_builds_pipeline():
    spec = make_valid_spec()
    compliance_result = check_compliance(spec)

    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    assert compiled.pipeline is not None
    assert compiled.pipeline.named_steps["preprocessing"] is not None
    assert compiled.pipeline.named_steps["model"].__class__.__name__ == (
        "RandomForestClassifier"
    )


def test_compiled_pipeline_can_fit_and_predict():
    spec = make_valid_spec()

    compliance_result = check_compliance(spec)

    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    X = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "income": [20_000, 30_000, 40_000, 50_000, 60_000, 70_000],
            "city": ["Pune", "Mumbai", "Pune", "Mumbai", "Pune", "Mumbai"],
        }
    )

    y = [0, 0, 0, 1, 1, 1]

    compiled.pipeline.fit(X, y)

    predictions = compiled.pipeline.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_compiled_pipeline_contains_preprocessing_and_model():
    spec = make_valid_spec()

    compliance_result = check_compliance(spec)

    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    assert list(compiled.pipeline.named_steps) == [
        "preprocessing",
        "model",
    ]


def make_regression_spec() -> TrainingSpec:
    return TrainingSpec(
        spec_version="1.0",
        task={
            "type": "regression",
            "target": "target",
        },
        features=FeatureSpec(
            numerical=["age", "income"],
            categorical=[],
        ),
        preprocessing={
            "steps": [
                {
                    "name": "numeric_imputation",
                    "operation": "impute",
                    "columns": "numeric",
                    "parameters": {
                        "strategy": "median",
                    },
                },
                {
                    "name": "numerical_scaling",
                    "operation": "scale",
                    "columns": "numeric",
                    "parameters": {
                        "method": "standard",
                    },
                },
            ],
        },
        split={
            "strategy": "random",
            "test_size": 0.2,
        },
        model={
            "template": "sklearn",
            "algorithm": "ridge",
            "hyperparameters": {
                "alpha": 1.0,
            },
        },
        evaluation={
            "metrics": ["mae", "r2"],
            "primary_metric": "r2",
        },
        resources={},
        reproducibility={},
    )


def test_compiler_builds_regression_pipeline():
    spec = make_regression_spec()

    compliance_result = check_compliance(spec)

    assert compliance_result.approved is True

    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    assert compiled.template == "sklearn"
    assert compiled.algorithm == "ridge"
    assert compiled.template_class.__name__ == "RidgeTemplate"

    assert list(compiled.pipeline.named_steps) == [
        "preprocessing",
        "model",
    ]
    assert compiled.pipeline.named_steps["model"].__class__.__name__ == (
        "Ridge"
    )


def test_compiled_regression_pipeline_can_fit_and_predict():
    spec = make_regression_spec()

    compliance_result = check_compliance(spec)

    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    X = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "income": [20_000, 30_000, 40_000, 50_000, 60_000, 70_000],
        }
    )

    y = [200, 300, 400, 500, 600, 700]

    compiled.pipeline.fit(X, y)

    predictions = compiled.pipeline.predict(X)

    assert len(predictions) == len(X)


def test_compiled_pipeline_preserves_spec_metadata():
    spec = make_valid_spec()

    compliance_result = check_compliance(spec)

    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    assert compiled.template == spec.model.template
    assert compiled.algorithm == spec.model.algorithm
    assert compiled.preprocessing_steps == [
        step.name for step in spec.preprocessing.steps
    ]
    assert compiled.hyperparameters == spec.model.hyperparameters
    assert compiled.evaluation_metrics == spec.evaluation.metrics
    assert compiled.primary_metric == spec.evaluation.primary_metric

