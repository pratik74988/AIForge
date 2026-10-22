from aiforge_template_registry import get_template_class

from compiler.models.compiled_pipeline import CompiledPipeline


def test_compiled_pipeline_creation():
    template_class = get_template_class(
        "sklearn",
        "classification",
        "random_forest",
    )

    pipeline = CompiledPipeline(
        template="sklearn",
        algorithm="random_forest",
        preprocessing_steps=[
            "numeric_imputation",
            "categorical_encoding",
        ],
        hyperparameters={
            "n_estimators": 100,
        },
        evaluation_metrics=[
            "accuracy",
            "f1_macro",
        ],
        primary_metric="accuracy",
        template_class=template_class,
    )

    assert pipeline.template == "sklearn"
    assert pipeline.algorithm == "random_forest"

    assert pipeline.preprocessing_steps == [
        "numeric_imputation",
        "categorical_encoding",
    ]

    assert pipeline.hyperparameters == {
        "n_estimators": 100,
    }

    assert pipeline.evaluation_metrics == [
        "accuracy",
        "f1_macro",
    ]

    assert pipeline.primary_metric == "accuracy"
    assert pipeline.template_class.__name__ == "RandomForestTemplate"


def test_compiled_pipeline_accepts_arbitrary_template_class():
    template_class = get_template_class(
        "sklearn",
        "classification",
        "random_forest",
    )

    pipeline = CompiledPipeline(
        template="sklearn",
        algorithm="random_forest",
        template_class=template_class,
        primary_metric="accuracy",
    )

    assert pipeline.template_class is template_class


def test_compiled_pipeline_defaults():
    template_class = get_template_class(
        "sklearn",
        "classification",
        "random_forest",
    )

    pipeline = CompiledPipeline(
        template="sklearn",
        algorithm="random_forest",
        template_class=template_class,
        primary_metric="accuracy",
    )

    assert pipeline.preprocessing_steps == []
    assert pipeline.hyperparameters == {}
    assert pipeline.evaluation_metrics == []

def test_compiled_pipeline_accepts_pipeline():
    template_class = get_template_class(
        "sklearn",
        "classification",
        "random_forest",
    )

    sentinel_pipeline = object()

    pipeline = CompiledPipeline(
        template="sklearn",
        algorithm="random_forest",
        template_class=template_class,
        primary_metric="accuracy",
        pipeline=sentinel_pipeline,
    )

    assert pipeline.pipeline is sentinel_pipeline