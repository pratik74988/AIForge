import pandas as pd

from compiler.compiler import PipelineCompiler
from planner.models.planner_result import PlannerResult
from policy.compliance import check_compliance


def make_planner_result() -> PlannerResult:
    return PlannerResult.model_validate(
        {
            "recommendations": [],
            "proposed_spec": {
                "spec_version": "1.0",
                "task": {
                    "type": "classification",
                    "target": "target",
                },
                "features": {
                    "include": None,
                    "exclude": [],
                    "numerical": ["age", "income"],
                    "categorical": ["city"],
                },
                "preprocessing": {
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
                            "name": "categorical_encoding",
                            "operation": "encode",
                            "columns": "categorical",
                            "parameters": {
                                "method": "one_hot",
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
                "split": {
                    "strategy": "random",
                    "test_size": 0.2,
                    "validation_size": 0.1,
                },
                "model": {
                    "template": "sklearn",
                    "algorithm": "random_forest",
                    "hyperparameters": {
                        "n_estimators": 50,
                    },
                },
                "evaluation": {
                    "metrics": ["accuracy", "f1_macro"],
                    "primary_metric": "f1_macro",
                },
                "resources": {},
                "reproducibility": {
                    "random_seed": 42,
                },
            },
            "requires_approval": False,
        }
    )


def test_planner_result_flows_through_compliance_and_compiler():
    planner_result = make_planner_result()
    spec = planner_result.proposed_spec

    compliance_result = check_compliance(spec)

    assert compliance_result.approved is True
    assert compliance_result.requires_human_approval is False
    assert compliance_result.errors == []

    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    assert compiled.template == "sklearn"
    assert compiled.algorithm == "random_forest"
    assert compiled.hyperparameters["n_estimators"] == 50
    assert compiled.primary_metric == "f1_macro"
    assert compiled.evaluation_metrics == [
        "accuracy",
        "f1_macro",
    ]

    assert list(compiled.pipeline.named_steps) == [
        "preprocessing",
        "model",
    ]


def test_planner_result_produces_executable_pipeline():
    planner_result = make_planner_result()
    spec = planner_result.proposed_spec

    compliance_result = check_compliance(spec)
    compiled = PipelineCompiler().compile(
        spec,
        compliance_result,
    )

    X = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "income": [20_000, 30_000, 40_000, 50_000, 60_000, 70_000],
            "city": [
                "Pune",
                "Mumbai",
                "Pune",
                "Mumbai",
                "Pune",
                "Mumbai",
            ],
        }
    )

    y = [0, 0, 0, 1, 1, 1]

    compiled.pipeline.fit(X, y)

    predictions = compiled.pipeline.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})

