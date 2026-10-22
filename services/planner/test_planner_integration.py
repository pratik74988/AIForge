from unittest.mock import Mock

from .context import PlanningContext
from .llm_service import LLMService
from .models.planner_result import PlannerResult
from .planner import Planner


def make_context() -> PlanningContext:
    return PlanningContext(
        user_requirement="Predict customer churn and prioritize recall.",
        dataset_profile={
            "dataset": {
                "rows": 1000,
                "columns": 10,
                "analyzed_rows": 1000,
                "sampled": False,
            },
            "columns": [],
            "quality": {
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
                "constant_columns": [],
                "high_missing_columns": [],
            },
            "relationships": {
                "highly_correlated_pairs": [],
            },
            "warnings": [],
        },
    )


def make_llm_response() -> dict:
    return {
        "recommendations": [
            {
                "type": "evaluation",
                "title": "Prioritize recall",
                "reasoning": (
                    "The user explicitly wants customer churn "
                    "cases prioritized."
                ),
                "decision": "Use recall as the primary metric.",
                "requires_approval": True,
            }
        ],
        "proposed_spec": {
            "spec_version": "1.0",
            "task": {
                "type": "classification",
                "target": "churn",
            },
            "features": {
                "include": None,
                "exclude": [],
            },
            "preprocessing": {
                "steps": [],
            },
            "split": {
                "strategy": "stratified",
                "test_size": 0.2,
                "validation_size": 0.1,
            },
            "model": {
                "template": "sklearn",
                "algorithm": "logistic_regression",
                "hyperparameters": {},
            },
            "evaluation": {
                "primary_metric": "recall",
                "metrics": [
                    "recall",
                    "precision",
                    "f1",
                ],
            },
            "resources": {
                "cpu": None,
                "memory": None,
                "gpu": False,
                "max_runtime_minutes": None,
            },
            "reproducibility": {
                "random_seed": 42,
            },
        },
        "requires_approval": True,
    }


def test_planner_end_to_end_with_mocked_provider():
    provider = Mock()
    provider.generate.return_value = make_llm_response()

    llm_service = LLMService(provider)
    planner = Planner(llm_service)

    result = planner.plan(make_context())

    assert isinstance(result, PlannerResult)

    assert result.proposed_spec.task.type == "classification"
    assert result.proposed_spec.task.target == "churn"

    assert result.proposed_spec.model.template == "sklearn"
    assert result.proposed_spec.model.algorithm == "logistic_regression"

    assert result.proposed_spec.evaluation.primary_metric == "recall"

    assert len(result.recommendations) == 1
    assert result.recommendations[0].type == "evaluation"

    assert result.requires_approval is True

    provider.generate.assert_called_once()