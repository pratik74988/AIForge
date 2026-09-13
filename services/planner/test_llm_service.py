from unittest.mock import Mock

import pytest

from .context import PlanningContext
from .llm_service import LLMService
from .models.planner_result import PlannerResult
from .prompt import PlannerPromptBuilder


def make_context() -> PlanningContext:
    return PlanningContext(
        user_requirement="Predict customer churn.",
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


def make_valid_response() -> dict:
    return {
        "recommendations": [],
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
                "primary_metric": "f1",
                "metrics": ["f1"],
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


def test_llm_service_generates_planner_result():
    provider = Mock()
    provider.generate.return_value = make_valid_response()

    service = LLMService(provider)

    result = service.generate(make_context())

    assert isinstance(result, PlannerResult)
    assert result.proposed_spec.task.target == "churn"
    provider.generate.assert_called_once()


def test_llm_service_builds_prompt_before_calling_provider():
    provider = Mock()
    provider.generate.return_value = make_valid_response()

    prompt_builder = Mock(spec=PlannerPromptBuilder)
    prompt = Mock()

    prompt_builder.build.return_value = prompt

    service = LLMService(
        provider,
        prompt_builder=prompt_builder,
    )

    context = make_context()

    service.generate(context)

    prompt_builder.build.assert_called_once_with(context)
    provider.generate.assert_called_once_with(prompt)


def test_llm_service_converts_provider_errors_to_llm_error():
    provider = Mock()
    provider.generate.side_effect = RuntimeError("API unavailable")

    service = LLMService(provider)

    with pytest.raises(Exception, match="LLM provider request failed"):
        service.generate(make_context())


def test_llm_service_preserves_valid_planner_result():
    expected = make_valid_response()

    provider = Mock()
    provider.generate.return_value = expected

    service = LLMService(provider)

    result = service.generate(make_context())

    assert result.model_dump() == PlannerResult.model_validate(
        expected
    ).model_dump()