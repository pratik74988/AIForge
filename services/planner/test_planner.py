from unittest.mock import Mock

from .context import KnowledgeContext, PlanningContext
from .knowledge_service import KnowledgeService
from .models.planner_result import PlannerResult
from .planner import Planner


def make_context() -> PlanningContext:
    return PlanningContext(
        user_requirement="Predict customer churn.",
        dataset_profile={
            "target": {
                "name": "churn",
                "task": "classification",
            }
        },
    )


def make_result() -> PlannerResult:
    return PlannerResult.model_validate(
        {
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
    )


def test_planner_delegates_to_llm_service():
    llm_service = Mock()
    llm_service.generate.return_value = make_result()

    planner = Planner(llm_service)
    context = make_context()

    result = planner.plan(context)

    assert result == llm_service.generate.return_value
    llm_service.generate.assert_called_once_with(context)


def test_planner_retrieves_knowledge_before_llm_generation():
    llm_service = Mock()
    llm_service.generate.return_value = make_result()

    knowledge_service = Mock(spec=KnowledgeService)
    knowledge_service.retrieve.return_value = KnowledgeContext(
        items=(
            "Use stratified splitting for classification.",
        ),
        metadata=(
            {
                "topic": "splitting",
            },
        ),
    )

    planner = Planner(
        llm_service=llm_service,
        knowledge_service=knowledge_service,
    )

    original_context = make_context()

    result = planner.plan(original_context)

    assert result == llm_service.generate.return_value

    knowledge_service.retrieve.assert_called_once_with(
        original_context
    )

    generated_context = llm_service.generate.call_args.args[0]

    assert generated_context.user_requirement == (
        original_context.user_requirement
    )

    assert generated_context.dataset_profile == (
        original_context.dataset_profile
    )

    assert generated_context.knowledge_context == KnowledgeContext(
        items=(
            "Use stratified splitting for classification.",
        ),
        metadata=(
            {
                "topic": "splitting",
            },
        ),
    )


def test_planner_does_not_require_knowledge_service():
    llm_service = Mock()
    llm_service.generate.return_value = make_result()

    planner = Planner(llm_service=llm_service)

    context = make_context()

    planner.plan(context)

    llm_service.generate.assert_called_once_with(context)