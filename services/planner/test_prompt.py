"""Tests for Planner prompt construction."""

import pytest

from .context import KnowledgeContext, PlanningContext
from .prompt import PlannerPrompt, PlannerPromptBuilder


def make_context() -> PlanningContext:
    """Create a representative planning context."""
    return PlanningContext(
        user_requirement=(
            "Predict customer churn and prioritize recall."
        ),
        dataset_profile={
            "target": {
                "name": "churn",
                "task": "classification",
                "imbalance_ratio": 4.8,
            },
            "quality": {
                "duplicate_rows": 10,
            },
        },
        knowledge_context=KnowledgeContext(
            items=(
                "Stratified splitting is appropriate for imbalanced "
                "classification.",
                "Recall should be considered when missing positive "
                "cases is costly.",
            ),
        ),
    )


def test_build_returns_planner_prompt() -> None:
    """Builder should return a PlannerPrompt."""
    prompt = PlannerPromptBuilder().build(make_context())

    assert isinstance(prompt, PlannerPrompt)


def test_system_prompt_contains_planner_instructions() -> None:
    """System prompt should define the Planner's role."""
    prompt = PlannerPromptBuilder().build(make_context())

    assert "AIForge ML Planning Engine" in prompt.system
    assert "Dataset Analyzer" in prompt.system
    assert "TrainingSpec" in prompt.system


def test_user_requirement_is_included() -> None:
    """User requirement must be passed to the LLM."""
    prompt = PlannerPromptBuilder().build(make_context())

    assert "Predict customer churn and prioritize recall." in prompt.user


def test_dataset_profile_is_included() -> None:
    """Dataset Analyzer facts must be included."""
    prompt = PlannerPromptBuilder().build(make_context())

    assert '"churn"' in prompt.user
    assert '"classification"' in prompt.user
    assert "4.8" in prompt.user


def test_knowledge_context_is_included() -> None:
    """Retrieved Knowledge Base information must be included."""
    prompt = PlannerPromptBuilder().build(make_context())

    assert "Stratified splitting" in prompt.user
    assert "missing positive cases" in prompt.user


def test_empty_knowledge_context_is_handled() -> None:
    """Planner should still work when no KB context exists."""
    context = PlanningContext(
        user_requirement="Predict churn.",
        dataset_profile={"target": "churn"},
    )

    prompt = PlannerPromptBuilder().build(context)

    assert "No additional Knowledge Base context was retrieved." in prompt.user


def test_prompt_is_deterministic() -> None:
    """Identical contexts should produce identical prompts."""
    builder = PlannerPromptBuilder()

    first = builder.build(make_context())
    second = builder.build(make_context())

    assert first == second


def test_invalid_dataset_profile_type_is_rejected() -> None:
    """Unsupported DatasetProfile representations should fail clearly."""
    context = PlanningContext(
        user_requirement="Predict churn.",
        dataset_profile="not-a-dataset-profile",
    )

    with pytest.raises(
        TypeError,
        match="dataset_profile must be a Pydantic model or dictionary",
    ):
        PlannerPromptBuilder().build(context)