"""Tests for the provider-independent LLM client interface."""

import pytest

from .llm_client import LLMClient
from .models.planner_result import PlannerResult
from .models.test_training_spec import make_valid_training_spec
from .prompt import PlannerPrompt


class FakeLLMClient(LLMClient):
    """Fake LLM client used to verify the interface contract."""

    def __init__(self) -> None:
        self.received_prompt: PlannerPrompt | None = None

    def generate(self, prompt: PlannerPrompt) -> PlannerResult:
        self.received_prompt = prompt

        return PlannerResult(
            proposed_spec=make_valid_training_spec(),
        )


def make_prompt() -> PlannerPrompt:
    """Create a representative Planner prompt."""
    return PlannerPrompt(
        system="You are the AIForge ML Planning Engine.",
        user="Predict customer churn.",
    )


def test_llm_client_can_be_implemented() -> None:
    """A concrete implementation should satisfy the interface."""
    client = FakeLLMClient()

    result = client.generate(make_prompt())

    assert isinstance(result, PlannerResult)
    assert result.proposed_spec.task.target == "churn"


def test_prompt_is_passed_to_client() -> None:
    """The client should receive the complete PlannerPrompt."""
    client = FakeLLMClient()
    prompt = make_prompt()

    client.generate(prompt)

    assert client.received_prompt is prompt


def test_llm_client_is_abstract() -> None:
    """The abstract client must not be directly instantiated."""
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]