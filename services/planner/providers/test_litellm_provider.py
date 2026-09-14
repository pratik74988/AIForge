from unittest.mock import patch

import pytest

from ..exceptions import LLMConfigurationError
from ..prompt import PlannerPrompt
from .litellm_provider import LiteLLMProvider
from unittest.mock import patch

PATCH_TARGET = f"{LiteLLMProvider.__module__}.completion"

def make_prompt() -> PlannerPrompt:
    return PlannerPrompt(
        system="You are a test planner.",
        user="Create a test plan.",
    )


def make_response(content: str):
    class Message:
        def __init__(self, content: str):
            self.content = content

    class Choice:
        def __init__(self, content: str):
            self.message = Message(content)

    class Response:
        def __init__(self, content: str):
            self.choices = [Choice(content)]

    return Response(content)


def test_provider_sends_prompt_to_litellm():
    provider = LiteLLMProvider(
        model="test/model",
        api_key="test-key",
        temperature=0.2,
        max_tokens=500,
    )

    prompt = make_prompt()

    with patch(
        PATCH_TARGET
    ) as mock_completion:
        mock_completion.return_value = make_response(
            '{"test": "response"}'
        )

        result = provider.generate(prompt)

    assert result == '{"test": "response"}'

    mock_completion.assert_called_once_with(
        model="test/model",
        messages=[
            {
                "role": "system",
                "content": "You are a test planner.",
            },
            {
                "role": "user",
                "content": "Create a test plan.",
            },
        ],
        temperature=0.2,
        api_key="test-key",
        max_tokens=500,
    )


def test_provider_omits_optional_arguments_when_not_configured():
    provider = LiteLLMProvider(model="test/model")

    with patch(
        PATCH_TARGET
    ) as mock_completion:
        mock_completion.return_value = make_response("response")

        result = provider.generate(make_prompt())

    assert result == "response"

    mock_completion.assert_called_once_with(
        model="test/model",
        messages=[
            {
                "role": "system",
                "content": "You are a test planner.",
            },
            {
                "role": "user",
                "content": "Create a test plan.",
            },
        ],
        temperature=0.0,
    )


def test_provider_rejects_empty_model():
    with pytest.raises(LLMConfigurationError):
        LiteLLMProvider(model="")


def test_provider_rejects_whitespace_model():
    with pytest.raises(LLMConfigurationError):
        LiteLLMProvider(model="   ")


def test_provider_raises_when_litellm_request_fails():
    provider = LiteLLMProvider(model="test/model")

    with patch(
        PATCH_TARGET,
        side_effect=Exception("API failure"),
    ):
        with pytest.raises(RuntimeError, match="LiteLLM request failed"):
            provider.generate(make_prompt())


def test_provider_rejects_response_without_choices():
    provider = LiteLLMProvider(model="test/model")

    class EmptyResponse:
        choices = []

    with patch(
        PATCH_TARGET,
        return_value=EmptyResponse(),
    ):
        with pytest.raises(RuntimeError, match="without assistant content"):
            provider.generate(make_prompt())


def test_provider_rejects_empty_content():
    provider = LiteLLMProvider(model="test/model")

    with patch(
        PATCH_TARGET,
        return_value=make_response(""),
    ):
        with pytest.raises(RuntimeError, match="empty assistant content"):
            provider.generate(make_prompt())