from ..config import PlannerConfig
from .factory import create_litellm_provider
from .litellm_provider import LiteLLMProvider


def test_create_litellm_provider():
    config = PlannerConfig(
        model="groq/test-model",
        api_key="test-key",
        temperature=0.2,
        max_tokens=2048,
    )

    provider = create_litellm_provider(config)

    assert isinstance(provider, LiteLLMProvider)
    assert provider._model == "groq/test-model"
    assert provider._api_key == "test-key"
    assert provider._temperature == 0.2
    assert provider._max_tokens == 2048


def test_create_litellm_provider_passes_optional_values():
    config = PlannerConfig(
        model="test/model",
    )

    provider = create_litellm_provider(config)

    assert provider._model == "test/model"
    assert provider._api_key is None
    assert provider._temperature == 0.0
    assert provider._max_tokens is None