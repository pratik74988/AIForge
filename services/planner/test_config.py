import pytest

from .config import PlannerConfig
from .exceptions import LLMConfigurationError


def test_config_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("AIFORGE_LLM_MODEL", "groq/test-model")
    monkeypatch.setenv("AIFORGE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("AIFORGE_LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("AIFORGE_LLM_MAX_TOKENS", "4096")

    config = PlannerConfig()

    assert config.model == "groq/test-model"
    assert config.api_key == "test-key"
    assert config.temperature == 0.2
    assert config.max_tokens == 4096


def test_config_accepts_explicit_values():
    config = PlannerConfig(
        model="test/model",
        api_key="test-key",
        temperature=0.5,
        max_tokens=1000,
    )

    assert config.model == "test/model"
    assert config.api_key == "test-key"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000


def test_config_requires_model(monkeypatch):
    monkeypatch.delenv("AIFORGE_LLM_MODEL", raising=False)

    with pytest.raises(
        LLMConfigurationError,
        match="AIFORGE_LLM_MODEL must be configured",
    ):
        PlannerConfig()


def test_config_rejects_invalid_temperature(monkeypatch):
    monkeypatch.setenv("AIFORGE_LLM_MODEL", "test/model")
    monkeypatch.setenv("AIFORGE_LLM_TEMPERATURE", "invalid")

    with pytest.raises(
        LLMConfigurationError,
        match="AIFORGE_LLM_TEMPERATURE must be a valid number",
    ):
        PlannerConfig()


@pytest.mark.parametrize("temperature", ["-0.1", "2.1"])
def test_config_rejects_temperature_outside_range(
    monkeypatch,
    temperature,
):
    monkeypatch.setenv("AIFORGE_LLM_MODEL", "test/model")
    monkeypatch.setenv("AIFORGE_LLM_TEMPERATURE", temperature)

    with pytest.raises(
        LLMConfigurationError,
        match="between 0.0 and 2.0",
    ):
        PlannerConfig()


def test_config_rejects_invalid_max_tokens(monkeypatch):
    monkeypatch.setenv("AIFORGE_LLM_MODEL", "test/model")
    monkeypatch.setenv("AIFORGE_LLM_MAX_TOKENS", "invalid")

    with pytest.raises(
        LLMConfigurationError,
        match="AIFORGE_LLM_MAX_TOKENS must be a valid integer",
    ):
        PlannerConfig()


def test_config_rejects_non_positive_max_tokens(monkeypatch):
    monkeypatch.setenv("AIFORGE_LLM_MODEL", "test/model")
    monkeypatch.setenv("AIFORGE_LLM_MAX_TOKENS", "0")

    with pytest.raises(
        LLMConfigurationError,
        match="greater than zero",
    ):
        PlannerConfig()


def test_config_allows_missing_optional_values(monkeypatch):
    monkeypatch.setenv("AIFORGE_LLM_MODEL", "test/model")
    monkeypatch.delenv("AIFORGE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("AIFORGE_LLM_MAX_TOKENS", raising=False)

    config = PlannerConfig()

    assert config.model == "test/model"
    assert config.api_key is None
    assert config.temperature == 0.0
    assert config.max_tokens is None