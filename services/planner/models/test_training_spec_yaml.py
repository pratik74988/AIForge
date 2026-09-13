"""Tests for TrainingSpec YAML serialization and deserialization."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from .test_training_spec import make_valid_training_spec
from .training_spec import TrainingSpec
from .training_spec_yaml import (
    TrainingSpecYAMLError,
    load_training_spec_yaml,
    save_training_spec_yaml,
    training_spec_from_yaml,
    training_spec_to_yaml,
)


def test_training_spec_to_yaml() -> None:
    """A TrainingSpec should serialize to readable YAML."""
    spec = make_valid_training_spec()

    yaml_content = training_spec_to_yaml(spec)

    assert isinstance(yaml_content, str)
    assert "spec_version: '1.0'" in yaml_content
    assert "task:" in yaml_content
    assert "target: churn" in yaml_content
    assert "template: sklearn" in yaml_content
    assert "algorithm: logistic_regression" in yaml_content


def test_training_spec_from_yaml() -> None:
    """Valid YAML should deserialize into a TrainingSpec."""
    yaml_content = """
spec_version: "1.0"

task:
  type: classification
  target: churn

features:
  exclude:
    - customer_id

preprocessing:
  steps:
    - name: scale_numeric
      operation: scale
      columns: numeric
      parameters:
        method: standard

split:
  strategy: stratified
  test_size: 0.2
  validation_size: 0.1

model:
  template: sklearn
  algorithm: logistic_regression
  hyperparameters:
    C: 1.0
    penalty: l2
    class_weight: balanced

evaluation:
  primary_metric: recall
  metrics:
    - recall
    - precision
    - f1

resources:
  gpu: false

reproducibility:
  random_seed: 42
"""

    spec = training_spec_from_yaml(yaml_content)

    assert isinstance(spec, TrainingSpec)
    assert spec.task.target == "churn"
    assert spec.model.template == "sklearn"
    assert spec.model.algorithm == "logistic_regression"
    assert spec.evaluation.primary_metric == "recall"
    assert spec.reproducibility.random_seed == 42


def test_yaml_round_trip() -> None:
    """Serializing and deserializing a TrainingSpec should preserve it."""
    original = make_valid_training_spec()

    yaml_content = training_spec_to_yaml(original)
    reconstructed = training_spec_from_yaml(yaml_content)

    assert reconstructed == original


def test_save_and_load_training_spec_yaml(tmp_path: Path) -> None:
    """TrainingSpec should be correctly saved to and loaded from a file."""
    original = make_valid_training_spec()
    destination = tmp_path / "training_spec.yaml"

    save_training_spec_yaml(original, destination)

    assert destination.exists()

    loaded = load_training_spec_yaml(destination)

    assert loaded == original


def test_empty_yaml_is_rejected() -> None:
    """An empty YAML document should not be accepted."""
    with pytest.raises(TrainingSpecYAMLError, match="must not be empty"):
        training_spec_from_yaml("")


def test_whitespace_only_yaml_is_rejected() -> None:
    """Whitespace-only YAML should not be accepted."""
    with pytest.raises(TrainingSpecYAMLError, match="must not be empty"):
        training_spec_from_yaml("   \n\t")


def test_non_mapping_yaml_is_rejected() -> None:
    """The YAML root must be a mapping/object."""
    with pytest.raises(
        TrainingSpecYAMLError,
        match="must contain a mapping/object",
    ):
        training_spec_from_yaml("- classification\n- churn")


def test_malformed_yaml_is_rejected() -> None:
    """Malformed YAML should produce a TrainingSpecYAMLError."""
    malformed_yaml = """
task:
  type: classification
  target: churn
  invalid: [unclosed
"""

    with pytest.raises(TrainingSpecYAMLError, match="Invalid TrainingSpec YAML"):
        training_spec_from_yaml(malformed_yaml)


def test_schema_invalid_yaml_is_rejected() -> None:
    """Valid YAML that violates the TrainingSpec schema must be rejected."""
    invalid_yaml = """
spec_version: "1.0"

task:
  type: classification
  target: churn

split:
  strategy: stratified
  test_size: 0.2

model:
  template: sklearn
  algorithm: logistic_regression

evaluation:
  primary_metric: recall
  metrics:
    - precision
"""

    with pytest.raises(
        TrainingSpecYAMLError,
        match="TrainingSpec validation failed",
    ):
        training_spec_from_yaml(invalid_yaml)


def test_unknown_yaml_fields_are_rejected() -> None:
    """Unknown fields should not bypass the Pydantic contract."""
    yaml_content = """
spec_version: "1.0"

task:
  type: classification
  target: churn

split:
  strategy: stratified
  test_size: 0.2

model:
  template: sklearn
  algorithm: logistic_regression

evaluation:
  primary_metric: recall
  metrics:
    - recall

unexpected_field: true
"""

    with pytest.raises(
        TrainingSpecYAMLError,
        match="TrainingSpec validation failed",
    ):
        training_spec_from_yaml(yaml_content)


def test_invalid_yaml_file_path_is_rejected(tmp_path: Path) -> None:
    """Loading a nonexistent file should raise a useful error."""
    missing_file = tmp_path / "does_not_exist.yaml"

    with pytest.raises(TrainingSpecYAMLError, match="Failed to read"):
        load_training_spec_yaml(missing_file)


def test_save_creates_expected_yaml_content(tmp_path: Path) -> None:
    """The saved file should contain valid YAML representing the spec."""
    spec = make_valid_training_spec()
    destination = tmp_path / "training_spec.yaml"

    save_training_spec_yaml(spec, destination)

    content = destination.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)

    assert parsed["spec_version"] == "1.0"
    assert parsed["task"]["type"] == "classification"
    assert parsed["model"]["template"] == "sklearn"
    assert parsed["model"]["algorithm"] == "logistic_regression"


def test_invalid_output_path_is_rejected(tmp_path: Path) -> None:
    """Writing to an invalid destination should raise a useful error."""
    spec = make_valid_training_spec()
    invalid_destination = tmp_path / "missing_directory" / "training_spec.yaml"

    with pytest.raises(TrainingSpecYAMLError, match="Failed to write"):
        save_training_spec_yaml(spec, invalid_destination)