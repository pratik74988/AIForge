"""
YAML serialization utilities for the AIForge TrainingSpec.

This module provides conversion between validated TrainingSpec objects
and their YAML representation.

Validation remains the responsibility of the Pydantic models. YAML is
only the external serialization format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .training_spec import TrainingSpec


class TrainingSpecYAMLError(ValueError):
    """Raised when TrainingSpec YAML cannot be serialized or parsed."""


def training_spec_to_yaml(spec: TrainingSpec) -> str:
    """
    Serialize a validated TrainingSpec to YAML.

    Args:
        spec: Validated TrainingSpec instance.

    Returns:
        A YAML string representing the TrainingSpec.

    Raises:
        TrainingSpecYAMLError: If serialization fails.
    """
    try:
        data = spec.model_dump(mode="json")

        return yaml.safe_dump(
            data,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise TrainingSpecYAMLError(
            f"Failed to serialize TrainingSpec to YAML: {exc}"
        ) from exc


def training_spec_from_yaml(yaml_content: str) -> TrainingSpec:
    """
    Parse and validate a TrainingSpec from YAML.

    Args:
        yaml_content: YAML string containing a TrainingSpec.

    Returns:
        A validated TrainingSpec instance.

    Raises:
        TrainingSpecYAMLError: If the YAML is malformed or does not
            conform to the TrainingSpec schema.
    """
    if not yaml_content.strip():
        raise TrainingSpecYAMLError("TrainingSpec YAML must not be empty.")

    try:
        data: Any = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        raise TrainingSpecYAMLError(
            f"Invalid TrainingSpec YAML: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise TrainingSpecYAMLError(
            "TrainingSpec YAML must contain a mapping/object at the root."
        )

    try:
        return TrainingSpec.model_validate(data)
    except ValueError as exc:
        raise TrainingSpecYAMLError(
            f"TrainingSpec validation failed: {exc}"
        ) from exc


def save_training_spec_yaml(
    spec: TrainingSpec,
    path: str | Path,
) -> None:
    """
    Serialize a TrainingSpec and save it to a YAML file.

    Args:
        spec: Validated TrainingSpec instance.
        path: Destination file path.

    Raises:
        TrainingSpecYAMLError: If serialization or file writing fails.
    """
    destination = Path(path)

    try:
        yaml_content = training_spec_to_yaml(spec)
        destination.write_text(yaml_content, encoding="utf-8")
    except OSError as exc:
        raise TrainingSpecYAMLError(
            f"Failed to write TrainingSpec to '{destination}': {exc}"
        ) from exc


def load_training_spec_yaml(path: str | Path) -> TrainingSpec:
    """
    Load and validate a TrainingSpec from a YAML file.

    Args:
        path: YAML file containing a TrainingSpec.

    Returns:
        A validated TrainingSpec instance.

    Raises:
        TrainingSpecYAMLError: If the file cannot be read or contains
            invalid TrainingSpec YAML.
    """
    source = Path(path)

    try:
        yaml_content = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrainingSpecYAMLError(
            f"Failed to read TrainingSpec from '{source}': {exc}"
        ) from exc

    return training_spec_from_yaml(yaml_content)