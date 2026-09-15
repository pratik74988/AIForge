import pytest

from aiforge_template_registry import (
    TEMPLATE_REGISTRY,
    get_template_class,
)


def test_sklearn_classification_template_is_registered():
    assert (
        "random_forest"
        in TEMPLATE_REGISTRY["sklearn"]["classification"]
    )


def test_sklearn_regression_template_is_registered():
    assert (
        "random_forest"
        in TEMPLATE_REGISTRY["sklearn"]["regression"]
    )


def test_unknown_template_raises_value_error():
    with pytest.raises(ValueError, match="Unknown template"):
        get_template_class(
            "unknown_framework",
            "classification",
            "random_forest",
        )


def test_unknown_algorithm_raises_value_error():
    with pytest.raises(ValueError, match="Unknown template"):
        get_template_class(
            "sklearn",
            "classification",
            "does_not_exist",
        )