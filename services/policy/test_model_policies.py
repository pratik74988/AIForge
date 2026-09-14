from policy.model_policies import SUPPORTED_MODEL_POLICIES


def test_sklearn_classification_models_are_supported():
    models = SUPPORTED_MODEL_POLICIES["sklearn"]["classification"]

    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "svm" in models


def test_sklearn_regression_models_are_supported():
    models = SUPPORTED_MODEL_POLICIES["sklearn"]["regression"]

    assert "linear_regression" in models
    assert "ridge" in models
    assert "lasso" in models
    assert "random_forest" in models


def test_unsupported_framework_is_not_present():
    assert "pytorch" not in SUPPORTED_MODEL_POLICIES