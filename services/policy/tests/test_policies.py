from policy.policies import (
    CLASSIFICATION_METRICS,
    MAX_CPU,
    MAX_MEMORY_GB,
    REGRESSION_METRICS,
    SUPPORTED_PREPROCESSING_OPERATIONS,
    SUPPORTED_TEMPLATES,
    HUMAN_APPROVAL_CPU_THRESHOLD,
    HUMAN_APPROVAL_MEMORY_GB_THRESHOLD,
)


def test_sklearn_template_is_supported():
    assert "sklearn" in SUPPORTED_TEMPLATES


# def test_classification_algorithms_are_supported():
#     assert "logistic_regression" in SUPPORTED_ALGORITHMS["classification"]
#     assert "random_forest" in SUPPORTED_ALGORITHMS["classification"]
#     assert "svm" in SUPPORTED_ALGORITHMS["classification"]


# def test_regression_algorithms_are_supported():
#     assert "linear_regression" in SUPPORTED_ALGORITHMS["regression"]
#     assert "ridge" in SUPPORTED_ALGORITHMS["regression"]
#     assert "lasso" in SUPPORTED_ALGORITHMS["regression"]
#     assert "random_forest" in SUPPORTED_ALGORITHMS["regression"]


def test_preprocessing_operations_are_supported():
    assert "impute" in SUPPORTED_PREPROCESSING_OPERATIONS
    assert "encode" in SUPPORTED_PREPROCESSING_OPERATIONS
    assert "scale" in SUPPORTED_PREPROCESSING_OPERATIONS


def test_classification_metrics_are_supported():
    assert "accuracy" in CLASSIFICATION_METRICS
    assert "f1_macro" in CLASSIFICATION_METRICS
    assert "roc_auc" in CLASSIFICATION_METRICS


def test_regression_metrics_are_supported():
    assert "mae" in REGRESSION_METRICS
    assert "mse" in REGRESSION_METRICS
    assert "rmse" in REGRESSION_METRICS
    assert "r2" in REGRESSION_METRICS


def test_resource_limits_are_positive():
    assert MAX_CPU > 0
    assert MAX_MEMORY_GB > 0    

def test_human_approval_cpu_threshold():
    assert HUMAN_APPROVAL_CPU_THRESHOLD == 8


def test_human_approval_memory_threshold():
    assert HUMAN_APPROVAL_MEMORY_GB_THRESHOLD == 32