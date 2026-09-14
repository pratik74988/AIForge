from copy import deepcopy

from planner.models.training_spec import (
    TrainingSpec,
    PreprocessingStep,
)

from policy.validator import ComplianceValidator
from policy.models.compliance_result import ComplianceViolation


def make_valid_spec() -> TrainingSpec:
    return TrainingSpec.model_validate(
        {
            "spec_version": "1.0",
            "task": {
                "type": "classification",
                "target": "target",
            },
            "features": {},
            "preprocessing": {
                "steps": [
                    {
                        "name": "numeric_imputation",
                        "operation": "impute",
                        "columns": "numeric",
                        "parameters": {
                            "strategy": "median",
                        },
                    },
                    {
                        "name": "categorical_encoding",
                        "operation": "encode",
                        "columns": "categorical",
                        "parameters": {
                            "method": "one_hot",
                        },
                    },
                    {
                        "name": "numerical_scaling",
                        "operation": "scale",
                        "columns": "numeric",
                        "parameters": {
                            "method": "standard",
                        },
                    },
                ],
            },
            "split": {
                "strategy": "random",
                "test_size": 0.2,
            },
            "model": {
                "template": "sklearn",
                "algorithm": "logistic_regression",
                "hyperparameters": {},
            },
            "evaluation": {
                "metrics": ["accuracy", "f1_macro"],
                "primary_metric": "accuracy",
            },
            "resources": {},
            "reproducibility": {},
        }
    )


def test_valid_training_spec_is_approved():
    spec = make_valid_spec()

    result = ComplianceValidator().validate(spec)

    assert result.approved is True
    assert result.errors == []


def test_unsupported_template_is_rejected():
    spec = make_valid_spec()
    spec.model.template = "pytorch"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "UNSUPPORTED_TEMPLATE"
        for error in result.errors
    )


def test_unsupported_algorithm_is_rejected():
    spec = make_valid_spec()
    spec.model.algorithm = "xgboost"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "UNSUPPORTED_ALGORITHM"
        for error in result.errors
    )


def test_invalid_task_algorithm_combination_is_rejected():
    spec = make_valid_spec()
    spec.model.algorithm = "linear_regression"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "UNSUPPORTED_ALGORITHM"
        for error in result.errors
    )


def test_unsupported_preprocessing_is_rejected():
    spec = make_valid_spec()

    spec.preprocessing.steps.append(
        PreprocessingStep(
            name="unknown_step",
            operation="magic_transform",
            columns="numeric",
            parameters={},
        )
    )

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "UNSUPPORTED_PREPROCESSING"
        for error in result.errors
    )


def test_invalid_metric_for_classification_is_rejected():
    spec = make_valid_spec()
    spec.evaluation.metrics = ["r2"]

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "INVALID_METRIC"
        for error in result.errors
    )


def test_validator_does_not_modify_training_spec():
    spec = make_valid_spec()
    before = deepcopy(spec.model_dump())

    ComplianceValidator().validate(spec)

    after = spec.model_dump()

    assert after == before

def test_primary_metric_must_be_in_metrics():
    spec = make_valid_spec()
    spec.evaluation.primary_metric = "roc_auc"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "PRIMARY_METRIC_NOT_SELECTED"
        for error in result.errors
    )


def test_primary_metric_must_be_valid_for_task():
    spec = make_valid_spec()
    spec.evaluation.metrics = ["accuracy", "r2"]
    spec.evaluation.primary_metric = "r2"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "INVALID_METRIC"
        for error in result.errors
    )
    assert any(
        error.code == "INVALID_PRIMARY_METRIC"
        for error in result.errors
    )


def test_valid_primary_metric_is_accepted():
    spec = make_valid_spec()
    spec.evaluation.metrics = ["accuracy", "f1_macro"]
    spec.evaluation.primary_metric = "f1_macro"

    result = ComplianceValidator().validate(spec)

    assert result.approved is True
    assert result.errors == []


def test_unsupported_task_is_rejected():
    spec = make_valid_spec()
    spec.task.type = "unsupported_task"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "UNSUPPORTED_TASK"
        for error in result.errors
    )

def test_random_split_with_test_size_is_valid():
    spec = make_valid_spec()

    result = ComplianceValidator().validate(spec)

    assert result.approved is True
    assert result.errors == []


def test_missing_test_size_is_rejected():
    spec = make_valid_spec()
    spec.split.test_size = None

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "MISSING_TEST_SIZE"
        for error in result.errors
    )


def test_invalid_test_size_is_rejected():
    spec = make_valid_spec()
    spec.split.test_size = 1.0

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "INVALID_TEST_SIZE"
        for error in result.errors
    )




def test_unsupported_split_strategy_is_rejected():
    spec = make_valid_spec()
    spec.split.strategy = "time_series"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "UNSUPPORTED_SPLIT_STRATEGY"
        for error in result.errors
    )

def test_valid_imputation_strategy_is_accepted():
    spec = make_valid_spec()

    result = ComplianceValidator().validate(spec)

    assert result.approved is True


def test_invalid_imputation_strategy_is_rejected():
    spec = make_valid_spec()

    spec.preprocessing.steps[0] = PreprocessingStep(
        name="numeric_imputation",
        operation="impute",
        columns="numeric",
        parameters={
            "strategy": "banana",
        },
    )

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "INVALID_IMPUTATION_STRATEGY"
        for error in result.errors
    )


def test_valid_encoding_method_is_accepted():
    spec = make_valid_spec()

    result = ComplianceValidator().validate(spec)

    assert result.approved is True


def test_invalid_encoding_method_is_rejected():
    spec = make_valid_spec()

    spec.preprocessing.steps[1] = PreprocessingStep(
        name="categorical_encoding",
        operation="encode",
        columns="categorical",
        parameters={
            "method": "teleport",
        },
    )

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "INVALID_ENCODING_METHOD"
        for error in result.errors
    )


def test_valid_scaling_method_is_accepted():
    spec = make_valid_spec()

    result = ComplianceValidator().validate(spec)

    assert result.approved is True


def test_invalid_scaling_method_is_rejected():
    spec = make_valid_spec()

    spec.preprocessing.steps[2] = PreprocessingStep(
        name="numerical_scaling",
        operation="scale",
        columns="numeric",
        parameters={
            "method": "banana",
        },
    )

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "INVALID_SCALING_METHOD"
        for error in result.errors
    )

def test_model_policy_is_used_for_algorithm_validation():
    spec = make_valid_spec()

    spec.model.algorithm = "linear_regression"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False

    error = next(
        error
        for error in result.errors
        if error.code == "UNSUPPORTED_ALGORITHM"
    )

    assert "linear_regression" in error.message
    assert "classification" in error.message
    assert "sklearn" in error.message


def test_cpu_limit_is_rejected():
    spec = make_valid_spec()
    spec.resources.cpu = 17

    result = validate(spec)

    assert result.approved is False
    assert any(
        error.code == "CPU_LIMIT_EXCEEDED"
        for error in result.errors
    )


def test_cpu_limit_is_rejected():
    spec = make_valid_spec()
    spec.resources.cpu = 17

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "CPU_LIMIT_EXCEEDED"
        for error in result.errors
    )


def test_memory_limit_is_rejected():
    spec = make_valid_spec()
    spec.resources.memory = "65GB"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert any(
        error.code == "MEMORY_LIMIT_EXCEEDED"
        for error in result.errors
    )


def test_cpu_at_limit_is_allowed():
    spec = make_valid_spec()
    spec.resources.cpu = 16

    result = ComplianceValidator().validate(spec)

    assert result.approved is True


def test_memory_at_limit_is_allowed():
    spec = make_valid_spec()
    spec.resources.memory = "64GB"

    result = ComplianceValidator().validate(spec)

    assert result.approved is True


def test_multiple_resource_violations_are_reported():
    spec = make_valid_spec()
    spec.resources.cpu = 17
    spec.resources.memory = "65GB"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False

    error_codes = {error.code for error in result.errors}

    assert "CPU_LIMIT_EXCEEDED" in error_codes
    assert "MEMORY_LIMIT_EXCEEDED" in error_codes


def test_valid_spec_does_not_require_human_approval():
    spec = make_valid_spec()

    result = ComplianceValidator().validate(spec)

    assert result.approved is True
    assert result.requires_human_approval is False
    assert result.errors == []
    assert result.warnings == []


def test_rejected_spec_does_not_require_human_approval():
    spec = make_valid_spec()
    spec.model.algorithm = "unsupported_algorithm"

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert result.requires_human_approval is False
    assert len(result.errors) > 0


def test_warning_requires_human_approval(monkeypatch):
    spec = make_valid_spec()

    def fake_approval_policy(warnings):
        return True

    monkeypatch.setattr(
        "policy.validator.requires_human_approval",
        fake_approval_policy,
    )

    validator = ComplianceValidator()

    # Inject a warning-producing condition without changing
    # the TrainingSpec contract.
    monkeypatch.setattr(
        validator,
        "_check_resources",
        lambda spec, errors, warnings: warnings.append(
            ComplianceViolation(
                code="RESOURCE_WARNING",
                message="Human review required.",
            )
        ),
    )

    result = validator.validate(spec)

    assert result.approved is True
    assert result.requires_human_approval is True
    assert result.errors == []
    assert len(result.warnings) == 1

def test_high_cpu_requires_human_approval():
    spec = make_valid_spec()
    spec.resources.cpu = 12

    result = ComplianceValidator().validate(spec)

    assert result.approved is True
    assert result.requires_human_approval is True
    assert any(
        warning.code == "CPU_HUMAN_APPROVAL_REQUIRED"
        for warning in result.warnings
    )


def test_high_memory_requires_human_approval():
    spec = make_valid_spec()
    spec.resources.memory = "48GB"

    result = ComplianceValidator().validate(spec)

    assert result.approved is True
    assert result.requires_human_approval is True
    assert any(
        warning.code == "MEMORY_HUMAN_APPROVAL_REQUIRED"
        for warning in result.warnings
    )


def test_resource_above_hard_limit_is_rejected_not_approval_required():
    spec = make_valid_spec()
    spec.resources.cpu = 20

    result = ComplianceValidator().validate(spec)

    assert result.approved is False
    assert result.requires_human_approval is False
    assert any(
        error.code == "CPU_LIMIT_EXCEEDED"
        for error in result.errors
    )


def test_resources_below_approval_threshold_need_no_approval():
    spec = make_valid_spec()
    spec.resources.cpu = 4
    spec.resources.memory = "16GB"

    result = ComplianceValidator().validate(spec)

    assert result.approved is True
    assert result.requires_human_approval is False
    assert result.warnings == []