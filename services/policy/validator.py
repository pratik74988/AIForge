from typing import Any

from planner.models.training_spec import TrainingSpec

from policy.models.compliance_result import (
    ComplianceResult,
    ComplianceViolation,
)
from policy.model_policies import SUPPORTED_MODEL_POLICIES
from policy.policies import (
    CLASSIFICATION_METRICS,
    HUMAN_APPROVAL_CPU_THRESHOLD,
    HUMAN_APPROVAL_MEMORY_GB_THRESHOLD,
    MAX_CPU,
    MAX_MEMORY_GB,
    REGRESSION_METRICS,
    SUPPORTED_PREPROCESSING_OPERATIONS,
    SUPPORTED_SPLIT_STRATEGIES,
    SUPPORTED_TEMPLATES,
)
from policy.approval_policy import requires_human_approval

class ComplianceValidator:
    """Deterministically validates a TrainingSpec.

    The validator reports violations but never modifies the TrainingSpec.
    """

    def validate(self, spec: TrainingSpec) -> ComplianceResult:
        errors: list[ComplianceViolation] = []
        warnings: list[ComplianceViolation] = []

        self._check_model(spec, errors)
        self._check_preprocessing(spec, errors)
        self._check_metrics(spec, errors)
        self._check_split(spec, errors)
        self._check_resources(spec, errors, warnings)

        approved = len(errors) == 0
        human_approval_required = (
            approved and requires_human_approval(warnings)
        )

        return ComplianceResult(
            approved=approved,
            requires_human_approval=human_approval_required,
            errors=errors,
            warnings=warnings,
        )

    def _check_model(
        self,
        spec: TrainingSpec,
        errors: list[ComplianceViolation],
    ) -> None:
        template = spec.model.template
        algorithm = spec.model.algorithm
        task = getattr(spec.task.type, "value", spec.task.type)

        if template not in SUPPORTED_TEMPLATES:
            errors.append(
                ComplianceViolation(
                    code="UNSUPPORTED_TEMPLATE",
                    message=f"Template '{template}' is not supported.",
                )
            )
            return

        template_policies = SUPPORTED_MODEL_POLICIES.get(template)

        if template_policies is None:
            errors.append(
                ComplianceViolation(
                    code="MISSING_TEMPLATE_POLICY",
                    message=(
                        f"No model policy is defined for "
                        f"template '{template}'."
                    ),
                )
            )
            return

        if task not in template_policies:
            errors.append(
                ComplianceViolation(
                    code="UNSUPPORTED_TASK",
                    message=(
                        f"Task type '{task}' is not supported "
                        f"by template '{template}'."
                    ),
                )
            )
            return

        supported_algorithms = template_policies[task]

        if algorithm not in supported_algorithms:
            errors.append(
                ComplianceViolation(
                    code="UNSUPPORTED_ALGORITHM",
                    message=(
                        f"Algorithm '{algorithm}' is not supported "
                        f"for task '{task}' with template '{template}'."
                    ),
                )
            )

    def _check_preprocessing(
        self,
        spec: TrainingSpec,
        errors: list[ComplianceViolation],
    ) -> None:
        for step in spec.preprocessing.steps:
            operation = step.operation
            parameters = step.parameters

            if operation not in SUPPORTED_PREPROCESSING_OPERATIONS:
                errors.append(
                    ComplianceViolation(
                        code="UNSUPPORTED_PREPROCESSING",
                        message=(
                            f"Preprocessing operation '{operation}' "
                            "is not supported."
                        ),
                    )
                )
                continue

            if operation == "impute":
                self._check_imputation(parameters, errors)

            elif operation == "encode":
                self._check_encoding(parameters, errors)

            elif operation == "scale":
                self._check_scaling(parameters, errors)

    @staticmethod
    def _check_imputation(
        parameters: dict,
        errors: list[ComplianceViolation],
    ) -> None:
        supported_strategies = {
            "mean",
            "median",
            "most_frequent",
            "constant",
        }

        strategy = parameters.get("strategy")

        if strategy not in supported_strategies:
            errors.append(
                ComplianceViolation(
                    code="INVALID_IMPUTATION_STRATEGY",
                    message=(
                        f"Imputation strategy '{strategy}' is not supported."
                    ),
                )
            )

    @staticmethod
    def _check_encoding(
        parameters: dict,
        errors: list[ComplianceViolation],
    ) -> None:
        supported_methods = {
            "one_hot",
            "ordinal",
        }

        method = parameters.get("method")

        if method not in supported_methods:
            errors.append(
                ComplianceViolation(
                    code="INVALID_ENCODING_METHOD",
                    message=(
                        f"Encoding method '{method}' is not supported."
                    ),
                )
            )

    @staticmethod
    def _check_scaling(
        parameters: dict,
        errors: list[ComplianceViolation],
    ) -> None:
        supported_methods = {
            "standard",
            "minmax",
            "robust",
        }

        method = parameters.get("method")

        if method not in supported_methods:
            errors.append(
                ComplianceViolation(
                    code="INVALID_SCALING_METHOD",
                    message=(
                        f"Scaling method '{method}' is not supported."
                    ),
                )
            )

    def _check_metrics(
        self,
        spec: TrainingSpec,
        errors: list[ComplianceViolation],
    ) -> None:
        task = getattr(spec.task.type, "value", spec.task.type)

        if task == "classification":
            supported_metrics = CLASSIFICATION_METRICS

        elif task == "regression":
            supported_metrics = REGRESSION_METRICS

        else:
            errors.append(
                ComplianceViolation(
                    code="UNSUPPORTED_TASK",
                    message=f"Task type '{task}' is not supported.",
                )
            )
            return

        metrics = self._get_metrics(spec)
        primary_metric = spec.evaluation.primary_metric

        for metric in metrics:
            if metric not in supported_metrics:
                errors.append(
                    ComplianceViolation(
                        code="INVALID_METRIC",
                        message=(
                            f"Metric '{metric}' is not valid for "
                            f"task '{task}'."
                        ),
                    )
                )

        if primary_metric not in metrics:
            errors.append(
                ComplianceViolation(
                    code="PRIMARY_METRIC_NOT_SELECTED",
                    message=(
                        f"Primary metric '{primary_metric}' must be "
                        "included in evaluation.metrics."
                    ),
                )
            )

        if primary_metric not in supported_metrics:
            errors.append(
                ComplianceViolation(
                    code="INVALID_PRIMARY_METRIC",
                    message=(
                        f"Primary metric '{primary_metric}' is not valid "
                        f"for task '{task}'."
                    ),
                )
            )

    def _check_split(
        self,
        spec: TrainingSpec,
        errors: list[ComplianceViolation],
    ) -> None:
        split = spec.split

        strategy = getattr(split.strategy, "value", split.strategy)
        test_size = getattr(split, "test_size", None)

        if strategy not in SUPPORTED_SPLIT_STRATEGIES:
            errors.append(
                ComplianceViolation(
                    code="UNSUPPORTED_SPLIT_STRATEGY",
                    message=(
                        f"Split strategy '{strategy}' is not supported."
                    ),
                )
            )
            return

        if test_size is None:
            errors.append(
                ComplianceViolation(
                    code="MISSING_TEST_SIZE",
                    message=(
                        f"Split strategy '{strategy}' requires test_size."
                    ),
                )
            )
            return

        if not 0 < test_size < 1:
            errors.append(
                ComplianceViolation(
                    code="INVALID_TEST_SIZE",
                    message="test_size must be between 0 and 1.",
                )
            )

    def _check_resources(
        self,
        spec: TrainingSpec,
        errors: list[ComplianceViolation],
        warnings: list[ComplianceViolation],
    ) -> None:
        resources = spec.resources

        cpu = self._get_resource_value(resources, "cpu")
        memory = self._get_memory_gb(resources.memory)

        if cpu is not None:
            if cpu > MAX_CPU:
                errors.append(
                    ComplianceViolation(
                        code="CPU_LIMIT_EXCEEDED",
                        message=(
                            f"Requested CPU ({cpu}) exceeds limit ({MAX_CPU})."
                        ),
                    )
                )

            elif cpu > HUMAN_APPROVAL_CPU_THRESHOLD:
                warnings.append(
                    ComplianceViolation(
                        code="CPU_HUMAN_APPROVAL_REQUIRED",
                        message=(
                            f"Requested CPU ({cpu}) exceeds the "
                            f"human-approval threshold "
                            f"({HUMAN_APPROVAL_CPU_THRESHOLD})."
                        ),
                    )
                )

        if memory is not None:
            if memory > MAX_MEMORY_GB:
                errors.append(
                    ComplianceViolation(
                        code="MEMORY_LIMIT_EXCEEDED",
                        message=(
                            f"Requested memory ({memory} GB) exceeds "
                            f"limit ({MAX_MEMORY_GB} GB)."
                        ),
                    )
                )

            elif memory > HUMAN_APPROVAL_MEMORY_GB_THRESHOLD:
                warnings.append(
                    ComplianceViolation(
                        code="MEMORY_HUMAN_APPROVAL_REQUIRED",
                        message=(
                            f"Requested memory ({memory} GB) exceeds the "
                            f"human-approval threshold "
                            f"({HUMAN_APPROVAL_MEMORY_GB_THRESHOLD} GB)."
                        ),
                    )
                )

    @staticmethod
    def _get_metrics(spec: TrainingSpec) -> list[str]:
        evaluation: Any = spec.evaluation
        metrics = getattr(evaluation, "metrics", [])

        if isinstance(metrics, str):
            return [metrics]

        return list(metrics)

    @staticmethod
    def _get_memory_gb(memory: str | None) -> float | None:
        if memory is None:
            return None

        value = memory.strip().lower()

        try:
            if value.endswith("gib"):
                return float(value[:-3].strip())

            if value.endswith("gb"):
                return float(value[:-2].strip())

            if value.endswith("g"):
                return float(value[:-1].strip())

            return float(value)

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_resource_value(
        resources: Any,
        name: str,
    ) -> float | None:
        value = getattr(resources, name, None)

        if value is None:
            return None

        try:
            return float(value)

        except (TypeError, ValueError):
            return None