from __future__ import annotations

from typing import Any

from dataset_analyzer.profile.models import DatasetProfile
from planner.bootstrap import create_planner
from planner.context import PlanningContext
from planner.models.planner_result import PlannerResult
from planner.models.training_spec import TrainingSpec
from policy.compliance import check_compliance
from compiler.compiler import PipelineCompiler


def _resolve_feature_types(
    spec: TrainingSpec,
    profile: DatasetProfile,
) -> TrainingSpec:
    """Fill missing numerical/categorical feature lists from the profile."""

    feature_spec = spec.features

    excluded = set(feature_spec.exclude)
    target = spec.task.target

    if feature_spec.include:
        candidates = set(feature_spec.include)
    else:
        candidates = {
            column.name
            for column in profile.columns
            if column.name != target
        }

    candidates -= excluded
    candidates.discard(target)

    numerical = list(feature_spec.numerical)
    categorical = list(feature_spec.categorical)

    already_assigned = set(numerical) | set(categorical)

    for column in profile.columns:
        if column.name not in candidates:
            continue
        if column.name in already_assigned:
            continue

        semantic_type = str(column.semantic_type).lower().split(".")[-1]

        if semantic_type in {
            "numeric",
            "numerical",
            "continuous",
            "integer",
            "float",
        }:
            numerical.append(column.name)
        elif semantic_type in {
            "categorical",
            "category",
            "boolean",
            "ordinal",
        }:
            categorical.append(column.name)

    resolved_features = feature_spec.model_copy(
        update={
            "numerical": numerical,
            "categorical": categorical,
        }
    )

    return spec.model_copy(update={"features": resolved_features})


def analyze_dataframe(
    dataframe: Any,
    target: str | None = None,
) -> DatasetProfile:
    from dataset_analyzer.analyzer import DatasetAnalyzer

    return DatasetAnalyzer(dataframe, target=target).analyze()


def plan_pipeline(
    profile: DatasetProfile,
    user_requirement: str,
) -> PlannerResult:
    context = PlanningContext(
        user_requirement=user_requirement,
        dataset_profile=profile,
    )

    planner = create_planner()
    result = planner.plan(context)

    resolved_spec = _resolve_feature_types(
        result.proposed_spec,
        profile,
    )

    return result.model_copy(
        update={"proposed_spec": resolved_spec}
    )


def validate_spec(spec: TrainingSpec):
    return check_compliance(spec)


def compile_spec(
    spec: TrainingSpec,
    user_approved: bool,
) -> dict:
    if not user_approved:
        raise PermissionError(
            "The proposed plan must be explicitly approved before compilation."
        )

    # Re-run compliance on the backend. Do not trust a result supplied by the UI.
    compliance_result = check_compliance(spec)

    if not compliance_result.approved:
        raise ValueError(
            "Compilation blocked: compliance validation reported errors."
        )

    if compliance_result.requires_human_approval:
        raise PermissionError(
            "Compilation blocked: the compliance policy requires additional "
            "human approval."
        )

    compiled = PipelineCompiler().compile(spec, compliance_result)

    # Return only JSON-safe metadata, not the sklearn pipeline or Python class.
    return {
        "status": "compiled",
        "template": compiled.template,
        "algorithm": compiled.algorithm,
        "preprocessing_steps": compiled.preprocessing_steps,
        "hyperparameters": compiled.hyperparameters,
        "evaluation_metrics": compiled.evaluation_metrics,
        "primary_metric": compiled.primary_metric,
    }