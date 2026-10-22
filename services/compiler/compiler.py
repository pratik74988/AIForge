from planner.models.training_spec import TrainingSpec

from aiforge_template_registry import get_template_class

from compiler.models.compiled_pipeline import CompiledPipeline
from compiler.preprocessing import build_training_preprocessor

from policy.models.compliance_result import ComplianceResult


class PipelineCompiler:
    """Compile an approved TrainingSpec into an executable pipeline."""

    def compile(
        self,
        spec: TrainingSpec,
        compliance_result: ComplianceResult,
    ) -> CompiledPipeline:
        """Compile an approved TrainingSpec."""

        if not compliance_result.approved:
            raise ValueError(
                "TrainingSpec cannot be compiled because compliance "
                "approval was not granted."
            )

        if compliance_result.requires_human_approval:
            raise ValueError(
                "TrainingSpec cannot be compiled because human approval "
                "is required."
            )

        template_class = get_template_class(
            framework=spec.model.template,
            task=spec.task.type,
            model=spec.model.algorithm,
        )

        preprocessor = build_training_preprocessor(
            preprocessing_steps=spec.preprocessing.steps,
            numerical_features=spec.features.numerical,
            categorical_features=spec.features.categorical,
        )

        template = template_class(
            preprocessor=preprocessor,
            model_config=dict(spec.model.hyperparameters),
        )

        pipeline = template.build_pipeline()

        return CompiledPipeline(
            template=spec.model.template,
            algorithm=spec.model.algorithm,
            preprocessing_steps=[
                step.name for step in spec.preprocessing.steps
            ],
            hyperparameters=dict(spec.model.hyperparameters),
            evaluation_metrics=list(spec.evaluation.metrics),
            primary_metric=spec.evaluation.primary_metric,
            template_class=template_class,
            pipeline=pipeline,
        )