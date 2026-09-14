"""
Prompt construction for the AIForge Planner.

This module converts PlanningContext into a structured prompt for the
LLM. It contains no provider-specific SDK code and performs no planning
itself.

The prompt builder is intentionally deterministic so that the same
PlanningContext produces the same prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .context import PlanningContext
from .models.planner_result import PlannerResult

@dataclass(frozen=True)
class PlannerPrompt:
    """
    Prompt supplied to the LLM.

    Args:
        system:
            Instructions defining the LLM's role and output contract.

        user:
            Planning-specific input containing the user's requirement,
            DatasetProfile, and retrieved Knowledge Base context.
    """

    system: str
    user: str


class PlannerPromptBuilder:
    """Build prompts for the AIForge Planner LLM."""

    SYSTEM_PROMPT = """\
You are the AIForge ML Planning Engine.

Your job is to design an ML training plan from:
1. The user's machine-learning requirement.
2. Facts produced by the Dataset Analyzer.
3. Relevant Knowledge Base guidance.

Follow these rules:

- Treat Dataset Analyzer information as factual observations.
- Do not invent dataset facts that are not provided.
- Make planning decisions based on the supplied facts and user intent.
- Use Knowledge Base information as guidance, not as a substitute for
  dataset-specific facts.
- Recommend preprocessing using framework-agnostic semantic operations.
- Select a supported model template and algorithm appropriate for the task.
- Choose evaluation metrics according to the user's stated priorities.
- Explain important decisions through recommendations.
- Produce a proposed TrainingSpec that contains only decisions required
  by the TrainingSpec schema.
- Do not include framework-specific implementation classes in preprocessing.
- Do not include execution infrastructure such as Docker, Kubernetes,
  Argo, or cloud-specific commands in the TrainingSpec.
- Do not add fields that are not part of the TrainingSpec contract.

IMPORTANT OUTPUT RULES:

Return ONLY valid JSON.

The root JSON object MUST contain exactly these fields:

{
  "recommendations": [...],
  "proposed_spec": {...},
  "requires_approval": true
}

Each recommendation MUST contain exactly these fields:

{
  "type": "preprocessing | model | evaluation | split | features | resources",
  "title": "short title",
  "reasoning": "why this decision is appropriate",
  "decision": "the actual decision",
  "requires_approval": true
}

Do NOT use:
- category
- description
- rationale

for recommendations.

The proposed_spec MUST use the exact AIForge TrainingSpec field names:

{
  "spec_version": "1.0",
  "task": {
    "type": "classification | regression | clustering",
    "target": "target column"
  },
  "features": {
    "include": null,
    "exclude": []
  },
  "preprocessing": {
    "steps": []
  },
  "split": {
    "strategy": "random | stratified | group | time",
    "test_size": 0.2,
    "validation_size": 0.1
  },
  "model": {
    "template": "template name",
    "algorithm": "algorithm name",
    "hyperparameters": {}
  },
  "evaluation": {
    "primary_metric": "metric",
    "metrics": []
  },
  "resources": {
    "cpu": null,
    "memory": null,
    "gpu": false,
    "max_runtime_minutes": null
  },
  "reproducibility": {
    "random_seed": 42
  }
}

Do NOT use "training_spec" as the root field name.
Use "proposed_spec".

Do NOT invent alternative field names.

For preprocessing steps, use exactly:

{
  "name": "step name",
  "operation": "operation",
  "columns": "numeric | categorical | all | [column names]",
  "parameters": {}
}

Return no Markdown fences, no explanation outside the JSON object,
and no additional root fields.
"""
    def build(self, context: PlanningContext) -> PlannerPrompt:
        """
        Build a deterministic prompt from PlanningContext.

        Args:
            context: Complete planning context.

        Returns:
            A PlannerPrompt containing system and user messages.
        """
        dataset_profile = self._serialize_dataset_profile(
            context.dataset_profile
        )

        knowledge_context = self._serialize_knowledge_context(context)

        schema_json = json.dumps(
            PlannerResult.model_json_schema(),
            indent=2,
            ensure_ascii=False,
        )

        user_prompt = f"""\
USER REQUIREMENT
----------------
{context.user_requirement}

DATASET PROFILE
---------------
{dataset_profile}

KNOWLEDGE BASE CONTEXT
----------------------
{knowledge_context}

Using the information above, produce a planning result containing:

1. Recommendations explaining important planning decisions.
2. A proposed TrainingSpec satisfying the AIForge TrainingSpec contract.
3. Whether the proposed plan requires user approval.
"""

        return PlannerPrompt(
            system=self.SYSTEM_PROMPT,
            user=user_prompt,
        )

    @staticmethod
    def _serialize_dataset_profile(dataset_profile: object) -> str:
        """
        Serialize DatasetProfile into readable JSON.

        DatasetProfile is expected to be a Pydantic model, but accepting
        object here keeps this layer independent from the Analyzer's
        package location.
        """
        if hasattr(dataset_profile, "model_dump"):
            data = dataset_profile.model_dump(mode="json")
        elif isinstance(dataset_profile, dict):
            data = dataset_profile
        else:
            raise TypeError(
                "dataset_profile must be a Pydantic model or dictionary."
            )

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _serialize_knowledge_context(
        context: PlanningContext,
    ) -> str:
        """Serialize retrieved Knowledge Base context."""
        if not context.knowledge_context.items:
            return "No additional Knowledge Base context was retrieved."

        lines = []

        for index, item in enumerate(
            context.knowledge_context.items,
            start=1,
        ):
            lines.append(f"[{index}] {item}")

        return "\n".join(lines)