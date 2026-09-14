from __future__ import annotations

import pandas as pd

from .context import PlanningContext
from .llm_service import LLMService
from .planner import Planner
from .response_parser import PlannerResponseParser
from dataset_analyzer import DatasetAnalyzer
import pytest

@pytest.fixture
def analyzer():
    class _AnalyzerFactory:
        @staticmethod
        def analyze(dataframe, target=None):
            return DatasetAnalyzer(dataframe, target=target).analyze()
    return _AnalyzerFactory()

class FakeLLMProvider:
    def generate(self, prompt) -> str:
        return """
        {
          "recommendations": [
            {
              "type": "split",
              "title": "Use stratified splitting",
              "reasoning": "The task is classification, so class proportions should be preserved.",
              "decision": "Use stratified train/validation/test splitting.",
              "requires_approval": true
            },
            {
              "type": "evaluation",
              "title": "Prioritize recall",
              "reasoning": "The user prioritizes identifying positive cases.",
              "decision": "Use recall as the primary evaluation metric.",
              "requires_approval": true
            }
          ],
          "proposed_spec": {
            "spec_version": "1.0",
            "task": {
              "type": "classification",
              "target": "churn"
            },
            "features": {
              "include": null,
              "exclude": []
            },
            "preprocessing": {
              "steps": [
                {
                  "name": "numeric_imputation",
                  "operation": "impute",
                  "columns": "numeric",
                  "parameters": {
                    "strategy": "median"
                  }
                },
                {
                  "name": "categorical_imputation",
                  "operation": "impute",
                  "columns": "categorical",
                  "parameters": {
                    "strategy": "most_frequent"
                  }
                },
                {
                  "name": "categorical_encoding",
                  "operation": "encode",
                  "columns": "categorical",
                  "parameters": {
                    "method": "one_hot"
                  }
                },
                {
                  "name": "numerical_scaling",
                  "operation": "scale",
                  "columns": "numeric",
                  "parameters": {
                    "method": "standard"
                  }
                }
              ]
            },
            "split": {
              "strategy": "stratified",
              "test_size": 0.2,
              "validation_size": 0.1
            },
            "model": {
              "template": "sklearn",
              "algorithm": "logistic_regression",
              "hyperparameters": {
                "C": 1.0,
                "penalty": "l2",
                "class_weight": "balanced"
              }
            },
            "evaluation": {
              "primary_metric": "recall",
              "metrics": [
                "recall",
                "precision",
                "f1",
                "roc_auc"
              ]
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
          },
          "requires_approval": true
        }
        """


def test_dataset_analyzer_output_can_feed_planner(
    analyzer,
) -> None:
    dataset = pd.DataFrame(
        {
            "age": [21, 35, 42, 29, 51, 38, 27, 46, 33, 55],
            "income": [
                30000,
                50000,
                70000,
                42000,
                90000,
                65000,
                35000,
                80000,
                48000,
                95000,
            ],
            "city": [
                "Pune",
                "Mumbai",
                "Pune",
                "Nashik",
                "Mumbai",
                "Pune",
                "Nashik",
                "Mumbai",
                "Pune",
                "Mumbai",
            ],
            "churn": [
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

    dataset_profile = analyzer.analyze(
        dataset,
        target="churn",
    )

    context = PlanningContext(
        user_requirement=(
            "Build a classification model where identifying customers "
            "likely to churn is more important than avoiding false alarms."
        ),
        dataset_profile=dataset_profile,
    )

    planner = Planner(
        llm_service=LLMService(
            provider=FakeLLMProvider(),
            response_parser=PlannerResponseParser(),
        )
    )

    result = planner.plan(context)

    assert result.requires_approval is True

    assert result.proposed_spec.task.type == "classification"
    assert result.proposed_spec.task.target == "churn"

    assert result.proposed_spec.model.template == "sklearn"
    assert result.proposed_spec.model.algorithm == "logistic_regression"

    assert result.proposed_spec.split.strategy == "stratified"

    assert result.proposed_spec.evaluation.primary_metric == "recall"

    assert len(result.recommendations) == 2