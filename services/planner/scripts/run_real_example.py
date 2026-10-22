"""Manual one-shot: real Dataset Analyzer -> real Gemini -> PlannerResult."""

import pandas as pd

from dataset_analyzer import DatasetAnalyzer
from planner.bootstrap import create_planner
from planner.context import PlanningContext

df = pd.DataFrame({
    "age": [21, 35, 42, 29, 51, 38, 27, 46, 33, 55],
    "income": [30000, 50000, 70000, 42000, 90000, 65000, 35000, 80000, 48000, 95000],
    "city": ["Pune", "Mumbai", "Pune", "Nashik", "Mumbai", "Pune", "Nashik", "Mumbai", "Pune", "Mumbai"],
    "churn": [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
})

profile = DatasetAnalyzer(df, target="churn").analyze()

context = PlanningContext(
    user_requirement=(
        "Build a classification model where identifying customers "
        "likely to churn is more important than avoiding false alarms."
    ),
    dataset_profile=profile,
)

planner = create_planner()  # reads AIFORGE_LLM_MODEL / AIFORGE_LLM_API_KEY from env
result = planner.plan(context)

print(result.model_dump_json(indent=2))