"""Target-column analysis for supervised learning datasets."""

from __future__ import annotations

import pandas as pd

from ..profile.models import TargetProfile


class TargetAnalyzer:
    """Analyze a user-specified target column."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
    ) -> None:
        """Initialize the analyzer with a DataFrame and target column."""
        self.dataframe = dataframe
        self.target = target

    def analyze(self) -> TargetProfile:
        """Analyze the target column and determine its learning task.

        The target must be explicitly provided by the caller. This analyzer
        determines its semantic type, number of unique values, class
        distribution for classification targets, and class imbalance ratio.

        Raises
        ------
        ValueError
            If the target column does not exist or contains no non-null values.
        """
        if self.target not in self.dataframe.columns:
            raise ValueError(
                f"Target column '{self.target}' was not found in the dataset."
            )

        series = self.dataframe[self.target].dropna()

        if series.empty:
            raise ValueError(
                f"Target column '{self.target}' contains no non-null values."
            )

        unique_count = int(series.nunique())

        semantic_type = self._infer_semantic_type(series)
        task = self._infer_task(series, semantic_type)

        class_distribution: dict[str, int] | None = None
        imbalance_ratio: float | None = None

        if task in {"binary_classification", "multiclass_classification"}:
            counts = series.value_counts()

            class_distribution = {
                str(label): int(count)
                for label, count in counts.items()
            }

            imbalance_ratio = self._calculate_imbalance_ratio(counts)

        return TargetProfile(
            name=self.target,
            semantic_type=semantic_type,
            task=task,
            unique_count=unique_count,
            class_distribution=class_distribution,
            imbalance_ratio=imbalance_ratio,
        )

    @staticmethod
    def _infer_semantic_type(series: pd.Series) -> str:
        """Infer the semantic type of the target column."""
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if pd.api.types.is_timedelta64_dtype(series):
            return "timedelta"

        if pd.api.types.is_numeric_dtype(series):
            return "numerical"

        return "categorical"

    @staticmethod
    def _infer_task(
        series: pd.Series,
        semantic_type: str,
    ) -> str:
        """Determine whether the target represents classification or regression."""
        unique_count = series.nunique()

        if semantic_type in {"categorical", "boolean"}:
            if unique_count == 2:
                return "binary_classification"
            return "multiclass_classification"

        if semantic_type == "numerical":
            if unique_count == 2:
                return "binary_classification"

            return "regression"

        return "unsupported"

    @staticmethod
    def _calculate_imbalance_ratio(
        counts: pd.Series,
    ) -> float | None:
        """Calculate the ratio between the largest and smallest class."""
        if len(counts) < 2:
            return None

        smallest = counts.min()

        if smallest == 0:
            return None

        return round(float(counts.max() / smallest), 6)