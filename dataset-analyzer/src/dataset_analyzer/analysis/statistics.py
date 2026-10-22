"""Statistical analysis for numerical dataset columns."""

from __future__ import annotations

import pandas as pd

from ..profile.models import ColumnStatistics


class StatisticsAnalyzer:
    """Calculate descriptive statistics for numerical columns."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize the analyzer with a DataFrame."""
        self.dataframe = dataframe

    def analyze(self) -> dict[str, ColumnStatistics]:
        """Return descriptive statistics for numerical columns.

        Non-numerical columns are excluded from the result. Missing values
        are ignored by pandas when calculating the statistics.

        Returns
        -------
        dict[str, ColumnStatistics]
            Mapping of numerical column names to their descriptive
            statistics.

        Notes
        -----
        An entirely null numerical column produces ``None`` for all
        statistics. Columns containing infinite values are handled by
        excluding infinities from the calculations.
        """
        results: dict[str, ColumnStatistics] = {}

        numerical_columns = self.dataframe.select_dtypes(
            include="number"
        ).columns

        for column in numerical_columns:
            series = pd.to_numeric(
                self.dataframe[column],
                errors="coerce",
            ).replace([float("inf"), float("-inf")], pd.NA).dropna()

            if series.empty:
                results[str(column)] = ColumnStatistics()
                continue

            results[str(column)] = ColumnStatistics(
                mean=self._safe_float(series.mean()),
                median=self._safe_float(series.median()),
                std=self._safe_float(series.std()),
                min=self._safe_float(series.min()),
                max=self._safe_float(series.max()),
            )

        return results

    @staticmethod
    def _safe_float(value: object) -> float | None:
        """Convert a statistical value to a finite Python float."""
        if pd.isna(value):
            return None

        result = float(value)

        if result != float("inf") and result != float("-inf"):
            return result

        return None