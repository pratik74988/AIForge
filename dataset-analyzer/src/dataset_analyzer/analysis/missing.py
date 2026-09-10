"""Missing-value analysis for tabular datasets."""
from __future__ import annotations

import pandas as pd


class MissingValueAnalyzer:
    """Analyze missing values across DataFrame columns."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize the analyzer with a DataFrame."""
        self.dataframe = dataframe

    def analyze(self) -> dict[str, dict[str, float | int]]:
        """Analyze missing values for each column."""
        row_count = len(self.dataframe)

        results: dict[str, dict[str, float | int]] = {}

        for column in self.dataframe.columns:
            missing_count = int(self.dataframe[column].isna().sum())

            missing_percentage = (
                (missing_count / row_count) * 100
                if row_count > 0
                else 0.0
            )

            results[str(column)] = {
                "missing_count": missing_count,
                "missing_percentage": round(missing_percentage, 4),
            }

        return results

    def high_missing_columns(
        self,
        threshold: float = 50.0,
    ) -> list[str]:
        """Return columns whose missing percentage meets the threshold.

        Parameters
        ----------
        threshold:
            Missing-value percentage at or above which a column is considered
            high-missing. Must be between 0 and 100 inclusive.

        Returns
        -------
        list[str]
            Column names meeting or exceeding the threshold.

        Raises
        ------
        ValueError
            If ``threshold`` is outside the range 0 to 100.
        """
        if not 0 <= threshold <= 100:
            raise ValueError("threshold must be between 0 and 100.")

        row_count = len(self.dataframe)

        if row_count == 0:
            return []

        return [
            str(column)
            for column in self.dataframe.columns
            if (self.dataframe[column].isna().sum() / row_count) * 100
            >= threshold
        ]