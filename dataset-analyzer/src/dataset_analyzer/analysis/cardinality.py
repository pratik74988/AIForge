"""Cardinality analysis for tabular datasets."""

from __future__ import annotations

import pandas as pd


class CardinalityAnalyzer:
    """Analyze the number and proportion of unique values in each column."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize the analyzer with a DataFrame."""
        self.dataframe = dataframe

    def analyze(self) -> dict[str, dict[str, int | float]]:
        """Return cardinality statistics for every column.

        Null values are excluded when counting unique values because
        cardinality describes the distinct observed values in the column.

        Returns
        -------
        dict[str, dict[str, int | float]]
            Mapping of column names to unique-value count and cardinality
            ratio.

            Example:
            {
                "gender": {
                    "unique_count": 2,
                    "cardinality_ratio": 0.02,
                }
            }

        Notes
        -----
        For an entirely null column, ``unique_count`` is zero and
        ``cardinality_ratio`` is zero.
        """
        results: dict[str, dict[str, int | float]] = {}

        for column in self.dataframe.columns:
            series = self.dataframe[column].dropna()

            unique_count = int(series.nunique())
            cardinality_ratio = (
                unique_count / len(series)
                if len(series) > 0
                else 0.0
            )

            results[str(column)] = {
                "unique_count": unique_count,
                "cardinality_ratio": round(cardinality_ratio, 6),
            }

        return results