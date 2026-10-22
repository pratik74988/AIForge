"""Duplicate-row analysis for tabular datasets."""

from __future__ import annotations

import pandas as pd


class DuplicateAnalyzer:
    """Analyze duplicate rows in a pandas DataFrame."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize the analyzer with a DataFrame."""
        self.dataframe = dataframe

    def analyze(self) -> dict[str, int | float]:
        """Return duplicate-row count and percentage.

        Duplicate rows are counted using all columns. The first occurrence
        of each row is treated as the original, so only subsequent identical
        rows are counted as duplicates.

        Returns
        -------
        dict[str, int | float]
            Dictionary containing:

            - ``duplicate_rows``: number of duplicate rows
            - ``duplicate_percentage``: duplicates as a percentage of all rows

        Notes
        -----
        An empty DataFrame has zero duplicate rows and a zero duplicate
        percentage.
        """
        row_count = len(self.dataframe)

        if row_count == 0:
            return {
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
            }

        duplicate_rows = int(self.dataframe.duplicated().sum())
        duplicate_percentage = duplicate_rows / row_count * 100

        return {
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": round(duplicate_percentage, 4),
        }