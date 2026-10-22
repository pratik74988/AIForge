"""Outlier analysis for numerical dataset columns."""

from __future__ import annotations

import pandas as pd


class OutlierAnalyzer:
    """Detect statistical outliers in numerical columns using the IQR method."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        multiplier: float = 1.5,
    ) -> None:
        """Initialize the outlier analyzer.

        Parameters
        ----------
        dataframe:
            DataFrame to analyze.
        multiplier:
            IQR multiplier used to define the lower and upper bounds.
            The conventional value is 1.5.
        """
        if multiplier < 0:
            raise ValueError("multiplier must be non-negative.")

        self.dataframe = dataframe
        self.multiplier = multiplier

    def analyze(self) -> dict[str, dict[str, float | int]]:
        """Return outlier statistics for each numerical column.

        Returns
        -------
        dict[str, dict[str, float | int]]
            Mapping of column names to:

            - ``outlier_count``
            - ``outlier_percentage``
            - ``lower_bound``
            - ``upper_bound``

        Notes
        -----
        Missing and infinite values are excluded from the calculation.
        Columns with fewer than two valid observations are reported with
        zero outliers because an IQR-based estimate is not meaningful.
        """
        results: dict[str, dict[str, float | int]] = {}

        numerical_columns = self.dataframe.select_dtypes(
            include="number"
        ).columns

        for column in numerical_columns:
            series = (
                pd.to_numeric(
                    self.dataframe[column],
                    errors="coerce",
                )
                .replace([float("inf"), float("-inf")], pd.NA)
                .dropna()
            )

            if len(series) < 2:
                results[str(column)] = {
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                    "lower_bound": None,
                    "upper_bound": None,
                }
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            lower_bound = q1 - self.multiplier * iqr
            upper_bound = q3 + self.multiplier * iqr

            outlier_count = int(
                ((series < lower_bound) | (series > upper_bound)).sum()
            )

            outlier_percentage = outlier_count / len(series) * 100

            results[str(column)] = {
                "outlier_count": outlier_count,
                "outlier_percentage": round(outlier_percentage, 4),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

        return results