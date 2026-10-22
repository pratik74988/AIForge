"""Correlation analysis for numerical dataset columns."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..profile.models import RelationshipProfile


class CorrelationAnalyzer:
    """Analyze pairwise correlations between numerical columns."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.90,
    ) -> None:
        """Initialize the correlation analyzer.

        Parameters
        ----------
        dataframe:
            DataFrame to analyze.
        threshold:
            Absolute correlation value at or above which a pair is considered
            highly correlated. Must be between 0 and 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

        self.dataframe = dataframe
        self.threshold = threshold

    def analyze(self) -> RelationshipProfile:
        """Return highly correlated numerical feature pairs.

        Correlations are calculated using Pearson correlation. Only the
        upper triangle of the correlation matrix is inspected, so each pair
        appears once and self-correlations are excluded.

        Columns containing insufficient valid numeric observations are
        naturally ignored by pandas' correlation calculation.
        """
        numerical = self.dataframe.select_dtypes(include="number")

        if numerical.shape[1] < 2:
            return RelationshipProfile(
                highly_correlated_pairs=[]
            )

        correlation_matrix = numerical.corr(method="pearson")

        pairs: list[dict[str, Any]] = []

        columns = correlation_matrix.columns

        for i, first_column in enumerate(columns):
            for second_column in columns[i + 1:]:
                correlation = correlation_matrix.loc[
                    first_column,
                    second_column,
                ]

                if pd.isna(correlation):
                    continue

                if abs(correlation) >= self.threshold:
                    pairs.append(
                        {
                            "feature_1": str(first_column),
                            "feature_2": str(second_column),
                            "correlation": round(float(correlation), 6),
                        }
                    )

        return RelationshipProfile(
            highly_correlated_pairs=pairs
        )