"""Main entry point for dataset analysis."""

from __future__ import annotations

import pandas as pd

from .analysis.cardinality import CardinalityAnalyzer
from .analysis.correlation import CorrelationAnalyzer
from .analysis.duplicates import DuplicateAnalyzer
from .analysis.missing import MissingValueAnalyzer
from .analysis.outliers import OutlierAnalyzer
from .analysis.schema import SchemaAnalyzer
from .analysis.statistics import StatisticsAnalyzer
from .analysis.target import TargetAnalyzer
from .analysis.warnings import WarningAnalyzer
from .profile.builder import DatasetProfileBuilder
from .profile.models import DatasetProfile


class DatasetAnalyzer:
    """Run all dataset analysis components and build a DatasetProfile."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str | None = None,
        sampled: bool = False,
        analyzed_rows: int | None = None,
    ) -> None:
        """Initialize the dataset analyzer.

        Parameters
        ----------
        dataframe:
            DataFrame to analyze.
        target:
            Optional name of the target column. If provided, target analysis
            is performed.
        sampled:
            Whether the provided DataFrame represents a sample of the
            original dataset.
        analyzed_rows:
            Number of rows actually analyzed. If omitted, the DataFrame row
            count is used.
        """
        self.dataframe = dataframe
        self.target = target
        self.sampled = sampled
        self.analyzed_rows = analyzed_rows

    def analyze(self) -> DatasetProfile:
        """Run all analyzers and return the complete dataset profile."""
        columns = SchemaAnalyzer(self.dataframe).analyze()

        missing = MissingValueAnalyzer(self.dataframe).analyze()
        cardinality = CardinalityAnalyzer(self.dataframe).analyze()
        statistics = StatisticsAnalyzer(self.dataframe).analyze()
        duplicates = DuplicateAnalyzer(self.dataframe).analyze()

        outliers = OutlierAnalyzer(self.dataframe).analyze()
        correlations = CorrelationAnalyzer(self.dataframe).analyze()

        target_profile = None

        if self.target is not None:
            target_profile = TargetAnalyzer(
                self.dataframe,
                self.target,
            ).analyze()

        columns = self._merge_column_analysis(
            columns=columns,
            missing=missing,
            cardinality=cardinality,
            statistics=statistics,
            outliers=outliers,
        )

        warnings = WarningAnalyzer(columns).analyze()

        builder = DatasetProfileBuilder(
            dataframe=self.dataframe,
            sampled=self.sampled,
            analyzed_rows=self.analyzed_rows,
        )

        return builder.build(
            columns=columns,
            target=target_profile,
            duplicate_rows=duplicates["duplicate_rows"],
            duplicate_percentage=duplicates["duplicate_percentage"],
            correlations=correlations.highly_correlated_pairs,
            warnings=warnings,
        )

    @staticmethod
    def _merge_column_analysis(
        columns: list,
        missing: dict[str, dict[str, float | int]],
        cardinality: dict[str, dict[str, float | int]],
        statistics: dict,
        outliers: dict[str, dict[str, float | int | None]],
    ) -> list:
        """Merge individual analyzer results into column profiles."""
        for column in columns:
            name = column.name

            missing_result = missing.get(name, {})
            cardinality_result = cardinality.get(name, {})
            statistics_result = statistics.get(name)
            outlier_result = outliers.get(name, {})

            column.missing_count = int(
                missing_result.get(
                    "missing_count",
                    column.missing_count,
                )
            )

            column.missing_percentage = float(
                missing_result.get(
                    "missing_percentage",
                    column.missing_percentage,
                )
            )

            column.unique_count = int(
                cardinality_result.get(
                    "unique_count",
                    column.unique_count,
                )
            )

            column.cardinality_ratio = float(
                cardinality_result.get(
                    "cardinality_ratio",
                    column.cardinality_ratio,
                )
            )

            if statistics_result is not None:
                column.statistics = statistics_result

            column.outlier_count = _optional_int(
                outlier_result.get("outlier_count")
            )
            column.outlier_percentage = _optional_float(
                outlier_result.get("outlier_percentage")
            )
            column.lower_bound = _optional_float(
                outlier_result.get("lower_bound")
            )
            column.upper_bound = _optional_float(
                outlier_result.get("upper_bound")
            )

        return columns


def _optional_int(value: object) -> int | None:
    """Convert a value to an integer when present."""
    if value is None:
        return None

    return int(value)


def _optional_float(value: object) -> float | None:
    """Convert a value to a float when present."""
    if value is None:
        return None

    return float(value)