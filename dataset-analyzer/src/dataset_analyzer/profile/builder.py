"""Build the final DatasetProfile from analyzer results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..profile.models import (
    ColumnProfile,
    DatasetMetadata,
    DatasetProfile,
    QualityProfile,
    RelationshipProfile,
    TargetProfile,
)


class DatasetProfileBuilder:
    """Assemble individual analyzer results into a DatasetProfile."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sampled: bool = False,
        analyzed_rows: int | None = None,
    ) -> None:
        """Initialize the profile builder.

        Parameters
        ----------
        dataframe:
            DataFrame used for analysis.
        sampled:
            Whether the DataFrame represents a sample of the original dataset.
        analyzed_rows:
            Number of rows actually analyzed. If omitted, the DataFrame row
            count is used.
        """
        self.dataframe = dataframe
        self.sampled = sampled
        self.analyzed_rows = (
            len(dataframe)
            if analyzed_rows is None
            else analyzed_rows
        )

    def build(
        self,
        columns: list[ColumnProfile],
        target: TargetProfile | None = None,
        duplicate_rows: int = 0,
        duplicate_percentage: float = 0.0,
        correlations: list[dict[str, Any]] | None = None,
        warnings: list[Any] | None = None,
    ) -> DatasetProfile:
        """Assemble analyzer outputs into a complete DatasetProfile.

        Parameters
        ----------
        columns:
            Column profiles produced by the schema analyzer.
        target:
            Optional target profile produced by the target analyzer.
        duplicate_rows:
            Number of duplicate rows detected.
        duplicate_percentage:
            Percentage of duplicate rows.
        correlations:
            Highly correlated feature pairs.
        warnings:
            Warnings produced by the warning analyzer.

        Returns
        -------
        DatasetProfile
            Fully assembled dataset profile.
        """
        quality = self._build_quality_profile(
            columns=columns,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
        )

        relationships = RelationshipProfile(
            highly_correlated_pairs=correlations or []
        )

        return DatasetProfile(
            dataset=DatasetMetadata(
                rows=len(self.dataframe),
                columns=len(self.dataframe.columns),
                analyzed_rows=self.analyzed_rows,
                sampled=self.sampled,
            ),
            columns=columns,
            target=target,
            quality=quality,
            relationships=relationships,
            warnings=warnings or [],
        )

    def _build_quality_profile(
        self,
        columns: list[ColumnProfile],
        duplicate_rows: int,
        duplicate_percentage: float,
    ) -> QualityProfile:
        """Build the dataset-level quality profile."""
        constant_columns = [
            column.name
            for column in columns
            if column.unique_count <= 1
        ]

        high_missing_columns = [
            column.name
            for column in columns
            if column.missing_percentage >= 50.0
        ]

        return QualityProfile(
            duplicate_rows=duplicate_rows,
            duplicate_percentage=round(duplicate_percentage, 4),
            constant_columns=constant_columns,
            high_missing_columns=high_missing_columns,
        )