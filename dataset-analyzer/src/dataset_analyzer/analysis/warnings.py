"""Warning generation for dataset analysis results."""

from __future__ import annotations

from ..profile.models import ColumnProfile, DataWarning


class WarningAnalyzer:
    """Generate warnings from analyzed column profiles."""

    def __init__(
        self,
        columns: list[ColumnProfile],
        high_missing_threshold: float = 50.0,
    ) -> None:
        """Initialize the warning analyzer.

        Parameters
        ----------
        columns:
            Column profiles produced by the dataset analysis components.
        high_missing_threshold:
            Missing-value percentage at or above which a column is
            considered high-missing.
        """
        if not 0 <= high_missing_threshold <= 100:
            raise ValueError(
                "high_missing_threshold must be between 0 and 100."
            )

        self.columns = columns
        self.high_missing_threshold = high_missing_threshold

    def analyze(self) -> list[DataWarning]:
        """Generate warnings for notable column-level conditions."""
        warnings: list[DataWarning] = []

        for column in self.columns:
            self._add_missing_warning(column, warnings)
            self._add_constant_warning(column, warnings)
            self._add_identifier_warning(column, warnings)
            self._add_outlier_warning(column, warnings)

        return warnings

    def _add_missing_warning(
        self,
        column: ColumnProfile,
        warnings: list[DataWarning],
    ) -> None:
        """Add a warning when a column has substantial missing values."""
        if column.missing_percentage < self.high_missing_threshold:
            return

        warnings.append(
            DataWarning(
                type="high_missing_values",
                message=(
                    f"Column '{column.name}' has "
                    f"{column.missing_percentage:.2f}% missing values."
                ),
                severity="high",
            )
        )

    @staticmethod
    def _add_constant_warning(
        column: ColumnProfile,
        warnings: list[DataWarning],
    ) -> None:
        """Add a warning when a column has at most one unique value."""
        if column.unique_count > 1:
            return

        warnings.append(
            DataWarning(
                type="constant_column",
                message=(
                    f"Column '{column.name}' contains no more than one "
                    "unique non-null value."
                ),
                severity="medium",
            )
        )

    @staticmethod
    def _add_identifier_warning(
        column: ColumnProfile,
        warnings: list[DataWarning],
    ) -> None:
        """Add a warning when a column was classified as an identifier."""
        if column.semantic_type != "identifier":
            return

        warnings.append(
            DataWarning(
                type="identifier_column",
                message=(
                    f"Column '{column.name}' appears to be an identifier "
                    "and may not provide useful predictive information."
                ),
                severity="low",
            )
        )

    @staticmethod
    def _add_outlier_warning(
        column: ColumnProfile,
        warnings: list[DataWarning],
    ) -> None:
        """Add a warning when a numerical column contains many outliers."""
        if column.outlier_percentage is None:
            return

        if column.outlier_percentage < 5.0:
            return

        warnings.append(
            DataWarning(
                type="high_outlier_values",
                message=(
                    f"Column '{column.name}' contains "
                    f"{column.outlier_percentage:.2f}% statistical outliers."
                ),
                severity="medium",
            )
        )