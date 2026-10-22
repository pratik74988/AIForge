"""Schema analysis for tabular datasets."""

from __future__ import annotations

import re

import pandas as pd

from ..profile.models import ColumnProfile


class SchemaAnalyzer:
    """Analyze column dtypes and infer their semantic types."""

    _ID_PATTERN = re.compile(
        r"(^|_)(id|uuid|guid)(_|$)|"
        r"(^|_)(customer|user|account|transaction|order)_?id$",
        re.IGNORECASE,
    )

    _CATEGORICAL_PATTERN = re.compile(
        r"(^|_)(zip|zipcode|postal|postal_code|pin|pincode|"
        r"code|category|type|class|gender|sex|state|country|city)(_|$)",
        re.IGNORECASE,
    )

    _DATETIME_PATTERN = re.compile(
        r"(^|_)(date|datetime|timestamp|created|updated|modified|"
        r"birth|dob|time)(_|$)",
        re.IGNORECASE,
    )

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize the analyzer with a DataFrame."""
        self.dataframe = dataframe

    def analyze(self) -> list[ColumnProfile]:
        """Analyze all columns and return their schema profiles."""
        if self.dataframe.columns.has_duplicates:
            raise ValueError("Dataset contains duplicate column names.")

        return [
            self._analyze_column(name, self.dataframe[name])
            for name in self.dataframe.columns
        ]

    def _analyze_column(
        self,
        name: str,
        series: pd.Series,
    ) -> ColumnProfile:
        """Create a profile for a single column."""
        missing_count = int(series.isna().sum())
        missing_percentage = (
            missing_count / len(series) * 100
            if len(series) > 0
            else 0.0
        )

        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        cardinality_ratio = (
            unique_count / len(non_null)
            if len(non_null) > 0
            else 0.0
        )

        semantic_type = self._infer_semantic_type(
            name,
            series,
            unique_count,
            cardinality_ratio,
        )

        return ColumnProfile(
            name=str(name),
            raw_dtype=str(series.dtype),
            semantic_type=semantic_type,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            cardinality_ratio=cardinality_ratio,
        )

    def _infer_semantic_type(
        self,
        name: str,
        series: pd.Series,
        unique_count: int,
        cardinality_ratio: float,
    ) -> str:
        """Infer the semantic type of a column."""
        if self._looks_like_identifier(name, series, cardinality_ratio):
            return "identifier"

        if self._looks_like_datetime(name, series):
            return "datetime"

        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"

        if pd.api.types.is_numeric_dtype(series):
            if self._looks_like_categorical(name, unique_count, cardinality_ratio):
                return "categorical"
            return "numerical"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if pd.api.types.is_timedelta64_dtype(series):
            return "timedelta"

        return "categorical"

    def _looks_like_identifier(
        self,
        name: str,
        series: pd.Series,
        cardinality_ratio: float,
    ) -> bool:
        """Detect columns that are likely identifiers."""
        if self._ID_PATTERN.search(str(name)):
            return True

        if len(series.dropna()) < 10:
            return False

        if cardinality_ratio < 0.98:
            return False

        return (
            pd.api.types.is_integer_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
        )

    def _looks_like_categorical(
        self,
        name: str,
        unique_count: int,
        cardinality_ratio: float,
    ) -> bool:
        """Detect categorical columns, including numeric categories."""
        if self._CATEGORICAL_PATTERN.search(str(name)):
            return True

        return unique_count <= 20 and cardinality_ratio <= 0.05

    def _looks_like_datetime(
        self,
        name: str,
        series: pd.Series,
    ) -> bool:
        """Detect date-like string columns when the name supports the inference."""
        if not self._DATETIME_PATTERN.search(str(name)):
            return False

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False

        non_null = series.dropna()

        if non_null.empty:
            return False

        sample = non_null.astype("string").head(1000)

        parsed = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed",
        )

        return float(parsed.notna().mean()) >= 0.90