"""Tests for schema analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from dataset_analyzer.analysis.schema import SchemaAnalyzer


def test_numeric_column_is_classified_as_numerical() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [21, 25, 30, 35],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].name == "age"
    assert profiles[0].raw_dtype == "int64"
    assert profiles[0].semantic_type == "numerical"


def test_string_column_is_classified_as_categorical() -> None:
    dataframe = pd.DataFrame(
        {
            "city": ["Pune", "Mumbai", "Delhi", "Pune"],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].semantic_type == "categorical"


def test_boolean_column_is_classified_as_boolean() -> None:
    dataframe = pd.DataFrame(
        {
            "is_active": [True, False, True, False],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].semantic_type == "boolean"


def test_datetime_column_is_classified_as_datetime() -> None:
    dataframe = pd.DataFrame(
        {
            "created_at": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].semantic_type == "datetime"


def test_datetime_name_does_not_classify_invalid_strings_as_datetime() -> None:
    dataframe = pd.DataFrame(
        {
            "created_at": [
                "not-a-date",
                "another-value",
                "something-else",
                "invalid",
            ],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].semantic_type == "categorical"


def test_identifier_name_is_classified_as_identifier() -> None:
    dataframe = pd.DataFrame(
        {
            "user_id": [101, 102, 103, 104],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].semantic_type == "identifier"


def test_high_cardinality_integer_column_is_classified_as_identifier() -> None:
    dataframe = pd.DataFrame(
        {
            "value": list(range(100)),
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].semantic_type == "identifier"


def test_low_cardinality_numeric_column_can_be_categorical() -> None:
    dataframe = pd.DataFrame(
        {
            "category_code": [1, 2, 1, 3, 2, 1],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].semantic_type == "categorical"


def test_missing_and_cardinality_values_are_calculated() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [20, 20, None, 30, 30],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].missing_count == 1
    assert profiles[0].missing_percentage == 20.0
    assert profiles[0].unique_count == 2
    assert profiles[0].cardinality_ratio == 0.5


def test_empty_dataframe_is_handled() -> None:
    dataframe = pd.DataFrame(
        {
            "age": pd.Series(dtype="float64"),
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert len(profiles) == 1
    assert profiles[0].missing_count == 0
    assert profiles[0].missing_percentage == 0.0
    assert profiles[0].unique_count == 0
    assert profiles[0].cardinality_ratio == 0.0


def test_entirely_null_column_is_supported() -> None:
    dataframe = pd.DataFrame(
        {
            "unknown": [None, None, None, None],
        }
    )

    profiles = SchemaAnalyzer(dataframe).analyze()

    assert profiles[0].missing_count == 4
    assert profiles[0].missing_percentage == 100.0
    assert profiles[0].unique_count == 0
    assert profiles[0].cardinality_ratio == 0.0
    assert profiles[0].semantic_type == "categorical"


def test_duplicate_column_names_raise_error() -> None:
    dataframe = pd.DataFrame(
        [
            [1, 2],
            [3, 4],
        ],
        columns=["value", "value"],
    )

    with pytest.raises(
        ValueError,
        match="duplicate column names",
    ):
        SchemaAnalyzer(dataframe).analyze()