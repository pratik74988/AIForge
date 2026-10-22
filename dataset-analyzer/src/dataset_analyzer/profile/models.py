"""Pydantic models for dataset analysis and profiling."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata describing the analyzed dataset."""

    rows: int
    columns: int
    analyzed_rows: int
    sampled: bool = False


class ColumnStatistics(BaseModel):
    """Descriptive statistics for a numerical column."""

    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


class ColumnProfile(BaseModel):
    """Profile describing an individual dataset column."""

    name: str

    raw_dtype: str
    semantic_type: str

    missing_count: int
    missing_percentage: float

    unique_count: int
    cardinality_ratio: float

    statistics: ColumnStatistics | None = None

    outlier_count: int | None = None
    outlier_percentage: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


class TargetProfile(BaseModel):
    """Profile describing the user-specified target column."""

    name: str
    semantic_type: str
    task: str

    unique_count: int

    class_distribution: dict[str, int] | None = None
    imbalance_ratio: float | None = None


class QualityProfile(BaseModel):
    """Dataset-level data quality information."""

    duplicate_rows: int
    duplicate_percentage: float

    constant_columns: list[str]
    high_missing_columns: list[str]


class RelationshipProfile(BaseModel):
    """Relationships detected between dataset columns."""

    highly_correlated_pairs: list[dict[str, Any]]


class DataWarning(BaseModel):
    """A warning generated from dataset analysis."""

    type: str
    message: str
    severity: str


class DatasetProfile(BaseModel):
    """Complete structured profile produced by the dataset analyzer."""

    dataset: DatasetMetadata

    columns: list[ColumnProfile]

    target: TargetProfile | None = None

    quality: QualityProfile

    relationships: RelationshipProfile

    warnings: list[DataWarning] = Field(default_factory=list)