"""Ingestion registry for dataset sources."""

from __future__ import annotations

from typing import TypeAlias

from .base import BaseIngestor
from .csv import CSVIngestor
from .mongodb import MongoDBIngestor
from .sql import SQLIngestor


IngestorClass: TypeAlias = type[BaseIngestor]


class IngestionRegistry:
    """Registry mapping ingestion source names to ingestor classes."""

    _ingestors: dict[str, IngestorClass] = {
        "csv": CSVIngestor,
        "mongodb": MongoDBIngestor,
        "sql": SQLIngestor,
    }

    @classmethod
    def register(
        cls,
        name: str,
        ingestor: IngestorClass,
    ) -> None:
        """Register an ingestor under a source name.

        Parameters
        ----------
        name:
            Name used to identify the ingestion source.
        ingestor:
            Ingestor class implementing ``BaseIngestor``.

        Raises
        ------
        ValueError
            If the name is empty or already registered.
        TypeError
            If the supplied class does not implement ``BaseIngestor``.
        """
        normalized_name = name.strip().lower()

        if not normalized_name:
            raise ValueError("Ingestion source name cannot be empty.")

        if normalized_name in cls._ingestors:
            raise ValueError(
                f"Ingestion source '{normalized_name}' is already registered."
            )

        if not issubclass(ingestor, BaseIngestor):
            raise TypeError(
                "Registered ingestor must inherit from BaseIngestor."
            )

        cls._ingestors[normalized_name] = ingestor

    @classmethod
    def get(cls, name: str) -> IngestorClass:
        """Return the ingestor class registered for a source.

        Raises
        ------
        ValueError
            If no ingestor is registered for the requested source.
        """
        normalized_name = name.strip().lower()

        try:
            return cls._ingestors[normalized_name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._ingestors))
            raise ValueError(
                f"Unknown ingestion source '{name}'. "
                f"Available sources: {available}."
            ) from exc

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: object,
    ) -> BaseIngestor:
        """Create an ingestor instance for a registered source."""
        ingestor_class = cls.get(name)
        return ingestor_class(**kwargs)