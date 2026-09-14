"""MongoDB dataset ingestion."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseIngestor


class MongoDBIngestor(BaseIngestor):
    """Load documents from a MongoDB collection into a pandas DataFrame."""

    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str,
        **find_kwargs: Any,
    ) -> None:
        """Initialize the MongoDB ingestor.

        Parameters
        ----------
        connection_string:
            MongoDB connection URI.
        database:
            Name of the MongoDB database.
        collection:
            Name of the collection containing the dataset.
        **find_kwargs:
            Additional arguments passed to ``collection.find``.
        """
        self.connection_string = connection_string
        self.database = database
        self.collection = collection
        self.find_kwargs = find_kwargs

    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Load MongoDB documents into a pandas DataFrame.

        Parameters
        ----------
        **kwargs:
            Additional arguments passed to ``collection.find``. These
            override arguments supplied during initialization.

        Returns
        -------
        pandas.DataFrame
            Documents loaded from the MongoDB collection.

        Raises
        ------
        ImportError
            If ``pymongo`` is not installed.
        ValueError
            If the collection contains no documents.
        """
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise ImportError(
                "MongoDB ingestion requires the 'pymongo' package."
            ) from exc

        find_options = {
            **self.find_kwargs,
            **kwargs,
        }

        client = MongoClient(self.connection_string)

        try:
            collection = client[
                self.database
            ][
                self.collection
            ]

            documents = list(
                collection.find(
                    **find_options,
                )
            )

            if not documents:
                raise ValueError(
                    f"MongoDB collection '{self.collection}' contains "
                    "no documents."
                )

            return pd.DataFrame(documents)
        finally:
            client.close()