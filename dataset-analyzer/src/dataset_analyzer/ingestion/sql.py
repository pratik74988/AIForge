"""SQL dataset ingestion."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseIngestor


class SQLIngestor(BaseIngestor):
    """Load tabular datasets from a SQL database."""

    def __init__(
        self,
        connection_string: str,
        query: str,
        **read_sql_kwargs: Any,
    ) -> None:
        """Initialize the SQL ingestor.

        Parameters
        ----------
        connection_string:
            SQLAlchemy database connection URL.
        query:
            SQL query used to retrieve the dataset.
        **read_sql_kwargs:
            Additional keyword arguments passed to ``pandas.read_sql``.
        """
        self.connection_string = connection_string
        self.query = query
        self.read_sql_kwargs = read_sql_kwargs

    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Execute the SQL query and load its result into a DataFrame.

        Parameters
        ----------
        **kwargs:
            Additional arguments passed to ``pandas.read_sql``. These
            override arguments supplied during initialization.

        Returns
        -------
        pandas.DataFrame
            Data retrieved by the SQL query.

        Raises
        ------
        ImportError
            If SQLAlchemy is not installed.
        """
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise ImportError(
                "SQL ingestion requires the 'sqlalchemy' package."
            ) from exc

        read_options = {
            **self.read_sql_kwargs,
            **kwargs,
        }

        engine = create_engine(self.connection_string)

        try:
            return pd.read_sql(
                self.query,
                engine,
                **read_options,
            )
        finally:
            engine.dispose()