"""CSV dataset ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseIngestor


class CSVIngestor(BaseIngestor):
    """Load tabular datasets from CSV files."""

    def __init__(
        self,
        path: str | Path,
        **read_csv_kwargs: Any,
    ) -> None:
        """Initialize the CSV ingestor.

        Parameters
        ----------
        path:
            Path to the CSV file.
        **read_csv_kwargs:
            Additional keyword arguments passed to ``pandas.read_csv``.
        """
        self.path = Path(path)
        self.read_csv_kwargs = read_csv_kwargs

    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Load the CSV file into a pandas DataFrame.

        Parameters
        ----------
        **kwargs:
            Additional arguments passed to ``pandas.read_csv``. These
            override arguments supplied during initialization.

        Returns
        -------
        pandas.DataFrame
            Data loaded from the CSV file.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If the supplied path does not point to a file.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"CSV file was not found: {self.path}"
            )

        if not self.path.is_file():
            raise ValueError(
                f"CSV path does not point to a file: {self.path}"
            )

        read_options = {
            **self.read_csv_kwargs,
            **kwargs,
        }

        return pd.read_csv(
            self.path,
            **read_options,
        )