"""Base interface for dataset ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseIngestor(ABC):
    """Abstract interface implemented by dataset ingestion sources."""

    @abstractmethod
    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Load data from the configured source.

        Returns
        -------
        pandas.DataFrame
            Dataset loaded into memory for analysis.

        Raises
        ------
        Exception
            Implementations should raise an appropriate exception when the
            source cannot be accessed or the data cannot be loaded.
        """
        raise NotImplementedError