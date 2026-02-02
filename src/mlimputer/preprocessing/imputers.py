import numpy as np
import pandas as pd

from typing import Optional, Literal

from sklearn.impute import SimpleImputer

from mlimputer.preprocessing.abstract import AbstractEncoder
from mlimputer.utils.logging import logger


class AutoSimpleImputer(AbstractEncoder):
    def __init__(
        self,
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
        fill_value: Optional[float] = None,
    ):
        """
        Initialize AutoSimpleImputer.
        """
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self._imputer: Optional[SimpleImputer] = None
        self._numeric_columns: Optional[list] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AutoSimpleImputer":
        """Fit the simple imputer."""
        super().fit(X, y)

        # Identify numeric columns
        self._numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        if not self._numeric_columns:
            logger.warning("No numeric columns found for imputation")
            return self

        # Create and fit imputer
        self._imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self._imputer.fit(X[self._numeric_columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform with simple imputation."""
        self._validate_fitted()
        X_imputed = X.copy()

        if self._imputer and self._numeric_columns:
            # Apply imputation only to numeric columns
            imputed_values = self._imputer.transform(X[self._numeric_columns])
            X_imputed[self._numeric_columns] = imputed_values

        return X_imputed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Simple imputation is not reversible."""
        logger.warning("SimpleImputer transformation is not reversible")
        return X
