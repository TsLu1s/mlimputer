import pandas as pd

from abc import ABC, abstractmethod
from typing import Optional

from sklearn.base import TransformerMixin


class AbstractEncoder(TransformerMixin, ABC):
    """Abstract base class for encoders."""

    def __init__(self):
        self._is_fitted: bool = False
        self._columns: Optional[pd.Index] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AbstractEncoder":
        """Fit the encoder."""
        self._columns = X.columns
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._validate_fitted()
        return X

    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the data."""
        self._validate_fitted()
        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _validate_fitted(self) -> None:
        """Check if encoder is fitted."""
        if not self._is_fitted:
            raise ValueError(f"{self.__class__.__name__} has not been fitted yet.")

    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that columns match."""
        if self._columns is None:
            raise ValueError("Encoder has not been fitted yet.")
        if not all(col in X.columns for col in self._columns):
            missing_cols = set(self._columns) - set(X.columns)
            raise ValueError(f"Missing columns: {missing_cols}")
