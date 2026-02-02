import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AbstractImputer(ABC):
    """Abstract base class for all imputation models."""

    def __init__(self):
        self._is_fitted: bool = False
        self._feature_names: Optional[list] = None
        self._n_features: Optional[int] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AbstractImputer":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set_params(self, **params) -> "AbstractImputer":
        pass

    def _validate_is_fitted(self) -> None:
        """Check if the model is fitted."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet. Call 'fit' first.")

    def _validate_input_shape(self, X: np.ndarray) -> None:
        """Validate input shape consistency."""
        if self._n_features and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")


class AbstractPreprocessor(ABC):
    """Abstract base class for data preprocessors."""

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "AbstractPreprocessor":
        """Fit the preprocessor."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
