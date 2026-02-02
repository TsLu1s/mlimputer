import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseDatasetGenerator(ABC):
    """Abstract base class for dataset generation."""

    @abstractmethod
    def generate(self, n_samples: int, n_features: int, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate base dataset - must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default generation parameters."""
        pass

    def _validate_params(self, **kwargs) -> Dict[str, Any]:
        """Validate and merge parameters with defaults."""
        defaults = self._get_default_params()
        params = {**defaults, **kwargs}

        if params["n_samples"] < 10:
            raise ValueError("n_samples must be >= 10")
        if params["n_features"] < 2:
            raise ValueError("n_features must be >= 2")
        if not 0 <= params.get("missing_rate", 0) <= 1:
            raise ValueError("missing_rate must be between 0 and 1")

        return params
