import numpy as np

from mlimputer.core.abstract import AbstractImputer
from mlimputer.utils.logging import logger

from abc import abstractmethod
from typing import Dict, Any

class BaseImputationModel(AbstractImputer):
    """Base class for all imputation models."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.model_params = kwargs
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseImputationModel':
        """Fit the model."""
        self._n_features = X.shape[1] if X.ndim > 1 else 1
        self.model.fit(X, y)
        self._is_fitted = True
        logger.debug(f"{self.__class__.__name__} fitted on {len(X)} samples")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self._validate_is_fitted()
        self._validate_input_shape(X)
        return self.model.predict(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'BaseImputationModel':
        """Set model parameters."""
        self.model_params.update(params)
        self._is_fitted = False
        # Reinitialize model with new params
        self._initialize_model()
        return self
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the underlying model."""
        pass