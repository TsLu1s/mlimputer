from typing import Dict, Any, Optional
from dataclasses import dataclass
from mlimputer.schemas.config import (
    RandomForestConfig, ExtraTreesConfig, GradientBoostingConfig,
    KNNConfig, XGBoostConfig, CatBoostConfig
)


@dataclass
class ImputerParameters:
    """Container for all imputer parameters with dict-like interface."""
    
    RandomForest: Dict[str, Any]
    ExtraTrees: Dict[str, Any]
    GBR: Dict[str, Any]
    KNN: Dict[str, Any]
    XGBoost: Dict[str, Any]
    Catboost: Dict[str, Any]
    
    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Allow dict-like access."""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        """Allow dict-like setting."""
        setattr(self, key, value)
    
    def get(self, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Configuration dictionary or default
        """
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default if default is not None else {}
    
    def keys(self):
        """Get available model keys."""
        return [
            'RandomForest', 'ExtraTrees', 'GBR',
            'KNN', 'XGBoost', 'Catboost'
        ]
    
    def values(self):
        """Get all configuration values."""
        return [getattr(self, key) for key in self.keys()]
    
    def items(self):
        """Get all key-value pairs."""
        return [(key, getattr(self, key)) for key in self.keys()]
    
    def update(self, other: Dict[str, Dict[str, Any]]) -> None:
        """Update multiple configurations at once."""
        for key, value in other.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if a model configuration exists."""
        return hasattr(self, key)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ImputerParameters(models={self.keys()})"


def imputer_parameters() -> ImputerParameters:
    """
    Get default imputation parameters with validation.
    """
    # Validate each configuration with Pydantic
    rf_config = RandomForestConfig()
    et_config = ExtraTreesConfig()
    gbr_config = GradientBoostingConfig()
    knn_config = KNNConfig()
    xgb_config = XGBoostConfig()
    cb_config = CatBoostConfig()
    
    return ImputerParameters(
        RandomForest=rf_config.model_dump(),
        ExtraTrees=et_config.model_dump(),
        GBR=gbr_config.model_dump(),
        KNN=knn_config.model_dump(),
        XGBoost=xgb_config.model_dump(),
        Catboost=cb_config.model_dump()
    )


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    """
    params = imputer_parameters()
    if model_name not in params.keys():
        raise ValueError(f"Unknown model: {model_name}")
    return params[model_name]


def update_model_config(
    model_name: str,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update model configuration with validation.
    """
    config_map = {
        'RandomForest': RandomForestConfig,
        'ExtraTrees': ExtraTreesConfig,
        'GBR': GradientBoostingConfig,
        'KNN': KNNConfig,
        'XGBoost': XGBoostConfig,
        'Catboost': CatBoostConfig
    }
    
    config_class = config_map.get(model_name)
    if not config_class:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get current config
    current = get_model_config(model_name)
    current.update(updates)
    
    # Validate with Pydantic
    validated = config_class(**current)
    return validated.model_dump()