from mlimputer.core.base import BaseImputationModel
from mlimputer.models.tree_based import RandomForestImputation, ExtraTreesImputation, GBRImputation
from mlimputer.models.neighbors import KNNImputation

# Optional boosting models
try:
    from mlimputer.models.boosting import XGBoostImputation, CatBoostImputation
    BOOSTING_AVAILABLE = True
except (ImportError, OSError) as e:
    BOOSTING_AVAILABLE = False
    import warnings
    warnings.warn(f"Boosting models unavailable: {e}")

def register_all_models():
    """Register all imputation models with the factory."""
    from mlimputer.pipeline.factory import ImputerFactory
    from mlimputer.utils.constants import ImputationStrategy
    
    # Tree-based models (always available)
    ImputerFactory.register(ImputationStrategy.RANDOM_FOREST, RandomForestImputation)
    ImputerFactory.register(ImputationStrategy.EXTRA_TREES, ExtraTreesImputation)
    ImputerFactory.register(ImputationStrategy.GBR, GBRImputation)
    
    # Neighbor-based models
    ImputerFactory.register(ImputationStrategy.KNN, KNNImputation)
    
    # Boosting models (optional)
    if BOOSTING_AVAILABLE:
        ImputerFactory.register(ImputationStrategy.XGBOOST, XGBoostImputation)
        ImputerFactory.register(ImputationStrategy.CATBOOST, CatBoostImputation)

# Register all models when module is imported
register_all_models()

__all__ = [
    'BaseImputationModel',
    'RandomForestImputation',
    'ExtraTreesImputation',
    'GBRImputation',
    'KNNImputation',
]

if BOOSTING_AVAILABLE:
    __all__.extend(['XGBoostImputation', 'CatBoostImputation'])