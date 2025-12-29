from mlimputer.core.base import BaseImputationModel

from mlimputer.models.tree_based import RandomForestImputation, ExtraTreesImputation, GBRImputation
from mlimputer.models.neighbors import KNNImputation
from mlimputer.models.boosting import XGBoostImputation, CatBoostImputation

# Register all models with factory
def register_all_models():
    """Register all imputation models with the factory."""
    from mlimputer.pipeline.factory import ImputerFactory
    from mlimputer.utils.constants import ImputationStrategy
    
    # Tree-based models
    ImputerFactory.register(ImputationStrategy.RANDOM_FOREST, RandomForestImputation)
    ImputerFactory.register(ImputationStrategy.EXTRA_TREES, ExtraTreesImputation)
    ImputerFactory.register(ImputationStrategy.GBR, GBRImputation)
    
    # Neighbor-based models
    ImputerFactory.register(ImputationStrategy.KNN, KNNImputation)
    
    # Boosting models
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
    'XGBoostImputation',
    'CatBoostImputation',
]