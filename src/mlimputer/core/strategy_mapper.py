from typing import Union
from mlimputer.utils.constants import ImputationStrategy


class StrategyMapper:
    """Handles conversion between user-friendly strings and ImputationStrategy enums."""
    
    # Complete mapping of all user-facing strings to enum values
    STRING_TO_ENUM = {
        # User-facing names -> Enum values
        "RandomForest": ImputationStrategy.RANDOM_FOREST,
        "ExtraTrees": ImputationStrategy.EXTRA_TREES,
        "GBR": ImputationStrategy.GBR,
        "KNN": ImputationStrategy.KNN,
        "XGBoost": ImputationStrategy.XGBOOST,
        "Catboost": ImputationStrategy.CATBOOST,
        
        # Also support enum value strings (for backwards compatibility)
        "RANDOM_FOREST": ImputationStrategy.RANDOM_FOREST,
        "EXTRA_TREES": ImputationStrategy.EXTRA_TREES,
        "XGBOOST": ImputationStrategy.XGBOOST,
        "CATBOOST": ImputationStrategy.CATBOOST,
        
        # Support lowercase versions
        "randomforest": ImputationStrategy.RANDOM_FOREST,
        "extratrees": ImputationStrategy.EXTRA_TREES,
        "gbr": ImputationStrategy.GBR,
        "knn": ImputationStrategy.KNN,
        "xgboost": ImputationStrategy.XGBOOST,
        "catboost": ImputationStrategy.CATBOOST,
    }
    
    @classmethod
    def to_enum(cls, strategy: Union[ImputationStrategy, str]) -> ImputationStrategy:
        # If already an enum, return it
        if isinstance(strategy, ImputationStrategy):
            return strategy
        
        # If string, try direct enum conversion first
        if isinstance(strategy, str):
            try:
                return ImputationStrategy(strategy)
            except ValueError:
                pass
            
            # Try mapping lookup
            if strategy in cls.STRING_TO_ENUM:
                return cls.STRING_TO_ENUM[strategy]
            
            # Try case-insensitive lookup as last resort
            strategy_lower = strategy.lower()
            if strategy_lower in cls.STRING_TO_ENUM:
                return cls.STRING_TO_ENUM[strategy_lower]
        
        # If we get here, strategy is unknown
        available = list(set(cls.STRING_TO_ENUM.keys()))
        raise ValueError(
            f"Unknown strategy: '{strategy}'. "
            f"Available options: {sorted(available)}"
        )
    
    @classmethod
    def to_string(cls, strategy: ImputationStrategy) -> str:
        
        # Reverse mapping for display
        enum_to_string = {
            ImputationStrategy.RANDOM_FOREST: "RandomForest",
            ImputationStrategy.EXTRA_TREES: "ExtraTrees",
            ImputationStrategy.GBR: "GBR",
            ImputationStrategy.KNN: "KNN",
            ImputationStrategy.XGBOOST: "XGBoost",
            ImputationStrategy.CATBOOST: "Catboost",
        }
        return enum_to_string.get(strategy, strategy.value)