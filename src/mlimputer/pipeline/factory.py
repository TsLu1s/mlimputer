from typing import Dict, Any, Type, Optional, List, Union
from mlimputer.core.abstract import AbstractImputer
from mlimputer.utils.constants import ImputationStrategy 
from mlimputer.core.strategy_mapper import StrategyMapper
from mlimputer.schemas.config import (
    RandomForestConfig, ExtraTreesConfig, GradientBoostingConfig,
    KNNConfig, XGBoostConfig, CatBoostConfig
)
from mlimputer.utils.exceptions import ConfigurationError
from mlimputer.utils.logging import logger


class ImputerFactory:
    """Factory for creating imputation models."""
    
    _registry: Dict[ImputationStrategy, Type[AbstractImputer]] = {}
    _config_mapping: Dict[ImputationStrategy, Type] = {
        ImputationStrategy.RANDOM_FOREST: RandomForestConfig,
        ImputationStrategy.EXTRA_TREES: ExtraTreesConfig,
        ImputationStrategy.GBR: GradientBoostingConfig,
        ImputationStrategy.KNN: KNNConfig,
        ImputationStrategy.XGBOOST: XGBoostConfig,
        ImputationStrategy.CATBOOST: CatBoostConfig,
    }
    
    @classmethod
    def register(
        cls,
        strategy: Union[ImputationStrategy, str],
        imputer_class: Type[AbstractImputer]
    ) -> None:
        """Register a new imputer class."""
        # Convert to enum using centralized mapper
        try:
            strategy_enum = StrategyMapper.to_enum(strategy)
        except ValueError as e:
            return
        
        cls._registry[strategy_enum] = imputer_class
    
    @classmethod
    def create(
        cls,
        strategy: Union[ImputationStrategy, str],
        config: Optional[Dict[str, Any]] = None
    ) -> AbstractImputer:
        """Create an imputer instance."""
        # Convert to enum using centralized mapper
        try:
            strategy_enum = StrategyMapper.to_enum(strategy)
        except ValueError as e:
            raise ConfigurationError(
                str(e),
                details={"available": cls.get_available_strategies()}
            )
        
        if strategy_enum not in cls._registry:
            # Attempt to auto-import and register
            cls._auto_register_models()
            
            if strategy_enum not in cls._registry:
                raise ConfigurationError(
                    f"Strategy not registered: {strategy_enum.value}",
                    details={
                        "requested": StrategyMapper.to_string(strategy_enum),
                        "available": [StrategyMapper.to_string(s) for s in cls._registry.keys()],
                        "registry_size": len(cls._registry)
                    }
                )
        
        imputer_class = cls._registry[strategy_enum]
        
        # Validate configuration
        if config:
            config_class = cls._config_mapping.get(strategy_enum)
            if config_class:
                validated_config = config_class(**config)
                config = validated_config.model_dump()
        
        try:
            imputer = imputer_class(**config) if config else imputer_class()
            return imputer
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create imputer: {str(e)}",
                details={
                    "strategy": StrategyMapper.to_string(strategy_enum),
                    "config": config
                }
            )
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategies as user-friendly strings."""
        if not cls._registry:
            cls._auto_register_models()
        return [StrategyMapper.to_string(s) for s in cls._registry.keys()]
    
    @classmethod
    def _auto_register_models(cls):
        """Auto-register all available models."""
        try:
            from mlimputer.models import tree_based, neighbors, boosting
        except ImportError as e:
            logger.warning(f"Could not load models: {e}")
    
    @classmethod
    def debug_registry(cls):
        """Debug method to check registry status."""
        print(f"Registry size: {len(cls._registry)}")
        print("Registered strategies:")
        for strategy, imputer_class in cls._registry.items():
            user_string = StrategyMapper.to_string(strategy)
            print(f"  {user_string} ({strategy.value}): {imputer_class.__name__}")

























