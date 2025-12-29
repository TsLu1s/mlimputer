import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Optional, Union, List

from mlimputer.pipeline.base import AbstractMLImputer, ImputationState
from mlimputer.utils.constants import ImputationStrategy, ImputationContext, ImputationColumn, ImputerStrategy
from mlimputer.core.strategy_mapper import StrategyMapper
from mlimputer.preprocessing.encoders import AutoLabelEncoder
from mlimputer.preprocessing.imputers import AutoSimpleImputer
from mlimputer.pipeline.factory import ImputerFactory
from mlimputer.schemas.parameters import imputer_parameters
from mlimputer.schemas.config import (
    ImputerConfig, RandomForestConfig, ExtraTreesConfig,
    GradientBoostingConfig, KNNConfig, XGBoostConfig,
    CatBoostConfig
)
from mlimputer.utils.logging import logger
from mlimputer.utils.exceptions import ModelNotFittedError


class MLimputer(AbstractMLImputer):
    """Optimized ML-based imputation implementation."""
    
    # Config mapping for validation
    _CONFIG_MAPPING = {
        "RandomForest": RandomForestConfig,
        "ExtraTrees": ExtraTreesConfig,
        "GBR": GradientBoostingConfig,
        "KNN": KNNConfig,
        "XGBoost": XGBoostConfig,
        "Catboost": CatBoostConfig
    }
    
    def __init__(
        self,
        imput_model: str = "RandomForest",
        imputer_configs: Optional[Union[Dict[str, Any], ImputerConfig]] = None
    ):
        """
        Initialize MLimputer with validation.
        """
        # Map string to strategy enum
        strategy = self._map_strategy(imput_model)
        super().__init__(strategy)
        
        # Validate and store configuration
        self._model_name = imput_model
        self._config = self._validate_config(imput_model, imputer_configs)
            
    def fit(self, X: pd.DataFrame) -> 'MLimputer':
        """
        Fit imputation models for columns with missing data.
        """
        super().fit(X)  # Validates data
        
        # Identify columns needing imputation
        missing_report = self._get_missing_columns(X)
        if missing_report.empty:
            logger.info("No missing values found in numeric columns")
            self._state = ImputationState(
                strategy=self._strategy,
                fit_timestamp=datetime.now()
            )
            return self
        
        # Extract target columns
        target_columns = missing_report[ImputationColumn.COLUMN_NAME.value].tolist()
        
        # Fit imputers for each column
        fitted_configs = self._fit_all_columns(X, target_columns, missing_report)
        
        # Update state
        self._state = ImputationState(
            strategy=self._strategy,
            column_order=target_columns,
            fitted_models=fitted_configs,
            fit_timestamp=datetime.now(),
            metadata={'n_columns': len(target_columns)}
        )
        self._fitted_configs = fitted_configs
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by imputing missing values.
        """
        super().transform(X)  # Validates
        
        X_imputed = X.copy()
        
        # Apply imputation to each fitted column
        for column in tqdm(
            self._state.column_order, 
            desc="Imputing", 
            disable=not self._should_show_progress()
        ):
            if column in X_imputed.columns and column in self._fitted_configs:
                self._apply_column_imputation(
                    X_imputed, 
                    column, 
                    self._fitted_configs[column]
                )
        
        return X_imputed
    
    # Private methods (optimized and focused)
    
    def _map_strategy(self, model_name: str) -> ImputationStrategy:
        """Map model name to strategy enum."""
        try:
            return StrategyMapper.to_enum(model_name)
        except ValueError as e:
            logger.error(f"Invalid strategy: {model_name}")
            raise ValueError(f"Unknown imputation model: {model_name}") from e
    
    def _validate_config(
        self,
        model_name: str,
        config: Optional[Union[Dict, ImputerConfig]]
    ) -> Dict[str, Any]:
        """Validate and normalize configuration."""
        if config is None:
            config = imputer_parameters()
        
        if isinstance(config, ImputerConfig):
            return config.model_configs
        
        # Validate dict config if config class exists
        if isinstance(config, dict) and model_name in self._CONFIG_MAPPING:
            config_class = self._CONFIG_MAPPING[model_name]
            model_config = config.get(model_name, {})
            validated = config_class(**model_config)
            config[model_name] = validated.model_dump()
        
        return config
    
    def _fit_all_columns(
        self,
        X: pd.DataFrame,
        columns: List[str],
        missing_report: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Fit imputers for all columns efficiently."""
        fitted = {}
        
        for idx, column in enumerate(
            tqdm(columns, desc="Fitting", disable=not self._should_show_progress())
        ):
            context = self._create_context(X, column, idx, missing_report)
            fitted[column] = self._create_column_imputer(X, column, context)
        
        return fitted
    
    def _create_context(
        self,
        X: pd.DataFrame,
        column: str,
        order: int,
        report: pd.DataFrame
    ) -> ImputationContext:
        """Create imputation context for a column."""
        col_info = report[
            report[ImputationColumn.COLUMN_NAME.value] == column
        ].iloc[0]
        
        null_mask = X[column].isnull()
        
        return ImputationContext(
            column_name=column,
            missing_count=int(col_info[ImputationColumn.NULL_COUNT.value]),
            missing_percentage=float(col_info[ImputationColumn.NULL_PERCENTAGE.value]),
            column_dtype=str(X[column].dtype),
            imputation_order=order,
            train_indices=X.index[~null_mask].tolist(),
            test_indices=X.index[null_mask].tolist()
        )
    
    def _create_column_imputer(
        self,
        X: pd.DataFrame,
        column: str,
        context: ImputationContext
    ) -> Dict[str, Any]:
        """Create and fit imputer for a single column."""
        # Extract training data
        train_data = X.iloc[context.train_indices].copy()
        
        # Setup preprocessing pipeline
        encoder = self._setup_encoder(train_data, column)
        if encoder:
            train_data = encoder.transform(train_data)
        
        # Apply simple imputation to features
        simple_imputer = AutoSimpleImputer(strategy=ImputerStrategy.MEAN.value)
        simple_imputer.fit(train_data)
        train_data = simple_imputer.transform(train_data)
        
        # Prepare features and target
        feature_cols = [c for c in train_data.columns if c != column]
        X_train = train_data[feature_cols].values
        y_train = train_data[column].values
        
        # Create and fit model
        model = self._create_model()
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'encoder': encoder,
            'simple_imputer': simple_imputer,
            'feature_columns': feature_cols,
            'context': context
        }
    
    def _apply_column_imputation(
        self,
        X: pd.DataFrame,
        column: str,
        config: Dict[str, Any]
    ) -> None:
        """Apply imputation to a single column in-place."""
        # Get rows needing imputation
        null_mask = X[column].isnull()
        if not null_mask.any():
            return
        
        # Extract and preprocess test data
        test_data = X[null_mask].copy()
        
        if config['encoder']:
            test_data = config['encoder'].transform(test_data)
        
        test_data = config['simple_imputer'].transform(test_data)
        
        # Predict missing values
        X_test = test_data[config['feature_columns']].values
        predictions = config['model'].predict(X_test)
        
        # Update original DataFrame
        X.loc[null_mask, column] = predictions
    
    def _setup_encoder(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Optional[AutoLabelEncoder]:
        """Setup encoder for categorical columns if needed."""
        cat_cols = self._get_categorical_columns(data, exclude=target_column)
        
        if not cat_cols:
            return None
        
        encoder = AutoLabelEncoder()
        encoder.fit(data[cat_cols])
        return encoder
    
    def _create_model(self) -> Any:
        """Create imputation model instance."""
        model_config = self._config.get(self._model_name, {})
        return ImputerFactory.create(self._strategy, model_config)
    
    def _should_show_progress(self) -> bool:
        """Determine if progress bar should be shown."""
        # Could be based on config or environment
        return True
    
    # Public API methods
    
    def get_state(self) -> ImputationState:
        """Get current imputation state."""
        if not self.is_fitted:
            raise ModelNotFittedError("Imputer not fitted")
        return self._state
    
    def set_state(self, state: ImputationState) -> None:
        """Restore imputation state."""
        self._state = state
        self._strategy = state.strategy
        self._fitted_configs = state.fitted_models
    
    def get_summary(self) -> Dict[str, Any]:
        """Get imputation summary."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        return {
            'status': 'fitted',
            'strategy': self._strategy.value,
            'n_columns_imputed': len(self._state.column_order),
            'columns': self._state.column_order,
            'fit_timestamp': self._state.fit_timestamp.isoformat(),
            'model': self._model_name
        }