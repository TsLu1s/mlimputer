import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor

from mlimputer.core.base import BaseImputationModel

class XGBoostImputation(BaseImputationModel):
    """XGBoost imputation model."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = 'reg:squarederror',
        learning_rate: float = 0.1,
        max_depth: int = 3,
        reg_lambda: float = 1,
        reg_alpha: float = 0,
        subsample: float = 1,
        colsample_bytree: float = 1,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            learning_rate=learning_rate,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **kwargs
        )
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize XGBoost model."""
        params = self.model_params.copy()
        params['verbosity'] = 0
        self.model = xgb.XGBRegressor(**params)

class CatBoostImputation(BaseImputationModel):
    """CatBoost imputation model."""
    
    def __init__(
        self,
        iterations: int = 100,
        loss_function: str = 'RMSE',
        depth: int = 8,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3,
        border_count: int = 254,
        subsample: float = 1,
        **kwargs
    ):
        super().__init__(
            iterations=iterations,
            loss_function=loss_function,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
            subsample=subsample,
            **kwargs
        )
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize CatBoost model."""
        params = self.model_params.copy()
        params['save_snapshot'] = False
        params['verbose'] = False
        self.model = CatBoostRegressor(**params)
