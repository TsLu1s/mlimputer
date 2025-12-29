from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any, Optional, Union

class BaseConfig(BaseModel):
    """Base configuration model."""
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True
    )


class RandomForestConfig(BaseConfig):
    """Configuration for Random Forest imputation."""
    n_estimators: int = Field(default=100, ge=1, le=1000)
    random_state: Optional[int] = Field(default=42)
    criterion: str = Field(default="squared_error")
    max_depth: Optional[int] = Field(default=None, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: Union[str, int, float] = Field(default="sqrt")
    
    @field_validator('max_features')
    def validate_max_features(cls, v):
        if isinstance(v, str) and v not in ['sqrt', 'log2', 'auto']:
            raise ValueError("String max_features must be 'sqrt', 'log2', or 'auto'")
        return v


class ExtraTreesConfig(BaseConfig):
    """Configuration for Extra Trees imputation."""
    n_estimators: int = Field(default=100, ge=1, le=1000)
    random_state: Optional[int] = Field(default=42)
    criterion: str = Field(default="squared_error")
    max_depth: Optional[int] = Field(default=None, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: Union[str, int, float] = Field(default="sqrt")


class GradientBoostingConfig(BaseConfig):
    """Configuration for Gradient Boosting imputation."""
    n_estimators: int = Field(default=100, ge=1, le=1000)
    criterion: str = Field(default="friedman_mse")
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    max_depth: int = Field(default=3, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    loss: str = Field(default="squared_error")


class KNNConfig(BaseConfig):
    """Configuration for KNN imputation."""
    n_neighbors: int = Field(default=5, ge=1)
    weights: str = Field(default="uniform")
    algorithm: str = Field(default="auto")
    leaf_size: int = Field(default=30, ge=1)
    p: int = Field(default=2, ge=1)
    
    @field_validator('weights')
    def validate_weights(cls, v):
        if v not in ['uniform', 'distance']:
            raise ValueError("weights must be 'uniform' or 'distance'")
        return v
    
    @field_validator('algorithm')
    def validate_algorithm(cls, v):
        if v not in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            raise ValueError("Invalid algorithm")
        return v


class XGBoostConfig(BaseConfig):
    """Configuration for XGBoost imputation."""
    n_estimators: int = Field(default=100, ge=1)
    objective: str = Field(default="reg:squarederror")
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    max_depth: int = Field(default=3, ge=1)
    reg_lambda: float = Field(default=1.0, ge=0)
    reg_alpha: float = Field(default=0.0, ge=0)
    subsample: float = Field(default=1.0, gt=0, le=1)
    colsample_bytree: float = Field(default=1.0, gt=0, le=1)

class CatBoostConfig(BaseConfig):
    """Configuration for CatBoost imputation."""
    iterations: int = Field(default=100, ge=1)
    loss_function: str = Field(default="RMSE")
    depth: int = Field(default=8, ge=1, le=16)
    learning_rate: float = Field(default=0.1, gt=0)
    l2_leaf_reg: float = Field(default=3.0, ge=0)
    border_count: int = Field(default=254, ge=1)
    subsample: float = Field(default=1.0, gt=0, le=1)


class ImputerConfig(BaseConfig):
    """Main imputer configuration."""
    strategy: str
    model_configs: Dict[str, Union[
        RandomForestConfig,
        ExtraTreesConfig,
        GradientBoostingConfig,
        KNNConfig,
        XGBoostConfig,
        CatBoostConfig
    ]] = Field(default_factory=dict)
    preprocessing_config: Optional[Dict[str, Any]] = None
    evaluation_config: Optional[Dict[str, Any]] = None
    parallel_execution: bool = Field(default=False)
    n_jobs: int = Field(default=1, ge=1)
    verbose: int = Field(default=1, ge=0, le=2)
    random_state: Optional[int] = Field(default=42)