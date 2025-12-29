from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class ImputationColumn(str, Enum):
    """Column names for imputation reports."""
    COLUMN_NAME = "columns"
    NULL_COUNT = "null_count"
    NULL_PERCENTAGE = "null_percentage"
    
class ImputerStrategy(str, Enum):
    """Simple imputation strategies."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    CONSTANT = "constant"

class DataType(str, Enum):
    """Data type categories."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"

class ProblemType(str, Enum):
    """Enumeration of machine learning problem types."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"

class RegressionMetric(str, Enum):
    """Regression metric names."""
    MAE = "Mean Absolute Error"
    MAPE = "Mean Absolute Percentage Error"
    RMSE = "Root Mean Squared Error"
    EXPLAINED_VARIANCE = "Explained Variance Score"
    MAX_ERROR = "Max Error"
    R2 = "R2 Score"


class ClassificationMetric(str, Enum):
    """Classification metric names."""
    ACCURACY = "Accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"
    F1 = "F1"
    # Additional metrics that could be added
    # ROC_AUC = "ROC AUC"
    # LOG_LOSS = "Log Loss"
    # CONFUSION_MATRIX = "Confusion Matrix"

class MetricsComplement(str, Enum):
    """Classification metric names."""
    BINARY = "binary"
    WEIGHTED = "weighted"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class ResultColumn(str, Enum):
    MODEL = "Model"
    IMPUTER_MODEL = "Imputer Model"
    FOLD = "Fold"
    TRAIN_SAMPLES = "Train Samples"
    VALIDATION_SAMPLES = "Validation Samples"
    TRAINING_TIME = "Training Time"
    PREDICTION_TIME = "Prediction Time"
    AGGREGATE = "Aggregate"

class MetricAggregation(str, Enum):
    MEAN = "Mean"
    STD = "Std"
    MIN = "Min"
    MAX = "Max"

class ValidationStrategy(str, Enum):
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES = "time_series"
    
class FoldStrategy(str, Enum):
    """Cross-validation strategies."""
    KFOLD = "kfold"
    STRATIFIED = "stratified"

class ImputationStrategy(str, Enum):
    """Enumeration of available imputation strategies."""
    RANDOM_FOREST = "RandomForest"
    EXTRA_TREES = "ExtraTrees"
    GBR = "GradientBoosting"
    KNN = "KNN"
    XGBOOST = "XGBoost"
    CATBOOST = "CatBoost"

@dataclass
class ImputationContext:
    """Context information for imputation process."""
    column_name: str
    missing_count: int
    missing_percentage: float
    column_dtype: str
    imputation_order: int
    train_indices: list
    test_indices: list
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BaseValidationConfig:
    """Base configuration for validation."""
    random_state: int = 42
    verbose: int = 1
    strategy: ValidationStrategy = ValidationStrategy.CROSS_VALIDATION

@dataclass
class CrossValidationConfig(BaseValidationConfig):
    """Configuration for cross-validation."""
    n_splits: int = 5
    shuffle: bool = True
    fold_strategy: FoldStrategy = FoldStrategy.KFOLD
    
    def __post_init__(self):
        """Set strategy after initialization."""
        self.strategy = ValidationStrategy.CROSS_VALIDATION


@dataclass
class ImputationState:
    """Immutable state object for imputation process."""
    strategy: ImputationStrategy
    column_order: List[str] = field(default_factory=list)
    fitted_models: Dict[str, Any] = field(default_factory=dict)
    fit_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted_models)
    
##################################################################################

class MetricType(str, Enum):
    """Types of evaluation metrics."""
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    MAPE = "mape"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"


class EvaluationResult(BaseModel):
    """Model for evaluation results."""
    model_config = ConfigDict(use_enum_values=True)
    model_name: str
    metrics: Dict[MetricType, float]
    fold: Optional[int] = None
    train_samples: int
    validation_samples: int
    training_time: float
    prediction_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class CrossValidationResult(BaseModel):
    """Model for cross-validation results."""
    model_name: str
    strategy: str
    n_folds: int
    fold_results: List[EvaluationResult]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_fold: int
    worst_fold: int
    total_time: float


@dataclass
class PipelineStep:
    """Represents a single step in the pipeline."""
    name: str
    operation: str
    component: Any
    params: Dict[str, Any] = field(default_factory=dict)
    is_fitted: bool = False
