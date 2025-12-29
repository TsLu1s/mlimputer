import numpy as np
import pandas as pd

from typing import Dict, Optional, Union
from dataclasses import dataclass

from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

from mlimputer.utils.constants import ProblemType, RegressionMetric, ClassificationMetric, MetricsComplement
from mlimputer.utils.logging import logger

@dataclass
class MetricsResult:
    """Container for metrics results."""
    problem_type: Optional[ProblemType]
    metrics: Dict[str, float]
    metadata: Optional[Dict[str, any]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame([self.metrics])
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            'problem_type': self.problem_type.value,
            'metrics': self.metrics,
            'metadata': self.metadata
        }


class MetricsCalculator:
    """Calculate metrics for different problem types."""
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> MetricsResult:
        """Calculate regression metrics with abbreviated keys."""
        return MetricsResult(
            metrics={
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mape': mean_absolute_percentage_error(y_true, y_pred)
            },
            problem_type=ProblemType.REGRESSION
        )
    
    @staticmethod
    def calculate_binary_metrics(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> MetricsResult:
        """Calculate classification metrics."""
        return MetricsResult(
        metrics={
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label=1),
            'recall': recall_score(y_true, y_pred, pos_label=1),
            'f1': f1_score(y_true, y_pred, pos_label=1),
        },
        problem_type=ProblemType.BINARY_CLASSIFICATION 
    )

    @staticmethod
    def calculate_multiclass_metrics(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> MetricsResult:
        """Calculate classification metrics."""
        return MetricsResult(
        metrics={
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        },
        problem_type=ProblemType.MULTICLASS_CLASSIFICATION
    )
    
    @staticmethod
    def get_primary_metric(problem_type: ProblemType) -> str:
        """Get the primary metric for a problem type."""
        if problem_type in [ProblemType.REGRESSION]:
            return RegressionMetric.MAE.value
        else:
            return ClassificationMetric.F1.value
    
    @staticmethod
    def get_metric_direction(metric_name: str) -> str:
        """Determine if higher or lower values are better for a metric."""
        minimize_metrics = [
            RegressionMetric.MAE.value,
            RegressionMetric.MAPE.value,
            RegressionMetric.RMSE.value,
            RegressionMetric.MAX_ERROR.value,
        ]
        
        if metric_name in minimize_metrics:
            return MetricsComplement.MINIMIZE.value
        else:
            return MetricsComplement.MAXIMIZE.value



