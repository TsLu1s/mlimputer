import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from mlimputer.utils.constants import (
    ProblemType,
    MetricsComplement,
    BaseValidationConfig,
    EvaluationResult,
)
from mlimputer.evaluation.metrics import MetricsCalculator
from mlimputer.utils.logging import logger


class AbstractValidator(ABC):
    """Abstract base class for all validators."""

    def __init__(self, config: Optional[BaseValidationConfig] = None):
        """Initialize validator."""
        self.config = config or BaseValidationConfig()
        self.results: List[EvaluationResult] = []
        self.leaderboard: Optional[pd.DataFrame] = None
        self._problem_type: Optional[ProblemType] = None
        self._is_validated: bool = False

    @abstractmethod
    def validate(self, X: pd.DataFrame, target: str, models: List[BaseEstimator], **kwargs) -> Any:
        """Validate models."""
        pass

    @abstractmethod
    def _create_splits(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> List[tuple]:
        """Create validation splits."""
        pass

    @abstractmethod
    def _evaluate_model(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold: Optional[int] = None,
    ) -> EvaluationResult:
        """Evaluate a single model."""
        pass

    def get_leaderboard(self) -> pd.DataFrame:
        """Get validation leaderboard."""
        if self.leaderboard is None:
            raise ValueError("No results available. Run validation first.")
        return self.leaderboard.copy()

    def _detect_problem_type(self, y: pd.Series) -> ProblemType:
        """Detect problem type from target variable."""
        categorical_dtypes = {"object", "category"}

        if y.dtype in categorical_dtypes:
            n_classes = y.nunique()
            if n_classes == 2:
                return ProblemType.BINARY_CLASSIFICATION
            else:
                return ProblemType.MULTICLASS_CLASSIFICATION
        else:
            return ProblemType.REGRESSION

    def _validate_inputs(self, X, target, models, y=None):
        """Validate inputs for evaluation."""
        if X is None or X.empty:
            raise ValueError("Input DataFrame is empty")

        # Only require target in X if y is not provided separately
        if y is None and target not in X.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")

        if not models:
            raise ValueError("No models provided for validation")

    def _calculate_mean_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate mean metrics across results."""
        if not results:
            return {}

        metrics_dict = {}
        # Get all unique metric keys
        all_keys = set()
        for r in results:
            all_keys.update(r.metrics.keys())

        for key in all_keys:
            values = [r.metrics.get(key, np.nan) for r in results]
            values = [v for v in values if not np.isnan(v)]
            metrics_dict[key] = np.mean(values) if values else 0.0

        return metrics_dict

    def _calculate_std_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate standard deviation of metrics across results."""
        if not results:
            return {}

        metrics_dict = {}
        all_keys = set()
        for r in results:
            all_keys.update(r.metrics.keys())

        for key in all_keys:
            values = [r.metrics.get(key, np.nan) for r in results]
            values = [v for v in values if not np.isnan(v)]
            metrics_dict[key] = np.std(values) if values else 0.0

        return metrics_dict

    def _get_best_fold(self, results: List[EvaluationResult]) -> int:
        """Get the fold number with best performance."""
        return self._get_extreme_fold(results, get_best=True)

    def _get_worst_fold(self, results: List[EvaluationResult]) -> int:
        """Get the fold number with worst performance."""
        return self._get_extreme_fold(results, get_best=False)

    def _get_extreme_fold(self, results: List[EvaluationResult], get_best: bool = True) -> int:
        """
        Get the fold number with best or worst performance.
        """
        if not results:
            return 0

        primary_metric = MetricsCalculator.get_primary_metric(self._problem_type)
        direction = MetricsCalculator.get_metric_direction(primary_metric)

        # Map full metric name to abbreviated key
        metric_mapping = {
            "Mean Absolute Error": "mae",
            "MAE": "mae",
            "F1 Score": "f1",
            "F1": "f1",
            "Accuracy": "accuracy",
        }
        abbreviated_metric = metric_mapping.get(primary_metric, primary_metric.lower())

        # Get metric values with appropriate defaults
        default_value = float("inf") if direction == MetricsComplement.MINIMIZE.value else 0
        values = [r.metrics.get(abbreviated_metric, default_value) for r in results]

        # Determine if it is need minimum or maximum
        is_minimize = direction == MetricsComplement.MINIMIZE.value

        # XOR logic: get_best XOR is_minimize determines if it is wanted min or max
        want_minimum = get_best == is_minimize

        extreme_idx = np.argmin(values) if want_minimum else np.argmax(values)

        return results[extreme_idx].fold if results[extreme_idx].fold else 0

    def reset(self) -> None:
        """Reset validator state."""
        self.results = []
        self.leaderboard = None
        self._problem_type = None
        self._is_validated = False
        logger.debug("Validator state reset")
