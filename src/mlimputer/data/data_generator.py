import numpy as np
import pandas as pd
from typing import Tuple, Literal, Dict, Any, List
from sklearn.datasets import make_classification, make_regression

from mlimputer.data.base_generator import BaseDatasetGenerator


class ImputationDatasetGenerator(BaseDatasetGenerator):
    """Generate synthetic datasets for imputation testing."""

    TASKS = ["regression", "binary_classification", "multiclass_classification"]
    PATTERNS = ["random", "column_wise", "correlated"]

    def __init__(self, random_state: int = 42):
        """Initialize generator with random state."""
        self.random_state = random_state
        np.random.seed(random_state)

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default generation parameters."""
        return {
            "n_samples": 1000,
            "n_features": 20,
            "n_classes": 3,
            "missing_rate": 0.15,
            "missing_pattern": "random",
            "n_categorical": 0,
            "task": "regression",
        }

    def generate(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        task: Literal[
            "regression", "binary_classification", "multiclass_classification"
        ] = "regression",
        n_classes: int = 3,
        missing_rate: float = 0.15,
        missing_pattern: Literal["random", "column_wise", "correlated"] = "random",
        n_categorical: int = 0,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate dataset with missing values for imputation testing."""
        # Validate parameters
        params = self._validate_params(
            n_samples=n_samples,
            n_features=n_features,
            task=task,
            n_classes=n_classes,
            missing_rate=missing_rate,
            missing_pattern=missing_pattern,
            n_categorical=n_categorical,
            **kwargs,
        )

        # Generate base dataset
        X, y = self._create_base_dataset(params)

        # Convert to DataFrame
        X = self._to_dataframe(X, params["n_features"])

        # Add categorical features
        if params["n_categorical"] > 0:
            X = self._add_categorical_features(X, params)

        # Apply missing pattern
        X = self._apply_missingness(X, params)

        # Format target
        y = self._format_target(y, params["task"])

        return X, y

    def _create_base_dataset(self, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Create base numeric dataset based on task type."""
        task = params["task"]

        if task == "regression":
            return self._make_regression(params)
        elif task == "binary_classification":
            return self._make_binary_classification(params)
        elif task == "multiclass_classification":
            return self._make_multiclass_classification(params)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _make_regression(self, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regression dataset."""
        return make_regression(
            n_samples=params["n_samples"],
            n_features=params["n_features"],
            n_informative=int(params["n_features"] * 0.8),
            noise=10.0,
            random_state=self.random_state,
        )

    def _make_binary_classification(self, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate binary classification dataset."""
        return make_classification(
            n_samples=params["n_samples"],
            n_features=params["n_features"],
            n_informative=int(params["n_features"] * 0.7),
            n_redundant=int(params["n_features"] * 0.2),
            n_classes=2,
            n_clusters_per_class=2,
            flip_y=0.01,
            class_sep=1.0,
            random_state=self.random_state,
        )

    def _make_multiclass_classification(
        self, params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multiclass classification dataset."""
        n_classes = max(3, params["n_classes"])
        return make_classification(
            n_samples=params["n_samples"],
            n_features=params["n_features"],
            n_informative=int(params["n_features"] * 0.6),
            n_redundant=int(params["n_features"] * 0.2),
            n_classes=n_classes,
            n_clusters_per_class=2,
            flip_y=0.01,
            class_sep=0.8,
            random_state=self.random_state,
        )

    def _to_dataframe(self, X: np.ndarray, n_features: int) -> pd.DataFrame:
        """Convert numpy array to DataFrame with feature names."""
        feature_names = [f"num_{i}" for i in range(n_features)]
        return pd.DataFrame(X, columns=feature_names)

    def _add_categorical_features(self, X: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Add categorical columns to dataset."""
        n_categorical = params["n_categorical"]
        n_samples = len(X)

        for i in range(n_categorical):
            categories = self._generate_categories(i)
            X[f"cat_{i}"] = np.random.choice(categories, n_samples)

        return X

    def _generate_categories(self, index: int) -> List[str]:
        """Generate category labels based on index."""
        if index == 0:
            return [f"level_{j}" for j in range(5)]
        elif index == 1:
            return [f"cat_{chr(65+j)}" for j in range(5)]
        else:
            return [f"type_{j}{chr(65+j%26)}" for j in range(5)]

    def _apply_missingness(self, X: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply missing value pattern to dataset."""
        pattern = params["missing_pattern"]
        rate = params["missing_rate"]

        if pattern == "random":
            return self._apply_random_missing(X, rate)
        elif pattern == "column_wise":
            return self._apply_columnwise_missing(X, rate)
        elif pattern == "correlated":
            return self._apply_correlated_missing(X, rate)
        else:
            raise ValueError(f"Unknown missing pattern: {pattern}")

    def _apply_random_missing(self, X: pd.DataFrame, rate: float) -> pd.DataFrame:
        """Apply random missing pattern."""
        numeric_cols = self._get_numeric_columns(X)

        for col in numeric_cols:
            mask = np.random.random(len(X)) < rate
            X.loc[mask, col] = np.nan

        return X

    def _apply_columnwise_missing(self, X: pd.DataFrame, rate: float) -> pd.DataFrame:
        """Apply column-wise varying missing pattern."""
        numeric_cols = self._get_numeric_columns(X)

        for i, col in enumerate(numeric_cols):
            col_rate = self._calculate_column_rate(rate, i)
            mask = np.random.random(len(X)) < col_rate
            X.loc[mask, col] = np.nan

        return X

    def _apply_correlated_missing(self, X: pd.DataFrame, rate: float) -> pd.DataFrame:
        """Apply correlated missing pattern."""
        numeric_cols = self._get_numeric_columns(X)

        for col in numeric_cols:
            threshold = X[col].quantile(0.3)
            mask = (X[col] < threshold) & (np.random.random(len(X)) < rate * 2)
            X.loc[mask, col] = np.nan

        return X

    def _get_numeric_columns(self, X: pd.DataFrame) -> pd.Index:
        """Get numeric column names from DataFrame."""
        return X.select_dtypes(include=[np.number]).columns

    def _calculate_column_rate(self, base_rate: float, column_index: int) -> float:
        """Calculate column-specific missing rate."""
        return base_rate * (0.5 + (column_index % 3) * 0.3)

    def _format_target(self, y: np.ndarray, task: str) -> pd.Series:
        """Format target variable based on task type."""
        if task in ["binary_classification", "multiclass_classification"]:
            return pd.Series(y, name="target").astype("object")
        return pd.Series(y, name="target")

    def quick_regression(self, n_samples: int = 1000, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate regression dataset with default settings."""
        return self.generate(task="regression", n_samples=n_samples, **kwargs)

    def quick_binary(self, n_samples: int = 1000, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate binary classification dataset."""
        return self.generate(task="binary_classification", n_samples=n_samples, **kwargs)

    def quick_multiclass(
        self, n_samples: int = 1000, n_classes: int = 3, **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate multiclass dataset."""
        return self.generate(
            task="multiclass_classification", n_samples=n_samples, n_classes=n_classes, **kwargs
        )

    def create_benchmark(self) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Generate comprehensive benchmark datasets."""
        benchmark = {}

        # Regression datasets
        benchmark["small_reg"] = self.quick_regression(500, missing_rate=0.1)
        benchmark["medium_reg"] = self.quick_regression(2000, missing_rate=0.15)
        benchmark["large_reg"] = self.quick_regression(5000, missing_rate=0.2)

        # Binary classification
        benchmark["small_binary"] = self.quick_binary(500, missing_rate=0.1)
        benchmark["medium_binary"] = self.quick_binary(2000, missing_rate=0.15)

        # Multiclass classification
        benchmark["small_multi"] = self.quick_multiclass(500, n_classes=3, missing_rate=0.1)
        benchmark["medium_multi"] = self.quick_multiclass(2000, n_classes=5, missing_rate=0.15)
        benchmark["large_multi"] = self.quick_multiclass(5000, n_classes=4, missing_rate=0.2)

        # Special patterns
        benchmark["correlated_reg"] = self.generate(
            task="regression", n_samples=1000, missing_pattern="correlated", missing_rate=0.25
        )
        benchmark["columnwise_multi"] = self.generate(
            task="multiclass_classification",
            n_samples=1000,
            n_classes=3,
            missing_pattern="column_wise",
            missing_rate=0.2,
        )

        return benchmark
