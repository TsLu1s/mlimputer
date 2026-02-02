import numpy as np
import pandas as pd

from typing import List, Optional, Literal, Union
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator

from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
import time

from mlimputer.evaluation.abstract import AbstractValidator
from mlimputer.utils.constants import (
    ProblemType,
    MetricsComplement,
    EvaluationResult,
    CrossValidationResult,
    CrossValidationConfig,
    FoldStrategy,
)
from mlimputer.evaluation.metrics import MetricsCalculator
from mlimputer.utils.logging import logger

import warnings


class CrossValidator(AbstractValidator):
    """Cross-validation implementation"""

    def __init__(self, config: Optional[CrossValidationConfig] = None):
        """
        Initialize cross validator.
        """
        config = config or CrossValidationConfig()
        super().__init__(config)
        self.config: CrossValidationConfig = config

    def validate(
        self,
        X: pd.DataFrame,
        target: str,
        models: List[BaseEstimator],
        problem_type: Optional[
            Union[
                ProblemType,
                Literal["regression", "binary_classification", "multiclass_classification"],
            ]
        ] = None,
        y: Optional[pd.Series] = None,
    ) -> CrossValidationResult:
        """
        Performs cross-validation.
        """
        # Validate inputs
        self._validate_inputs(X, target, models, y)

        # Prepare features and target, ensuring no contamination
        X_features, y = self._prepare_data(X, target, y)

        # Convert string to ProblemType enum if necessary
        if problem_type is not None:
            if isinstance(problem_type, str):
                self._problem_type = ProblemType(problem_type)
            else:
                self._problem_type = problem_type
        else:
            self._problem_type = self._detect_problem_type(y)

        # Create splits based on problem type
        splits = self._create_splits(X_features, y)

        results_list = []

        # Suppress convergence warnings if needed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                if self.config.verbose > 0:
                    logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_splits}")

                X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                for model in models:
                    if self.config.verbose > 1:
                        logger.debug(f"Training {model.__class__.__name__}...")

                    try:
                        result = self._evaluate_model(
                            model=model,
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            fold=fold_idx + 1,
                        )
                        results_list.append(result)
                    except Exception as e:
                        logger.error(f"Error training {model.__class__.__name__}: {str(e)}")
                        continue

        self.results = results_list
        self.leaderboard = self._create_leaderboard(results_list)
        self._is_validated = True

        return self._create_cv_result(models, results_list)

    def _detect_problem_type(self, y: pd.Series) -> ProblemType:
        """
        Detect problem type from target variable.
        """
        n_unique = y.nunique()

        if y.dtype in ["float64", "float32"] or n_unique > 20:
            return ProblemType.REGRESSION
        elif n_unique == 2:
            return ProblemType.BINARY_CLASSIFICATION
        else:
            return ProblemType.MULTICLASS_CLASSIFICATION

    def _create_splits(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> List[tuple]:
        """
        Create cross-validation splits.
        """
        # Use stratified for classification, regular KFold for regression
        if (
            self._problem_type == ProblemType.REGRESSION
            or self.config.fold_strategy == FoldStrategy.KFOLD
        ):
            splitter = KFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            return list(splitter.split(X))
        else:
            # Stratified for both binary and multiclass
            splitter = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            return list(splitter.split(X, y))

    # Alias for backward compatibility
    def cross_validate(self, *args, **kwargs):
        """Alias for validate method (backward compatibility)."""
        return self.validate(*args, **kwargs)

    def _create_splits(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> List[tuple]:
        """
        Create cross-validation splits.
        """
        if (
            self._problem_type == ProblemType.REGRESSION
            or self.config.fold_strategy == FoldStrategy.KFOLD
        ):
            splitter = KFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            return list(splitter.split(X))
        else:
            splitter = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            return list(splitter.split(X, y))

    def _evaluate_model(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold: Optional[int] = None,
    ) -> EvaluationResult:
        """Evaluate a single model on a train/validation split."""
        model_name = model.__class__.__name__

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_val)
        prediction_time = time.time() - start_time

        # Calculate metrics based on problem type
        if self._problem_type == ProblemType.REGRESSION:
            metrics_result = MetricsCalculator.calculate_regression_metrics(y_val, y_pred)
        elif self._problem_type == ProblemType.BINARY_CLASSIFICATION:
            metrics_result = MetricsCalculator.calculate_binary_metrics(y_val, y_pred)
        else:  # MULTICLASS_CLASSIFICATION
            metrics_result = MetricsCalculator.calculate_multiclass_metrics(y_val, y_pred)

        # Map full metric names to abbreviated keys
        metric_mapping = {
            "Mean Absolute Error": "mae",
            "Mean Squared Error": "mse",
            "Root Mean Squared Error": "rmse",
            "R2 Score": "r2",
            "Mean Absolute Percentage Error": "mape",
            "Explained Variance Score": "explained_variance",
            "Max Error": "max_error",
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1 Score": "f1",
            "F1": "f1",
            "AUC": "auc",
        }

        # Convert metrics to use abbreviated keys
        abbreviated_metrics = {}
        for key, value in metrics_result.metrics.items():
            abbreviated_key = metric_mapping.get(key, key.lower())
            # Only include metrics that are in the expected schema
            if abbreviated_key in [
                "mae",
                "mse",
                "rmse",
                "r2",
                "mape",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]:
                abbreviated_metrics[abbreviated_key] = value

        return EvaluationResult(
            model_name=model_name,
            metrics=abbreviated_metrics,
            fold=fold,
            train_samples=len(X_train),
            validation_samples=len(X_val),
            training_time=training_time,
            prediction_time=prediction_time,
        )

    def _create_leaderboard(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Create leaderboard DataFrame from results."""
        rows = []
        for result in results:
            row = {
                "Model": result.model_name,
                "Fold": result.fold,
                "Train Samples": result.train_samples,
                "Validation Samples": result.validation_samples,
                "Training Time": result.training_time,
                "Prediction Time": result.prediction_time,
            }
            row.update(result.metrics)
            rows.append(row)

        leaderboard = pd.DataFrame(rows)

        if not leaderboard.empty:
            # Add aggregate statistics (now returns modified DataFrame)
            leaderboard = self._add_aggregate_metrics(leaderboard)

            # Get primary metric and map to abbreviated form
            primary_metric = MetricsCalculator.get_primary_metric(self._problem_type)

            # Mapping for primary metrics to abbreviated form
            metric_mapping = {
                "Mean Absolute Error": "mae",
                "MAE": "mae",
                "F1 Score": "f1",
                "F1": "f1",
                "Accuracy": "accuracy",
            }

            # Get the abbreviated column name
            abbreviated_metric = metric_mapping.get(primary_metric, primary_metric.lower())

            # Check if the column exists in the DataFrame
            if abbreviated_metric in leaderboard.columns:
                direction = MetricsCalculator.get_metric_direction(primary_metric)
                ascending = direction == MetricsComplement.MINIMIZE.value

                leaderboard = leaderboard.sort_values(
                    by=["Fold", abbreviated_metric], ascending=[True, ascending]
                )

        return leaderboard

    def _add_aggregate_metrics(self, leaderboard: pd.DataFrame) -> pd.DataFrame:
        """Add aggregate metrics to leaderboard and return modified DataFrame."""
        metric_columns = leaderboard.select_dtypes(include=[np.number]).columns
        metric_columns = metric_columns.drop(
            ["Fold", "Train Samples", "Validation Samples"], errors="ignore"
        )

        aggregated_rows = []
        for model_name in leaderboard["Model"].unique():
            model_results = leaderboard[leaderboard["Model"] == model_name]

            agg_row = {"Model": model_name, "Fold": "Aggregate"}

            for metric in metric_columns:
                # Only create uppercase versions for standard metrics
                display_name = (
                    metric.upper()
                    if metric
                    in ["mae", "mse", "rmse", "r2", "mape", "accuracy", "precision", "recall", "f1"]
                    else metric
                )
                agg_row[f"{display_name} Mean"] = model_results[metric].mean()
                if self.config.verbose > 1:
                    agg_row[f"{display_name} Std"] = model_results[metric].std()

            aggregated_rows.append(agg_row)

        agg_df = pd.DataFrame(aggregated_rows)
        return pd.concat([leaderboard, agg_df], ignore_index=True)

    def _create_cv_result(
        self, models: List[BaseEstimator], results: List[EvaluationResult]
    ) -> CrossValidationResult:
        """Create CrossValidationResult object."""
        return CrossValidationResult(
            model_name=models[0].__class__.__name__ if models else "Unknown",
            strategy=self.config.strategy.value,
            n_folds=self.config.n_splits,
            fold_results=results,
            mean_metrics=self._calculate_mean_metrics(results),
            std_metrics=self._calculate_std_metrics(results),
            best_fold=self._get_best_fold(results),
            worst_fold=self._get_worst_fold(results),
            total_time=sum(r.training_time + r.prediction_time for r in results),
        )

    def _prepare_data(
        self, X: pd.DataFrame, target: str, y: Optional[pd.Series]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target, ensuring target column is removed from features.
        """
        if y is None:
            if target not in X.columns:
                raise ValueError(f"Target column '{target}' must be in X when y is not provided")
            y = X[target].copy()
            X_features = X.drop(columns=[target])
        else:
            # Even if y is provided, remove target from X if it exists (prevent contamination)
            if target in X.columns:
                X_features = X.drop(columns=[target])
            else:
                X_features = X.copy()

        # Final safety check: ensure target is absolutely not in features
        if target in X_features.columns:
            X_features = X_features.drop(columns=[target])
            logger.warning(
                f"Target column '{target}' was found in features and removed to prevent contamination"
            )

        if y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)

        return X_features, y


# Backward compatibility function
def cross_validation(
    X: pd.DataFrame, target: str, n_splits: int = 5, models: List[BaseEstimator] = []
) -> pd.DataFrame:
    """
    Legacy cross-validation function for backward compatibility.
    """
    config = CrossValidationConfig(n_splits=n_splits)
    validator = CrossValidator(config)
    validator.validate(X, target, models)  # Returns CrossValidationResult
    return validator.get_leaderboard()  # Return DataFrame for backward compatibility
