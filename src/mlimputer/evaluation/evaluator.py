import pandas as pd

from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from mlimputer.imputation import MLimputer
from mlimputer.schemas.parameters import imputer_parameters
from mlimputer.preprocessing.encoders import AutoLabelEncoder
from mlimputer.evaluation.metrics import MetricsCalculator
from mlimputer.evaluation.cross_validation import CrossValidator
from mlimputer.utils.constants import ProblemType, ResultColumn, MetricAggregation, CrossValidationConfig
from mlimputer.utils.logging import logger
from mlimputer.utils.constants import ProblemType

@dataclass
class EvaluatorConfig:
    """Configuration for model evaluator."""
    imputation_models: List[str] = field(default_factory=list)
    n_splits: int = 3
    verbose: int = 1
    parallel: bool = False
    n_jobs: int = 1


class Evaluator:
    """Advanced evaluator for comparing imputation strategies."""
    
    def __init__(
        self,
        imputation_models: List[str],
        train: pd.DataFrame,
        target: str,
        n_splits: int = 3,
        hparameters: Optional[Dict[str, Dict[str, Any]]] = None,
        problem_type: Optional[Union[ProblemType, Literal['regression', 'binary_classification', 'multiclass_classification']]] = None
    ):
        """Initialize evaluator."""
        self.config = EvaluatorConfig(
            imputation_models=imputation_models,
            n_splits=n_splits
        )
        self.train = train
        self.target = target
        self.hparameters = hparameters or imputer_parameters()
        
        # Detect data types
        self.categorical_dtypes = {"object", "category"}
        self.numeric_types = {'int', 'int32', 'int64', 'float', 'float32', 'float64'}
        
        # Initialize components
        self.label_encoder: Optional[AutoLabelEncoder] = None
        self.cv_validator = CrossValidator(
            CrossValidationConfig(n_splits=n_splits)
        )
        self.leaderboard: Optional[pd.DataFrame] = None
        self.best_imputer: Optional[str] = None
        
        # Handle problem type (string or enum)
        if problem_type is not None:
            # Convert string to ProblemType enum if necessary
            if isinstance(problem_type, str):
                self.problem_type = ProblemType(problem_type)
                logger.info(f"Using user-specified problem type: {self.problem_type.value}")
            else:
                self.problem_type = problem_type
                logger.info(f"Using user-specified problem type: {self.problem_type.value}")
        else:
            self.problem_type = self._detect_problem_type()
            logger.info(f"Auto-detected problem type: {self.problem_type.value}")

        logger.info(
            f"Initialized Evaluator for {self.problem_type.value} "
            f"with {len(imputation_models)} imputation strategies"
        )
    
    def evaluate_imputation_models(
        self,
        models: List[BaseEstimator]
    ) -> pd.DataFrame:
        """Evaluate different imputation models using cross-validation."""
        all_results = []
        
        for imput_model in self.config.imputation_models:
            logger.info(f"Evaluating imputation model: {imput_model}")
            
            # Create imputed dataset
            train_imputed = self._apply_imputation(self.train.copy(), imput_model)
            
            # Apply label encoding if needed
            train_processed = self._apply_encoding(train_imputed)
            
            # Run cross-validation with problem_type
            self.cv_validator.validate(
                X=train_processed,
                target=self.target,
                models=models,
                problem_type=self.problem_type  # Pass the problem_type from Evaluator
            )
            
            # Get leaderboard and add imputation model column
            cv_leaderboard = self.cv_validator.get_leaderboard()
            cv_leaderboard[ResultColumn.IMPUTER_MODEL.value] = imput_model
            all_results.append(cv_leaderboard)
        
        # Combine all results
        self.leaderboard = pd.concat(all_results, ignore_index=True)
        
        # Drop duplicate metric columns (keep uppercase versions)
        cols = self.leaderboard.columns
        duplicates = [col for col in cols if col.lower() != col and col.lower() + ' Mean' in cols.str.lower()]
        self.leaderboard = self.leaderboard.drop(columns=duplicates, errors='ignore')
        
        # Identify best imputer
        self._identify_best_imputer()
        
        return self.leaderboard

    def evaluate_test_set(
        self,
        test: pd.DataFrame,
        imput_model: str,
        models: List[BaseEstimator]
    ) -> pd.DataFrame:
        """Evaluate models on test set with specified imputation."""
        logger.info(f"Evaluating test set with imputation model: {imput_model}")
        
        # Apply imputation to train and test
        train_imputed = self._apply_imputation(self.train.copy(), imput_model)
        test_imputed = self._apply_imputation_transform(
            self.train.copy(),
            test.copy(),
            imput_model
        )
        
        # Apply encoding
        train_processed = self._apply_encoding(train_imputed, fit=True)
        test_processed = self._apply_encoding(test_imputed, fit=False)
        
        # Prepare data
        X_train = train_processed.drop(columns=[self.target])
        y_train = train_processed[self.target]
        X_test = test_processed.drop(columns=[self.target])
        y_test = test_processed[self.target]
        
        # Evaluate models
        results_list = []
        
        for model in models:
            model_name = model.__class__.__name__
            logger.info(f"Testing model: {model_name}")
            
            try:
                # Train and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if self.problem_type == ProblemType.REGRESSION:
                    metrics_result = MetricsCalculator.calculate_regression_metrics(y_test, y_pred)
                elif self.problem_type == ProblemType.BINARY_CLASSIFICATION:
                    metrics_result = MetricsCalculator.calculate_binary_metrics(y_test, y_pred)
                else:  # MULTICLASS_CLASSIFICATION
                    metrics_result = MetricsCalculator.calculate_multiclass_metrics(y_test, y_pred)
                
                # Create result row using enum
                result_row = {
                    ResultColumn.MODEL.value: model_name,
                    ResultColumn.IMPUTER_MODEL.value: imput_model
                }
                result_row.update(metrics_result.metrics)
                results_list.append(result_row)
                
            except Exception as e:
                logger.error(f"Error testing {model_name}: {str(e)}")
                continue
        
        return pd.DataFrame(results_list)
    
    def get_best_imputer(self) -> str:
        """Get the best performing imputation model."""
        if self.best_imputer is None:
            self._identify_best_imputer()
        
        if self.best_imputer is None:
            raise ValueError("No best imputer identified. Run evaluation first.")
        
        return self.best_imputer
    
    def _detect_problem_type(self) -> ProblemType:
        """Detect problem type from target automatically."""
        target_dtype = self.train[self.target].dtype
        n_unique = self.train[self.target].nunique()

        # Handle categorical or object dtypes
        if target_dtype in self.categorical_dtypes:
            return ProblemType.BINARY_CLASSIFICATION if n_unique == 2 else ProblemType.MULTICLASS_CLASSIFICATION

        # Handle numeric dtypes that behave like classification labels
        if pd.api.types.is_integer_dtype(target_dtype) and n_unique <= 10:
            return ProblemType.BINARY_CLASSIFICATION if n_unique == 2 else ProblemType.MULTICLASS_CLASSIFICATION

        # Default to regression
        return ProblemType.REGRESSION
    
    def _apply_imputation(self, data: pd.DataFrame, imput_model: str) -> pd.DataFrame:
        """Apply imputation to dataset (excluding target column)."""
        # Separate features and target to prevent contamination
        X_features = data.drop(columns=[self.target])
        y_target = data[self.target]
        
        # Fit and transform only on features
        mli = MLimputer(imput_model=imput_model, imputer_configs=self.hparameters)
        mli.fit(X=X_features)
        X_imputed = mli.transform(X=X_features)
        
        # Add target back
        X_imputed[self.target] = y_target.values
        
        return self._ensure_target_dtype(X_imputed)
    
    def _apply_imputation_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        imput_model: str
    ) -> pd.DataFrame:
        """Apply imputation fitted on train to test data (excluding target)."""
        # Separate features and target
        X_train = train.drop(columns=[self.target])
        X_test = test.drop(columns=[self.target])
        y_test = test[self.target]
        
        # Fit on train features, transform test features
        mli = MLimputer(imput_model=imput_model, imputer_configs=self.hparameters)
        mli.fit(X=X_train)
        X_test_imputed = mli.transform(X=X_test)
        
        # Add target back
        X_test_imputed[self.target] = y_test.values
    
        return X_test_imputed
    
    def _apply_encoding(
        self,
        data: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """Apply label encoding to categorical columns (excluding target)."""
        # Get categorical columns EXCLUDING the target
        cat_cols = [
            col for col in data.select_dtypes(include=self.categorical_dtypes).columns
            if col != self.target
        ]
        
        if not cat_cols:
            # No categorical features to encode, but ensure target is properly typed
            return self._ensure_target_dtype(data)
        
        if fit or self.label_encoder is None:
            self.label_encoder = AutoLabelEncoder()
            self.label_encoder.fit(data[cat_cols])
        
        # Transform only the categorical features
        encoded_data = self.label_encoder.transform(data)
        
        # Ensure target has the correct dtype
        return self._ensure_target_dtype(encoded_data)

    def _ensure_target_dtype(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure target column has appropriate dtype for the problem type."""
        if self.target not in data.columns:
            return data
        
        data = data.copy()
        
        if self.problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            # For classification, ensure target is integer type
            if data[self.target].dtype == 'object':
                # Use a persistent target encoder to maintain consistency
                if not hasattr(self, 'target_encoder') or self.target_encoder is None:
                    self.target_encoder = LabelEncoder()
                    data[self.target] = self.target_encoder.fit_transform(data[self.target])
                    logger.debug(f"Fitted target encoder and converted target to integers for {self.problem_type.value}")
                else:
                    # Use the already fitted encoder
                    data[self.target] = self.target_encoder.transform(data[self.target])
                    logger.debug(f"Transformed target using existing encoder")
        
        return data
    
    def _identify_best_imputer(self) -> None:
        """Identify best performing imputation model."""
        if self.leaderboard is None or self.leaderboard.empty:
            logger.warning("No leaderboard available to identify best imputer")
            return
        
        # Filter for aggregate results
        aggregate_results = self.leaderboard[
            self.leaderboard[ResultColumn.FOLD.value] == ResultColumn.AGGREGATE.value
        ]
        
        if aggregate_results.empty:
            logger.warning("No aggregate results found")
            return
        
        # Get primary metric and direction
        primary_metric = MetricsCalculator.get_primary_metric(self.problem_type)
        metric_direction = MetricsCalculator.get_metric_direction(primary_metric)
        
        # Map to uppercase for aggregate columns
        metric_mapping = {
            'Mean Absolute Error': 'MAE',
            'MAE': 'MAE',
            'mae': 'MAE',
            'F1 Score': 'F1',
            'F1': 'F1',
            'f1': 'F1',
            'accuracy': 'ACCURACY',
            'Accuracy': 'ACCURACY',
            'r2': 'R2',
            'R2': 'R2',
            'rmse': 'RMSE',
            'RMSE': 'RMSE',
            'mape': 'MAPE',
            'MAPE': 'MAPE',
            'mse': 'MSE',
            'MSE': 'MSE'
        }
        
        # Get the UPPERCASE metric name for aggregate columns
        uppercase_metric = metric_mapping.get(primary_metric, primary_metric.upper())
        
        # Create the actual column name
        metric_column = f"{uppercase_metric} {MetricAggregation.MEAN.value}"
        
        # Check if column exists
        if metric_column not in aggregate_results.columns:
            logger.warning(f"Metric column '{metric_column}' not found in results")
            # Log available columns for debugging
            logger.debug(f"Available columns: {aggregate_results.columns.tolist()}")
            return
        
        # Sort and get best
        ascending = (metric_direction == 'minimize')
        sorted_results = aggregate_results.sort_values(
            by=metric_column,
            ascending=ascending
        )
        
        self.best_imputer = sorted_results.iloc[0][ResultColumn.IMPUTER_MODEL.value]
        logger.info(f"Best imputation model identified: {self.best_imputer}")

    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive summary report."""
        if self.leaderboard is None:
            raise ValueError("No evaluation results available")
        
        # Get primary metric for the summary
        primary_metric = MetricsCalculator.get_primary_metric(self.problem_type)
        
        # Extract best performance
        best_performance = {}
        if self.best_imputer:
            aggregate_results = self.leaderboard[
                (self.leaderboard[ResultColumn.FOLD.value] == ResultColumn.AGGREGATE.value) &
                (self.leaderboard[ResultColumn.IMPUTER_MODEL.value] == self.best_imputer)
            ]
            if not aggregate_results.empty:
                metric_column = f"{primary_metric} {MetricAggregation.MEAN.value}"
                if metric_column in aggregate_results.columns:
                    best_performance = {
                        'metric': primary_metric,
                        'value': aggregate_results.iloc[0][metric_column]
                    }
        
        return {
            'problem_type': self.problem_type.value,
            'primary_metric': primary_metric,
            'n_imputation_models': len(self.config.imputation_models),
            'n_folds': self.config.n_splits,
            'best_imputer': self.best_imputer,
            'best_performance': best_performance,
            'target_column': self.target,
            'dataset_shape': self.train.shape,
            'leaderboard': self.leaderboard
        }
    
    def get_metric_comparison(self, metric_name: str = None) -> pd.DataFrame:
        """Get comparison of a specific metric across imputation models."""
        if self.leaderboard is None:
            raise ValueError("No evaluation results available")
        
        # Use primary metric if none specified
        if metric_name is None:
            metric_name = MetricsCalculator.get_primary_metric(self.problem_type)
        
        # Filter aggregate results
        aggregate_results = self.leaderboard[
            self.leaderboard[ResultColumn.FOLD.value] == ResultColumn.AGGREGATE.value
        ]
        
        # Create comparison DataFrame
        comparison = aggregate_results[
            [ResultColumn.MODEL.value, ResultColumn.IMPUTER_MODEL.value]
        ].copy()
        
        # Add metric columns
        metric_mean = f"{metric_name} {MetricAggregation.MEAN.value}"
        if metric_mean in aggregate_results.columns:
            comparison[metric_mean] = aggregate_results[metric_mean]
        
        metric_std = f"{metric_name} {MetricAggregation.STD.value}"
        if metric_std in aggregate_results.columns:
            comparison[metric_std] = aggregate_results[metric_std]
        
        # Sort by metric
        if metric_mean in comparison.columns:
            direction = MetricsCalculator.get_metric_direction(metric_name)
            ascending = (direction == 'minimize')
            comparison = comparison.sort_values(metric_mean, ascending=ascending)
        
        return comparison