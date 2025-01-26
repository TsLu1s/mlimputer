from typing import List, Dict, Union
import pandas as pd
from sklearn.base import BaseEstimator
from atlantic.processing.encoders import AutoLabelEncoder
from mlimputer.imputation import MLimputer                
import mlimputer.cross_val as cv                    
from mlimputer.metrics import metrics_regression, metrics_classification 

class Evaluator:
    """
    A class to handle model evaluation with different imputation strategies,
    including cross-validation and test set evaluation.
    """
    
    def __init__(self, 
                 imputation_models: List[str] = None,
                 train: pd.DataFrame = pd.DataFrame(),
                 target : str = "Target_Column",
                 n_splits: int = 3,
                 hparameters: Dict[str, Dict[str, Union[int, float, str]]] = None):
        """
        Initialize the ModelEvaluator with configuration parameters.
        
        Args:
            imputation_models: List of imputation model names to evaluate
            train: Training dataset
            target: Name of the Target Column
            n_splits: Number of cross-validation splits
            hparameters: Dictionary of hyperparameters for each imputation model
        """
        self.imputation_models = imputation_models
        self.train = train
        self.target = target
        self.n_splits = n_splits
        self.hparameters = hparameters
        self.categorical_dtypes = {"object", "category"}
        self.numeric_types = {'int', 'int32', 'int64', 'float', 'float32', 'float64'}
        self.label_encoder = None

    
    def evaluate_imputation_models(self,
                                   models: List[BaseEstimator]) -> pd.DataFrame:
        """
        Evaluate different imputation models using cross-validation.
        
        Args:
            models: List of models to evaluate
            
        Returns:
            DataFrame containing evaluation results
        """
        list_leaderboards = []
        
        for imput_model in self.imputation_models:
            print(f"Evaluating imputation model: {imput_model}")
            _train = self.train.copy()
            
            # Apply imputation
            mli = MLimputer(imput_model=imput_model, imputer_configs=self.hparameters)
            mli.fit_imput(X=_train)
            train_imp = mli.transform_imput(X=_train)
            
            # Initialize and fit label encoder
            cat_cols = [col for col in _train.select_dtypes(include=self.categorical_dtypes).columns 
                       if col != self.target]
            if cat_cols:
                self.label_encoder = AutoLabelEncoder()
                self.label_encoder.fit(train_imp[cat_cols])
                train_imp = self._preprocess_data(train_imp)
            
            leaderboard_imp = cv.cross_validation(
                X=train_imp,
                target=self.target,
                n_splits=self.n_splits,
                models=models
            )
            
            leaderboard_imp['imputer_model'] = imput_model
            list_leaderboards.append(leaderboard_imp)
        
        self.leaderboard = pd.concat(list_leaderboards)
        return self._sort_leaderboard(self.leaderboard, self.train[self.target].dtype)
    
    def evaluate_test_set(self,
                         test: pd.DataFrame,
                         imput_model: str,
                         models: List[BaseEstimator]) -> Dict[str, float]:
        """
        Evaluate models on the test set using specified imputation.
        
        Args:
            test: Test dataset
            imput_model: Name of imputation model to use
            models: List of models to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        _train = self.train.copy()
        # Apply imputation
        mli = MLimputer(imput_model=imput_model, imputer_configs=self.hparameters)
        mli.fit_imput(X=_train)
        
        train_imp = mli.transform_imput(X=_train.copy())
        test_imp = mli.transform_imput(X=test.copy())
        
        # Initialize and fit label encoder on training data
        cat_cols = [col for col in self.train.select_dtypes(include=self.categorical_dtypes).columns 
                   if col != self.target]
        if cat_cols:
            self.label_encoder = AutoLabelEncoder()
            self.label_encoder.fit(train_imp[cat_cols])
            train_imp = self._preprocess_data(train_imp)
            test_imp = self._preprocess_data(test_imp)
        
        X_train = train_imp.drop(columns=[self.target])
        y_train = train_imp[self.target]
        X_test = test_imp.drop(columns=[self.target])
        y_test = test_imp[self.target]
        
        results_list = []
        for model in models:
            print(f"Testing performance of model: {model.__class__.__name__}")
            model.fit(X_train, y_train)
            metrics = (metrics_classification(y_test, model.predict(X_test))
                      if self.train[self.target].dtype in self.categorical_dtypes
                      else metrics_regression(y_test, model.predict(X_test)))
            metrics['model'] = model.__class__.__name__
            metrics['imputer_model'] = imput_model
            results_list.append(metrics)
        # Concatenate all results
        results = pd.concat(results_list, ignore_index=True)
    
        return results.reset_index(drop=True)
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted label encoder.
        
        Args:
            data: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        if self.label_encoder is not None:
            return self.label_encoder.transform(X=data)
        return data
    
    def _sort_leaderboard(self, leaderboard: pd.DataFrame, target_dtype) -> pd.DataFrame:
        """
        Sort leaderboard based on appropriate metric for the target type.
        
        Args:
            leaderboard: DataFrame containing evaluation results
            target_dtype: Data type of target variable
            
        Returns:
            Sorted leaderboard
        """
        if target_dtype in self.categorical_dtypes:
            return leaderboard.sort_values(by='F1', ascending=False)
        return leaderboard.sort_values(by='Mean Absolute Error', ascending=True)
    
    def get_best_imputer(self) -> str:
        """
        Identify the best-performing imputation model using the aggregated metrics from the leaderboard.
        
        Returns:
            Name of the best performing imputation model
            
        Raises:
            ValueError: If leaderboard is empty or missing required columns
        """
        if not isinstance(self.leaderboard, pd.DataFrame) or self.leaderboard.empty:
            raise ValueError("Leaderboard must be a non-empty DataFrame")
        
        # Filter for aggregate results only
        aggregate_results = self.leaderboard[self.leaderboard['Fold'] == 'Aggregate']
        
        # Sort based on target type
        if self.train[self.target].dtype in self.categorical_dtypes:
            self.sorted_results = aggregate_results.sort_values(
                by='F1 Mean', 
                ascending=False
            )
        else:
            self.sorted_results = aggregate_results.sort_values(
                by='Mean Absolute Error Mean', 
                ascending=True
            )
        
        # Get the imputation model with the best average performance
        return self.sorted_results.iloc[0]['imputer_model']