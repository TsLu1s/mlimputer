import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set


from mlimputer.utils.constants import ImputationColumn, ImputationStrategy, ImputationContext, ImputationState
from mlimputer.utils.exceptions import ModelNotFittedError

class AbstractMLImputer(ABC):
    """Abstract base class for MLimputer imputation pipeline."""
    
    # Class-level constants
    NUMERIC_DTYPES: Set[str] = {
        'int16', 'int32', 'int64', 
        'float16', 'float32', 'float64'
    }
    CATEGORICAL_DTYPES: Set[str] = {'object', 'category'}
    
    def __init__(self, strategy: ImputationStrategy):
        """
        Initialize abstract imputer.
        """
        self._strategy = strategy
        self._state: Optional[ImputationState] = None
        self._fitted_configs: Dict[str, Dict[str, Any]] = {}
    
    @property
    def is_fitted(self) -> bool:
        """Check if imputer is fitted."""
        return self._state is not None and self._state.is_fitted
    
    @property
    def strategy(self) -> ImputationStrategy:
        """Get imputation strategy."""
        return self._strategy
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'AbstractMLImputer':
        """
        Fit the imputer on data.
        """
        self._validate_fit_data(X)
        return self
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by imputing missing values.
        """
        self._validate_transform_data(X)
        return X
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    # Protected validation methods
    def _validate_fit_data(self, X: pd.DataFrame) -> None:
        """Validate data for fitting."""
        if X.empty:
            raise ValueError("Cannot fit on empty DataFrame")
        if X.isna().all().any():
            cols = X.columns[X.isna().all()].tolist()
            raise ValueError(f"Columns {cols} contain only missing values")
    
    def _validate_transform_data(self, X: pd.DataFrame) -> None:
        """Validate data for transformation."""
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"{self.__class__.__name__} must be fitted before transform"
            )
        if X.empty:
            raise ValueError("Cannot transform empty DataFrame")
    
    # Protected utility methods
    def _get_missing_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get columns with missing values."""
        num_cols = X.select_dtypes(include=self.NUMERIC_DTYPES).columns
        
        missing_info = []
        for col in num_cols:
            null_count = X[col].isna().sum()
            if null_count > 0:
                missing_info.append({
                    ImputationColumn.COLUMN_NAME.value: col,
                    ImputationColumn.NULL_COUNT.value: null_count,
                    ImputationColumn.NULL_PERCENTAGE.value: null_count / len(X)
                })
        
        if not missing_info:
            return pd.DataFrame()
        
        return pd.DataFrame(missing_info).sort_values(
            ImputationColumn.NULL_PERCENTAGE.value
        ).reset_index(drop=True)
    
    def _get_categorical_columns(
        self, 
        df: pd.DataFrame, 
        exclude: Optional[str] = None
    ) -> List[str]:
        """Get categorical columns, optionally excluding one."""
        cols = df.select_dtypes(include=self.CATEGORICAL_DTYPES).columns.tolist()
        return [c for c in cols if c != exclude] if exclude else cols
    
    @abstractmethod
    def _create_column_imputer(
        self, 
        X: pd.DataFrame, 
        column: str, 
        context: ImputationContext
    ) -> Dict[str, Any]:
        """Create imputer configuration for a single column."""
        pass
    
    @abstractmethod
    def _apply_column_imputation(
        self,
        X: pd.DataFrame,
        column: str,
        config: Dict[str, Any]
    ) -> None:
        """Apply imputation to a single column."""
        pass