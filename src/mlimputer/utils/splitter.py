from typing import Tuple, List, Union, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Flexible data splitter with automatic index resetting.
    Handles train/val/test splits and multiple datasets.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize splitter with random state for reproducibility."""
        self.random_state = random_state
    
    def split(
        self,
        *arrays,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify: Optional[np.ndarray] = None, # Pass the target variable
        shuffle: bool = True
    ) -> Union[Tuple, List]:
        """
        Split arrays into train/val/test sets with automatic index reset.
        
        Args:
            *arrays: Arrays to split (X, y) or just (X)
            test_size: Test set proportion (0-1)
            val_size: Validation set proportion (0-1). If None, returns train/test only
            stratify: Array for stratified splitting (typically y for classification)
            shuffle: Whether to shuffle before splitting
            
        Returns:
            - If val_size is None: (X_train, X_test) or (X_train, X_test, y_train, y_test)
            - If val_size provided: (X_train, X_val, X_test) or (X_train, X_val, X_test, y_train, y_val, y_test)
        
        Examples:
            >>> splitter = DataSplitter(random_state=42)
            >>> 
            >>> # Simple train/test
            >>> X_train, X_test, y_train, y_test = splitter.split(X, y, test_size=0.2)
            >>> 
            >>> # Train/val/test
            >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            ...     X, y, test_size=0.2, val_size=0.2
            ... )
        """
        if val_size is None:
            # Standard train/test split
            results = train_test_split(
                *arrays,
                test_size=test_size,
                stratify=stratify,
                shuffle=shuffle,
                random_state=self.random_state
            )
        else:
            # Train/val/test split
            # First split: train+val vs test
            temp_results = train_test_split(
                *arrays,
                test_size=test_size,
                stratify=stratify,
                shuffle=shuffle,
                random_state=self.random_state
            )
            
            # Separate arrays (handle both X,y and X only cases)
            n_arrays = len(arrays)
            train_val_arrays = temp_results[:n_arrays]
            test_arrays = temp_results[n_arrays:]
            
            # Second split: train vs val
            val_ratio = val_size / (1 - test_size)
            stratify_val = stratify[:len(train_val_arrays[0])] if stratify is not None else None
            
            val_results = train_test_split(
                *train_val_arrays,
                test_size=val_ratio,
                stratify=stratify_val,
                shuffle=shuffle,
                random_state=self.random_state + 1
            )
            
            # Combine results: train, val, test
            train_arrays = val_results[:n_arrays]
            val_arrays = val_results[n_arrays:]
            
            results = []
            for i in range(n_arrays):
                results.extend([train_arrays[i], val_arrays[i], test_arrays[i]])
        
        # Reset all indices
        return self._reset_indices(results)
    
    def split_multiple(
        self,
        datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        test_size: float = 0.2,
        val_size: Optional[float] = None
    ) -> Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]]:
        """
        Split multiple datasets at once.

        Returns:
            Dictionary with structure:
            {
                'dataset_name': {
                    'X_train': DataFrame,
                    'X_val': DataFrame (if val_size),
                    'X_test': DataFrame,
                    'y_train': Series,
                    'y_val': Series (if val_size),
                    'y_test': Series
                }
            }
        """
        results = {}
        
        for name, (X, y) in datasets.items():
            if val_size is None:
                X_train, X_test, y_train, y_test = self.split(
                    X, y, test_size=test_size
                )
                results[name] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
            else:
                X_train, X_val, X_test, y_train, y_val, y_test = self.split(
                    X, y, test_size=test_size, val_size=val_size
                )
                results[name] = {
                    'X_train': X_train,
                    'X_val': X_val,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test
                }
        
        return results
    
    def quick_split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        test_size: float = 0.2
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], 
               Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Quick train/test split with sensible defaults.

        Returns:
            (X_train, X_test) if y is None
            (X_train, X_test, y_train, y_test) if y provided
        """
        if y is None:
            return self.split(X, test_size=test_size)
        return self.split(X, y, test_size=test_size)
    
    @staticmethod
    def _reset_indices(arrays: List) -> List:
        """Reset indices for all arrays."""
        reset_arrays = []
        for arr in arrays:
            if hasattr(arr, 'reset_index'):
                reset_arrays.append(arr.reset_index(drop=True))
            else:
                reset_arrays.append(arr)
        return reset_arrays


# Convenience function (backward compatibility)
def adjusted_train_test_split(*arrays, **options):
    """Train test split with automatic index reset."""
    splitter = DataSplitter(random_state=options.get('random_state', 42))
    return splitter.split(*arrays, test_size=options.get('test_size', 0.25))