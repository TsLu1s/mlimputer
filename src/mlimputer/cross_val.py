import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List
from sklearn.base import BaseEstimator
import warnings
from sklearn.exceptions import ConvergenceWarning
from mlimputer.metrics import metrics_regression, metrics_classification 

def cross_validation(
    X: pd.DataFrame,
    target: str,
    n_splits: int = 5,
    models: List[BaseEstimator] = [],
) -> pd.DataFrame:
    """
    Perform cross-validation with automated context detection and comprehensive metrics.
    
    Args:
        X: Input DataFrame containing features and target
        target: Name of the target column
        n_splits: Number of cross-validation splits
        models: List of scikit-learn compatible models
    
    Returns:
        pd.DataFrame: Comprehensive leaderboard with cross-validation results
    """
    if not models:
        raise ValueError("No models provided for cross-validation")
    
    y = X[target].copy()
    X_features = X.drop(columns=[target])
    
    # Detect prediction context (classification or regression)
    categorical_dtypes = {"object", "category"}
    is_categorical = y.dtype in categorical_dtypes
    
    # Configure cross-validation strategy
    if is_categorical:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    list_metrics = []
    
    # Suppress convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_features, y if is_categorical else None)):
            print(f"\nFold {fold_idx + 1}/{n_splits}:")
            
            X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            for model in models:
                model_name = model.__class__.__name__
                print(f"Training {model_name}...")
                
                try:
                    # Train and predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    # Calculate metrics based on problem type
                    if is_categorical:
                        metrics = metrics_classification(y_val, y_pred)
                    else:
                        metrics = metrics_regression(y_val, y_pred)
                    
                    # Add metadata
                    metrics['Model'] = model_name
                    metrics['Fold'] = fold_idx + 1
                    metrics['Train Samples'] = len(X_train)
                    metrics['Validation Samples'] = len(X_val)
                    
                    list_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    continue
    
    # Combine all results
    leaderboard = pd.concat(list_metrics, ignore_index=True)
    
    # Calculate aggregate statistics
    metric_columns = leaderboard.select_dtypes(include=[np.number]).columns
    metric_columns = metric_columns.drop(['Fold', 'Train Samples', 'Validation Samples'])
    
    aggregated_metrics = []
    for model_name in leaderboard['Model'].unique():
        model_results = leaderboard[leaderboard['Model'] == model_name]
        
        agg_metrics = {'Model': model_name, 'Fold': 'Aggregate'}
        
        for metric in metric_columns:
            agg_metrics[f'{metric} Mean'] = model_results[metric].mean()
            #agg_metrics[f'{metric} Std'] = model_results[metric].std()
        
        aggregated_metrics.append(pd.DataFrame([agg_metrics]))
    
    # Append aggregates to leaderboard
    final_leaderboard = pd.concat([leaderboard] + aggregated_metrics, ignore_index=True)
    
    # Sort based on primary metric
    if is_categorical:
        primary_metric = 'F1'
        ascending_order = False  # Higher F1 is better
    else:
        primary_metric = 'Mean Absolute Error'
        ascending_order = True   # Lower MAE is better
    
    final_leaderboard = final_leaderboard.sort_values(
        by=['Fold', primary_metric],
        ascending=[True, ascending_order]
    )
    
    return final_leaderboard #leaderboard,
