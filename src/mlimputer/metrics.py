import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error,
                             explained_variance_score,
                             max_error,
                             r2_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def metrics_regression(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Calculate comprehensive regression metrics"""
    try:
        metrics = {
            'Mean Absolute Error': mean_absolute_error(y_true, y_pred),
            'Mean Absolute Percentage Error': mean_absolute_percentage_error(y_true, y_pred),
            'Mean Squared Error': mean_squared_error(y_true, y_pred),
            'Explained Variance Score': explained_variance_score(y_true, y_pred),
            'Max Error': max_error(y_true, y_pred),
            'R2 Score': r2_score(y_true, y_pred)
        }
    except Exception as e:
        print(f"Error calculating regression metrics: {str(e)}")
        metrics = {key: None for key in [
            'Mean Absolute Error', 'Mean Absolute Percentage Error',
            'Mean Squared Error', 'Explained Variance Score',
            'Max Error', 'R2 Score'
        ]}
    
    return pd.DataFrame([metrics])

def metrics_classification(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Calculate comprehensive classification metrics"""
    try:
        n_classes = len(np.unique(y_true))
        average = 'weighted' if n_classes > 2 else 'binary'
        
        metrics = {
            'Precision': precision_score(y_true, y_pred, average=average),
            'F1': f1_score(y_true, y_pred, average=average),
            'Recall': recall_score(y_true, y_pred, average=average)
        }
        
        if n_classes > 2:
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
            
    except Exception as e:
        print(f"Error calculating classification metrics: {str(e)}")
        metrics = {
            'Precision': None,
            'F1': None,
            'Recall': None,
            'Accuracy': None
        }
    
    return pd.DataFrame([metrics])