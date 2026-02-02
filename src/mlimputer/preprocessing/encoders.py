import numpy as np
import pandas as pd

from typing import Dict, Optional, Any

from sklearn.preprocessing import LabelEncoder

from mlimputer.preprocessing.abstract import AbstractEncoder
from mlimputer.utils.logging import logger


class AutoLabelEncoder(AbstractEncoder):
    """Automatic label encoder for categorical features with unknown value handling."""

    def __init__(self, unknown_value: str = "Unknown"):
        """Initialize AutoLabelEncoder."""
        super().__init__()
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._unknown_value = unknown_value
        self._unknown_indices: Dict[str, int] = {}
        self._categorical_columns: list = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AutoLabelEncoder":
        """Fit label encoders for each categorical column only."""
        super().fit(X, y)

        # Identify categorical columns (object or category dtype, not numeric or datetime)
        self._categorical_columns = X.select_dtypes(
            include=["object", "category"], exclude=["datetime64", "timedelta64"]
        ).columns.tolist()

        if not self._categorical_columns:
            logger.info("No categorical columns found to encode")
            return self

        logger.info(f"Found {len(self._categorical_columns)} categorical columns to encode")

        for col in self._categorical_columns:
            le = LabelEncoder()
            # Fit on unique values including unknown
            unique_values = X[col].dropna().unique()
            unique_with_unknown = np.append(unique_values, self._unknown_value)
            le.fit(unique_with_unknown)

            self._label_encoders[col] = le
            # Store the index for unknown value
            self._unknown_indices[col] = le.transform([self._unknown_value])[0]

            logger.debug(
                f"Fitted label encoder for column '{col}' with {len(unique_values)} unique values"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features to numerical values."""
        self._validate_fitted()
        X_encoded = X.copy()

        # Only transform categorical columns that were fitted
        for col in self._categorical_columns:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found in input DataFrame")
                continue

            le = self._label_encoders[col]
            unknown_idx = self._unknown_indices[col]

            # Transform with unknown value handling
            X_encoded[col] = X[col].apply(lambda x: self._safe_transform(x, le, unknown_idx))

        return X_encoded

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform numerical values back to categories."""
        self._validate_fitted()
        X_decoded = X.copy()

        for col in self._categorical_columns:
            if col not in X.columns:
                continue

            le = self._label_encoders[col]
            # Handle potential out-of-bounds indices
            X_decoded[col] = X[col].apply(lambda x: self._safe_inverse_transform(x, le))

        return X_decoded

    def _safe_transform(self, value: Any, encoder: LabelEncoder, unknown_idx: int) -> int:
        """Safely transform a value with unknown handling."""
        if pd.isna(value):
            return unknown_idx
        try:
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                return unknown_idx
        except:
            return unknown_idx

    def _safe_inverse_transform(self, value: Any, encoder: LabelEncoder) -> Any:
        """Safely inverse transform a value."""
        if pd.isna(value):
            return self._unknown_value
        try:
            return encoder.inverse_transform([int(value)])[0]
        except:
            return self._unknown_value

    def get_encoder_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about fitted encoders."""
        info = {}
        for col, le in self._label_encoders.items():
            info[col] = {
                "n_classes": len(le.classes_),
                "classes": le.classes_.tolist()[:10],  # First 10 classes
                "unknown_index": self._unknown_indices[col],
            }
        return info

    def get_categorical_columns(self) -> list:
        """Get list of categorical columns that will be/were encoded."""
        return self._categorical_columns
