from typing import Optional, Any, Dict

class MLImputerError(Exception):
    """Base exception class for MLimputer."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class DataValidationError(MLImputerError):
    """Raised when data validation fails."""
    pass


class ModelNotFittedError(MLImputerError):
    """Raised when trying to use an unfitted model."""
    pass


class ConfigurationError(MLImputerError):
    """Raised when configuration is invalid."""
    pass


class ImputationError(MLImputerError):
    """Raised when imputation fails."""
    pass


class PreprocessingError(MLImputerError):
    """Raised when preprocessing fails."""
    pass


class EvaluationError(MLImputerError):
    """Raised when evaluation fails."""
    pass


class SerializationError(MLImputerError):
    """Raised when serialization/deserialization fails."""
    pass


class IncompatibleDataError(MLImputerError):
    """Raised when data is incompatible with the model."""
    pass