import pickle
import json
import joblib
from pathlib import Path
from typing import Any, Optional, Dict, Union
from datetime import datetime
from mlimputer.utils.exceptions import SerializationError


class ModelSerializer:
    """Handle model serialization and deserialization."""
    
    SUPPORTED_FORMATS = ['pickle', 'joblib', 'json']
    
    @staticmethod
    def save(
        obj: Any,
        filepath: Union[str, Path],
        format: str = 'joblib',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save object to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format not in ModelSerializer.SUPPORTED_FORMATS:
            raise SerializationError(f"Unsupported format: {format}")
        
        try:
            if format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(obj, f)
            elif format == 'joblib':
                joblib.dump(obj, filepath)
            elif format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(obj, f, indent=2, default=str)
            
            # Save metadata if provided
            if metadata:
                metadata['save_timestamp'] = datetime.now().isoformat()
                metadata['format'] = format
                metadata_path = filepath.with_suffix('.meta.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                    
        except Exception as e:
            raise SerializationError(f"Failed to save object: {str(e)}")
    
    @staticmethod
    def load(
        filepath: Union[str, Path],
        format: str = 'joblib'
    ) -> Any:
        """Load object from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise SerializationError(f"File not found: {filepath}")
        
        if format not in ModelSerializer.SUPPORTED_FORMATS:
            raise SerializationError(f"Unsupported format: {format}")
        
        try:
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif format == 'joblib':
                return joblib.load(filepath)
            elif format == 'json':
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            raise SerializationError(f"Failed to load object: {str(e)}")
    
    @staticmethod
    def load_with_metadata(
        filepath: Union[str, Path],
        format: str = 'joblib'
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """Load object with its metadata."""
        obj = ModelSerializer.load(filepath, format)
        
        metadata_path = Path(filepath).with_suffix('.meta.json')
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return obj, metadata