import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional
from logging.handlers import RotatingFileHandler


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MLImputerLogger:
    """Custom singleton logger for MLImputer."""
    
    _instance: Optional['MLImputerLogger'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = self._setup_logger()
            self.initialized = True

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MLimputer')
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                ' %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)

            # File handler
            log_dir = Path(os.getenv("MLIMPUTER_LOG_DIR", Path.home() / ".mlimputer" / "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"mlimputer_{datetime.now():%Y%m%d}.log"

            file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)


logger = MLImputerLogger()
