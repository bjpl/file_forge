"""
Logging utilities for FileForge.

Provides centralized logging configuration with Rich console output,
file rotation, and configurable log levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler
from logging.handlers import RotatingFileHandler


# Global configuration
DEFAULT_LOG_DIR = Path.home() / ".fileforge" / "logs"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_FORMAT = "%(message)s"
DEFAULT_DATE_FORMAT = "[%Y-%m-%d %H:%M:%S]"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = DEFAULT_LOG_LEVEL,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
) -> None:
    """
    Configure global logging for FileForge.

    Sets up Rich console handler for beautiful terminal output and optional
    rotating file handler for persistent logs.

    Args:
        log_dir: Directory for log files. Defaults to ~/.fileforge/logs
        log_level: Minimum log level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to write logs to file
        enable_console_logging: Whether to output logs to console

    Example:
        >>> setup_logging(log_level=logging.DEBUG)
        >>> logger = get_logger(__name__)
        >>> logger.info("FileForge initialized")
    """
    # Use default log directory if not specified
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger("fileforge")
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler with Rich formatting
    if enable_console_logging:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True,
        )
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            DEFAULT_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file_logging:
        log_file = log_dir / "fileforge.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing file: %s", filename)
    """
    # Ensure logging is setup
    if not logging.getLogger("fileforge").handlers:
        setup_logging()

    # Return child logger under fileforge namespace
    return logging.getLogger(f"fileforge.{name}")


def set_log_level(level: int) -> None:
    """
    Change the global log level at runtime.

    Args:
        level: New log level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> set_log_level(logging.DEBUG)
    """
    root_logger = logging.getLogger("fileforge")
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def get_log_file_path() -> Optional[Path]:
    """
    Get the path to the current log file.

    Returns:
        Path to log file, or None if file logging is not enabled

    Example:
        >>> log_path = get_log_file_path()
        >>> if log_path:
        ...     print(f"Logs: {log_path}")
    """
    root_logger = logging.getLogger("fileforge")
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            return Path(handler.baseFilename)
    return None


# Initialize logging on module import with sensible defaults
setup_logging()
