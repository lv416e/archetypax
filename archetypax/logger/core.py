"""
Advanced logging system for the ArchetypAX project.

This module provides a sophisticated logging system that can be used throughout the ArchetypAX project.

It includes:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Customizable formatters for different outputs
- File and console handlers
- Contextualized logging with module information
- Performance tracking capabilities

Example usage:
    ```python
    from archetypax.logger import get_logger

    # Get a logger for your module
    logger = get_logger(__name__)

    # Use the logger
    logger.debug("Detailed debugging information")
    logger.info("General information message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical error message")

    # Performance tracking
    with logger.perf_timer("operation_name"):
        # Your code here
        pass
    ```
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_loggers: dict[str, "ArchetypAXLogger"] = {}


class ArchetypAXLogger(logging.Logger):
    """Enhanced logger for ArchetypAX with additional utilities."""

    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize the ArchetypAX logger.

        Args:
            name: Name of the logger (usually the module name)
            level: Default logging level
        """
        super().__init__(name, level)
        self._timers: dict[str, float] = {}

    @contextmanager
    def perf_timer(self, operation_name: str, level: int = logging.DEBUG):
        """
        Context manager to track and log the execution time of a code block.

        Args:
            operation_name: Descriptive name for the operation being timed
            level: Log level to use for the timing message
        """
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        self.log(
            level,
            f"Performance: {operation_name} completed in {elapsed_time:.4f} seconds",
        )


def configure_logger(
    logger_name: str,
    level: str | int = "info",
    log_file: str | None = None,
    console: bool = True,
    format_string: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> ArchetypAXLogger:
    """
    Configure a logger with specified settings.

    Args:
        logger_name: Name of the logger
        level: Logging level (can be string or logging level constant)
        log_file: Path to log file (if None, file logging is disabled)
        console: Whether to log to console
        format_string: Format string for log messages
        date_format: Format string for dates in log messages

    Returns:
        Configured logger instance
    """
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)

    logger = ArchetypAXLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(format_string, date_format)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(
    name: str,
    level: str | int = "info",
    log_file: str | None = None,
) -> ArchetypAXLogger:
    """
    Get or create a logger with the specified name.

    Args:
        name: Name of the logger (usually __name__ from the calling module)
        level: Logging level
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    if log_file is None:
        log_dir = os.path.expanduser("~/.archetypax/logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"archetypax_{date_str}.log")

    logger = configure_logger(name, level=level, log_file=log_file)
    _loggers[name] = logger

    return logger
