"""
ArchetypAX Logging System.

This package provides a robust, flexible, and standardized approach to logging
throughout the ArchetypAX project.

Modules:
    core: Core logging functionality
    messages: Standard log message templates

Usage:
    from archetypax.logger import get_logger, get_message

    # Get a logger for your module
    logger = get_logger(__name__)

    # Basic logging
    logger.info("General information message")

    # Using message templates
    logger.info(
        get_message(
            "init",
            "model_init",
            model_name="ArchetypalAnalysis",
            n_archetypes=5,
        )
    )

    # Performance tracking
    with logger.perf_timer("operation_name"):
        # Your code here
        pass
"""

# Import main components for convenient access
from .core import ArchetypAXLogger, configure_logger, get_logger
from .messages import get_message

# Define public API
__all__ = [
    "ArchetypAXLogger",
    "configure_logger",
    "get_logger",
    "get_message",
]
