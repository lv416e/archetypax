"""
Example demonstrating the usage of the ArchetypAX logging system.

This script shows how to:
1. Set up and configure loggers
2. Use different log levels
3. Log messages with the standard templates
4. Measure performance with timers
5. Handle exceptions with proper logging
"""

import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from archetypax.logger import get_logger, get_message
from archetypax.models.archetypes import ImprovedArchetypalAnalysis


def simulate_data_processing():
    """Simulate a data processing operation that takes time."""
    time.sleep(0.5)
    return np.random.rand(100, 20)


def demonstrate_basic_logging():
    """Demonstrate basic logging functionality."""
    logger = get_logger(__name__, level="debug")
    logger.info("Starting logging demonstration")
    logger.debug("This is a debug message with detailed information")
    logger.info("This is an informational message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    n_archetypes = 5
    model_name = "ImprovedArchetypalAnalysis"
    logger.info(get_message("init", "model_init", model_name=model_name, n_archetypes=n_archetypes))

    try:
        result = 10 / 0
    except Exception as e:
        logger.error(get_message("error", "computation_error", error_msg=str(e)))


def demonstrate_performance_timing():
    """Demonstrate performance timing capabilities."""
    logger = get_logger("performance_demo")
    with logger.perf_timer("data_processing"):
        data = simulate_data_processing()
    logger.info(f"Generated data with shape {data.shape}")

    with logger.perf_timer("complete_analysis"):
        logger.info("Starting data analysis")

        with logger.perf_timer("normalization"):
            data_normalized = (data - data.mean()) / data.std()
            logger.info(get_message("data", "normalization", mean=data.mean(), std=data.std()))

        with logger.perf_timer("model_fitting"):
            model = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=5)

            try:
                model.fit(data_normalized)
                loss = model.loss_history[-1] if model.loss_history else float("nan")
                logger.info(get_message("result", "final_loss", loss=loss, iterations=len(model.loss_history)))
            except Exception as e:
                logger.error(get_message("error", "computation_error", error_msg=str(e)))


def demonstrate_log_file_output():
    """Demonstrate logging to a custom file."""
    custom_log_file = "custom_archetypax.log"

    logger = get_logger("file_demo", level="info", log_file=custom_log_file)
    logger.info(f"This message is being written to {custom_log_file}")

    for i in range(3):
        logger.info(get_message("progress", "iteration_progress", current=i + 1, total=3, loss=0.1 / (i + 1)))

    logger.info(f"Check {custom_log_file} to see these messages")


if __name__ == "__main__":
    print("=== ArchetypAX Logger Demonstration ===")

    print("\n1. Basic Logging:")
    demonstrate_basic_logging()

    print("\n2. Performance Timing:")
    demonstrate_performance_timing()

    print("\n3. Custom Log File:")
    demonstrate_log_file_output()

    print("\nDemonstration complete. Check the logs for details.")
