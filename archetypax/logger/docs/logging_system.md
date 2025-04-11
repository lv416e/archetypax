# ArchetypAX Logging System

## Overview

The ArchetypAX logging system provides a robust, flexible, and standardized approach to logging throughout the project. It replaces ad-hoc print statements with a proper logging infrastructure that enables better debugging, monitoring, and analysis.

## Key Features

- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR, and CRITICAL levels for appropriate message categorization
- **Contextual Logging**: Automatically includes module names and timestamps
- **File and Console Output**: Configurable logging to both console and files
- **Performance Tracking**: Built-in timer utilities for measuring operation performance
- **Structured Message Templates**: Standardized, well-crafted log messages for consistency
- **Thread Safety**: Safe for use in multithreaded environments

## Components

The logging system consists of two main modules:

1. **`archetypax.logger.core`**: Core logging functionality
   - `get_logger()`: Main function to obtain a configured logger
   - `ArchetypAXLogger`: Enhanced logger class with performance tracking
   - Configuration utilities for customizing log behavior

2. **`archetypax.logger.messages`**: Standardized message templates
   - Categorized message templates (initialization, progress, warnings, etc.)
   - `get_message()` function for retrieving formatted messages

## Basic Usage

```python
from archetypax.logger import get_logger, get_message

# Get a logger for your module
logger = get_logger(__name__)

# Basic logging
logger.debug("Detailed debugging information")
logger.info("General information message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error message")

# Using message templates
logger.info(get_message("init", "model_init",
                       model_name="ArchetypalAnalysis",
                       n_archetypes=5))

# Performance tracking
with logger.perf_timer("complex_operation"):
    # Your code here
    complex_computation()
```

## Log File Location

By default, logs are stored in:

```
~/.archetypax/logs/archetypax_YYYYMMDD.log
```

You can specify a custom log file location when getting a logger:

```python
logger = get_logger(__name__, log_file="/path/to/custom.log")
```

## Message Categories

The `messages` module provides templates in these categories:

- **init**: Initialization messages
- **progress**: Progress and operation messages
- **warning**: Warning messages
- **error**: Error messages
- **data**: Data processing messages
- **result**: Results and evaluation messages
- **performance**: Performance metrics

## Best Practices

1. **Get a logger at the module level**:
   ```python
   logger = get_logger(__name__)
   ```

2. **Use appropriate log levels**:
   - DEBUG: Detailed information for diagnosing problems
   - INFO: Confirmation that things are working as expected
   - WARNING: Something unexpected happened
   - ERROR: A more serious problem
   - CRITICAL: A serious error that may prevent the program from continuing

3. **Use structured message templates** when available:
   ```python
   logger.info(get_message("category", "key", param1=value1, param2=value2))
   ```

4. **Log exceptions properly**:
   ```python
   try:
       # Code that might raise an exception
   except Exception as e:
       logger.error(get_message("error", "computation_error", error_msg=str(e)))
   ```

5. **Use performance timers** for operations you want to measure:
   ```python
   with logger.perf_timer("operation_name"):
       # Code to time
   ```

## Examples and Resources

- `archetypax/logger/examples/logger_usage.py`: Complete example of logger usage
- `archetypax/logger/examples/refactoring_guide.py`: Guide for refactoring existing code

## Refactoring Existing Code

When converting existing print statements to use the logging system:

1. Replace print statements with appropriate log level calls
2. Use message templates when available
3. Add context to log messages
4. Use try/except blocks with proper error logging

See `archetypax/logger/examples/refactoring_guide.py` for detailed examples.
