# Refactoring Guide: Using the ArchetypAX Logging System

This guide demonstrates how to convert print statements and other ad-hoc logging to use the new structured logging system in ArchetypAX. Below you'll find before/after examples and best practices for proper logging integration.

## Example 1: Basic Print Statement Conversion

### BEFORE: Using print statements
```python
def fit(self, X, normalize=False, **kwargs):
    # Code...
    print(f"Data shape: {X_jax.shape}")
    print(f"Data range: min={jnp.min(X_jax):.4f}, max={jnp.max(X_jax):.4f}")
    # More code...

    for i in range(max_iter):
        # Iteration code...
        print(f"Iteration {i}, Loss: {loss_value:.6f}")
```

### AFTER: Using the logging system
```python
from archetypax.logger import get_logger, get_message

# Get a logger for this module
logger = get_logger(__name__)

def fit(self, X, normalize=False, **kwargs):
    # Code...
    logger.info(get_message("data", "data_shape",
                           shape=X_jax.shape,
                           min=float(jnp.min(X_jax)),
                           max=float(jnp.max(X_jax))))
    # More code...

    for i in range(max_iter):
        # Iteration code...
        logger.info(get_message("progress", "iteration_progress",
                              current=i,
                              total=max_iter,
                              loss=loss_value))
```

## Example 2: Error Handling and Warnings

### BEFORE: Ad-hoc error and warning messages
```python
try:
    # Some computation
    result = complex_computation()
except Exception as e:
    print(f"Error at iteration {i}: {e!s}")

# Warning message
if np.isnan(loss_value):
    print(f"Warning: NaN detected at iteration {i}. Stopping early.")
```

### AFTER: Structured error and warning logging
```python
try:
    # Some computation
    result = complex_computation()
except Exception as e:
    logger.error(get_message("error", "computation_error",
                            error_msg=f"at iteration {i}: {e!s}"))

# Warning message
if np.isnan(loss_value):
    logger.warning(get_message("warning", "nan_detected",
                              iteration=i))
```

## Example 3: Tracking Performance

### BEFORE: Manual timing
```python
start_time = time.time()
# Perform some operation
complex_operation()
elapsed_time = time.time() - start_time
print(f"Operation completed in {elapsed_time:.4f} seconds")
```

### AFTER: Using the performance timer
```python
with logger.perf_timer("complex_operation"):
    # Perform some operation
    complex_operation()
```

## Example 4: Complete Refactoring of a Method

### BEFORE: Original method with print statements
```python
def _optimize(self, X, initial_archetypes, initial_weights, max_iter=500):
    print(f"Initial loss: {initial_loss:.6f}")

    for i in range(max_iter):
        try:
            # Update weights and archetypes
            weights, archetypes, loss_value = update_step(weights, archetypes, X)

            if np.isnan(loss_value):
                print(f"Warning: NaN detected at iteration {i}. Stopping early.")
                break

            if abs(previous_loss - loss_value) < self.tol:
                print(f"Converged at iteration {i}")
                break

            previous_loss = loss_value
            self.loss_history.append(loss_value)

            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss_value:.6f}")

        except Exception as e:
            print(f"Error at iteration {i}: {e!s}")
            break

    print(f"Final loss: {self.loss_history[-1]:.6f}")
```

### AFTER: Refactored method with proper logging
```python
def _optimize(self, X, initial_archetypes, initial_weights, max_iter=500):
    logger = get_logger(__name__)

    # Log initial state
    logger.info(get_message("progress", "iteration_progress",
                          current=0,
                          total=max_iter,
                          loss=initial_loss))

    with logger.perf_timer("optimization"):
        for i in range(max_iter):
            try:
                # Update weights and archetypes
                with logger.perf_timer(f"iteration_{i}"):
                    weights, archetypes, loss_value = update_step(weights, archetypes, X)

                if np.isnan(loss_value):
                    logger.warning(get_message("warning", "nan_detected",
                                             iteration=i))
                    break

                if abs(previous_loss - loss_value) < self.tol:
                    logger.info(get_message("progress", "converged",
                                          iteration=i,
                                          tolerance=self.tol))
                    break

                previous_loss = loss_value
                self.loss_history.append(loss_value)

                if i % 10 == 0:
                    logger.info(get_message("progress", "iteration_progress",
                                          current=i,
                                          total=max_iter,
                                          loss=loss_value))

            except Exception as e:
                logger.error(get_message("error", "computation_error",
                                        error_msg=f"at iteration {i}: {e!s}"))
                break

    logger.info(get_message("result", "final_loss",
                          loss=self.loss_history[-1],
                          iterations=len(self.loss_history)))
```

## Best Practices for Logging

1. **Initialize a logger at the module level:**
   - `logger = get_logger(__name__)`

2. **Use appropriate log levels:**
   - **DEBUG**: Detailed information, typically of interest only when diagnosing problems
   - **INFO**: Confirmation that things are working as expected
   - **WARNING**: An indication that something unexpected happened, but the program is still working
   - **ERROR**: Due to a more serious problem, the program has not been able to perform a function
   - **CRITICAL**: A serious error, indicating that the program itself may be unable to continue running

3. **Use structured message templates:**
   - Use `get_message()` with appropriate category and key
   - Provide all required parameters for message formatting

4. **Log exceptions properly:**
   - Catch exceptions and log them with appropriate context
   - Include relevant details like iteration number, parameters, etc.

5. **Use performance timers for timing operations:**
   - Wrap time-sensitive operations with `logger.perf_timer()`
   - Use descriptive names for operations

6. **Don't mix print statements and logging:**
   - Convert all print statements to appropriate log calls
   - Reserve print statements for direct user interaction only

7. **Include contextual information:**
   - Log messages should provide enough context to be understood
   - Include relevant variable values, operation names, etc.

8. **Don't overlog:**
   - Be selective about what to log at each level
   - Use DEBUG level for high-volume or detailed information
   - Consider using conditional logging for verbose operations

## Additional Resources

1. **Module Documentation:**
   - `archetypax.logger.core`: Main logging module
   - `archetypax.logger.messages`: Message templates

2. **Example Usage:**
   - `archetypax/logger/examples/logger_usage.py`: Complete example of logger usage

3. **Python Logging Documentation:**
   - https://docs.python.org/3/library/logging.html
