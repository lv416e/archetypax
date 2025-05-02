"""Standard log message templates for ArchetypAX.

This module provides a collection of well-crafted, consistent log messages for use throughout the ArchetypAX project.
Using these templates ensures that logs are informative, actionable, and maintain a consistent tone.

Each message follows best practices for logging:
- Clear and concise
- Appropriate for the log level
- Provides context
- Actionable when necessary
- Professional tone
"""

INIT_MESSAGES = {
    "model_init": "Initializing {model_name} with {n_archetypes} archetypes",
    "data_loading": "Loading dataset with {n_samples} samples and {n_features} features",
    "config_loaded": "Configuration loaded from {config_path}",
    "random_seed": "Random seed set to {seed} for reproducibility",
}

PROGRESS_MESSAGES = {
    "iteration_progress": "Iteration {current}/{total}: loss={loss:.6f}, boundary_weights={boundary_weights}",
    "converged": "Convergence achieved at iteration {iteration} with tolerance {tolerance}",
    "early_stopping": "Early stopping triggered at iteration {iteration}. No improvement for {patience} iterations",
    "operation_complete": "Operation '{operation_name}' completed successfully in {elapsed_time:.4f} seconds",
}

WARNING_MESSAGES = {
    "nan_detected": "NaN values detected during computation at iteration {iteration}. Stopping early",
    "slow_convergence": "Convergence is proceeding slowly. Consider adjusting learning rate or maximum iterations",
    "high_loss": "Unusually high loss value: {loss:.6f}. This may indicate optimization issues",
    "loss_increase": "Loss increased from {previous:.6f} to {current:.6f}. Restoring best parameters",
    "degenerate_solution": "Warning: Solution may be degenerate. Parameter {param_name} contains near-{value_type} values",
}

ERROR_MESSAGES = {
    "computation_error": "Error during computation: {error_msg}",
    "invalid_parameter": "Invalid parameter: {param_name}={param_value}. Expected {expected}",
    "dimension_mismatch": "Dimension mismatch: {component} expected shape {expected_shape}, got {actual_shape}",
    "convergence_failure": "Failed to converge after {max_iter} iterations. Final loss: {final_loss:.6f}",
    "initialization_failed": "Initialization strategy '{strategy}' failed: {error_msg}. Falling back to '{fallback}'",
}

DATA_MESSAGES = {
    "normalization": "Normalizing data with mean={mean:.4f} and std={std:.4f}",
    "transformation": "Transforming data using {method} method",
    "data_shape": "Data shape: {shape} with range: min={min:.4f}, max={max:.4f}",
    "missing_values": "Detected {count} missing values in {component}. Strategy: {strategy}",
}

RESULT_MESSAGES = {
    "final_loss": "Final loss: {loss:.6f} after {iterations} iterations",
    "reconstruction_error": "Reconstruction error: {error:.6f} ({metric})",
    "explained_variance": "Explained variance: {variance:.4f}",
    "evaluation_complete": "Evaluation complete. Key metrics: {metrics}",
}

PERFORMANCE_MESSAGES = {
    "memory_usage": "Peak memory usage: {memory_mb:.2f} MB",
    "time_breakdown": "Time breakdown: {breakdown}",
    "computation_device": "Computation performed on: {device}",
}


def get_message(category: str, key: str, **kwargs) -> str:
    """
    Get a formatted message from the template categories.

    Args:
        category: The message category (e.g., 'INIT_MESSAGES')
        key: The specific message key
        **kwargs: Values to insert into the message template

    Returns:
        The formatted message string
    """
    categories = {
        "init": INIT_MESSAGES,
        "progress": PROGRESS_MESSAGES,
        "warning": WARNING_MESSAGES,
        "error": ERROR_MESSAGES,
        "data": DATA_MESSAGES,
        "result": RESULT_MESSAGES,
        "performance": PERFORMANCE_MESSAGES,
    }

    template = categories.get(category, {}).get(key, "")
    if not template:
        return f"Unknown message: {category}.{key}"

    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"Missing key in message template {category}.{key}: {e}"
    except Exception as e:
        return f"Error formatting message {category}.{key}: {e}"
