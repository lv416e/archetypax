# ArchetypAX Tests

This directory contains the comprehensive test suite for the ArchetypAX library, ensuring robustness, correctness, and performance across all components.

## Running Tests

Execute all tests with the following command from the project root directory:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_models.py
```

For verbose output with detailed test results:

```bash
pytest -v
```

To generate a test coverage report:

```bash
pytest --cov=archetypax
```

For a more detailed HTML coverage report:

```bash
pytest --cov=archetypax --cov-report=html
```

## Test Structure

The test suite is organized to mirror the package structure:

- `test_models.py`: Validates all model implementations (ArchetypalAnalysis, ImprovedArchetypalAnalysis, BiarchetypalAnalysis)
- `test_tools.py`: Ensures correctness of evaluation, interpretation, visualization, and tracking utilities
- `conftest.py`: Contains pytest fixtures and common utilities for testing

## Test Data

The tests use a combination of:

- Synthetic data generated with scikit-learn's `make_blobs` for clustering validation
- Geometric data like simplexes for validating convex hull properties
- Specifically crafted extreme cases to test numerical stability
- Random data with controlled seeds for reproducibility

## Test Categories

Tests are organized into several categories:

1. **Functional tests**: Verify that each component works as expected
2. **Integration tests**: Ensure components work correctly together
3. **Edge case tests**: Validate proper handling of extreme or unusual inputs
4. **Performance tests**: Check that operations complete within expected time bounds
5. **Stability tests**: Verify numerical stability across optimization steps

## Dependencies

The following packages are required to run the tests:

- pytest (core testing framework)
- pytest-cov (for coverage analysis)
- numpy (for numerical operations)
- scikit-learn (for generating test data)
- jax (for testing JAX-specific functionality)

## Contributing Tests

When contributing to ArchetypAX, please ensure:

1. All new features are accompanied by appropriate tests
2. Existing tests pass with your changes
3. Test coverage remains high (preferably >80%)
4. Tests include both positive and negative cases
