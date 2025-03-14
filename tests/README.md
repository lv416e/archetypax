# Archetypax Tests

This directory contains unit tests for the Archetypax library.

## Running Tests

Execute the following command from the project root directory:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_models.py
```

For verbose output:

```bash
pytest -v
```

To generate a test coverage report:

```bash
pytest --cov=archetypax
```

## Test Structure

- `test_models.py`: Tests for model classes
- `test_tools.py`: Tests for tool classes (evaluation, interpretation, visualization)
- `conftest.py`: pytest configuration file

## Test Data

The tests use synthetic data generated with scikit-learn's `make_blobs` function, ensuring reproducible test results.

## Dependencies

The following packages are required to run the tests:

- pytest
- pytest-cov (for coverage reports)
- numpy
- scikit-learn
