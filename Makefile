.PHONY: test lint format clean

test:
	pytest

test-verbose:
	pytest -v

test-coverage:
	pytest --cov=archetypax tests/

lint:
	ruff check archetypax tests

format:
	ruff format archetypax tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "*.eggs" -exec rm -rf {} +
	find . -type d -name "*.pytest_cache" -exec rm -rf {} +

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-examples:
	pip install -e ".[examples]"

install-all:
	pip install -e ".[dev,examples]"
