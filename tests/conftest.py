"""Configuration for pytest test suite."""

import os
import sys

import pytest

# Add the parent directory to sys.path to enable imports from the archetypax package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is specified."""
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
            item.add_marker(skip_slow)
