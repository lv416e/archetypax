#!/usr/bin/env python
"""Test runner script for the archetypax package.

This script provides a convenient way to run all tests for the archetypax package
from the command line. It ensures proper path configuration and passes appropriate
arguments to pytest.
"""

import os
import sys

import pytest

if __name__ == "__main__":
    # Add the project root directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    # Run pytest with verbose output, targeting the tests directory
    sys.exit(pytest.main(["-v", "tests"]))
