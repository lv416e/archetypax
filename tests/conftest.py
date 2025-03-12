"""Configuration file for pytest.

This module ensures that the parent directory is added to the Python path,
allowing tests to import the archetypax package regardless of how pytest is invoked.
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
