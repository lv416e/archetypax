"""Utility modules for Archetypal Analysis."""

# Make modules accessible through the tools namespace
import sys
import types
from typing import Any

from . import evaluation, interpret, visualization

# Expose key classes at the tools level
from .evaluation import ArchetypalAnalysisEvaluator
from .interpret import ArchetypalAnalysisInterpreter
from .visualization import ArchetypalAnalysisVisualizer
