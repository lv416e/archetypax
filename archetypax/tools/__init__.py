"""
Utility modules for Archetypal Analysis.

This package provides tools for evaluating, interpreting, and visualizing
results from archetypal analysis models.

Components:
    evaluation: Tools for assessing model performance and quality
    interpret: Utilities for interpreting archetypes and their meaning
    visualization: Functions for plotting and visual analysis of results

Basic Usage:
    from archetypax.tools import ArchetypalAnalysisVisualizer

    # After fitting a model
    visualizer = ArchetypalAnalysisVisualizer(model)

    # Plot archetypes
    visualizer.plot_archetypes()

    # Visualize data in archetypal space
    visualizer.plot_simplex_embedding(data)
"""

# Make modules accessible through the tools namespace
import sys
import types
from typing import Any

from . import evaluation, interpret, visualization

# Expose key classes at the tools level
from .evaluation import ArchetypalAnalysisEvaluator
from .interpret import ArchetypalAnalysisInterpreter
from .visualization import ArchetypalAnalysisVisualizer
