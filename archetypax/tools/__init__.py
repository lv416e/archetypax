"""Tools for extracting insights from Archetypal Analysis results.

This package provides specialized utilities that transform archetypal models from
mathematical abstractions into actionable insights. These tools address the critical
gap between model fitting and practical application by enabling:

1. Rigorous evaluation of model quality and reliability
2. Interpretable translation of abstract archetypes into domain-specific meaning
3. Compelling visualization that communicates patterns to technical and non-technical audiences
4. Systematic tracking of archetype evolution during model training

These capabilities are essential for deriving value from archetypal analysis,
particularly in exploratory data analysis, scientific research, and
data-driven decision making contexts.

Components:
    evaluation: Quantitative assessment of model quality and fit characteristics
    interpret: Semantic analysis of archetypes and their real-world significance
    visualization: Advanced plotting and visual analysis techniques
    tracker: Progressive monitoring of archetype development during training

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

from . import evaluation, interpret, tracker, visualization

# Expose key classes at the tools level
from .evaluation import ArchetypalAnalysisEvaluator
from .interpret import ArchetypalAnalysisInterpreter
from .tracker import ArchetypeTracker
from .visualization import ArchetypalAnalysisVisualizer
