"""
Core model implementations for Archetypal Analysis.

This module provides specialized implementations of Archetypal Analysis algorithms,
each addressing specific analytical challenges and use cases. Archetypal Analysis
discovers extreme patterns in data that serve as the building blocks for representing
all observations as convex combinations.

Available Models:
    ArchetypalAnalysis:
        Foundational implementation suitable for low-dimensional datasets
        or initial exploration when computational efficiency matters

    ImprovedArchetypalAnalysis:
        Enhanced version with advanced initialization strategies,
        robust optimization, and boundary projection techniques -
        recommended for most applications due to superior stability
        and convergence properties

    SparseArchetypalAnalysis:
        Implementation enforcing feature sparsity in archetypes -
        essential for high-dimensional data where interpretability
        is a priority and feature selection is desirable

    BiarchetypalAnalysis:
        Dual-direction analysis revealing patterns in both observation
        and feature spaces simultaneously - ideal for datasets where
        understanding relationships between features is as important
        as clustering observations

Basic Usage:
    from archetypax.models import ArchetypalAnalysis

    model = ArchetypalAnalysis(n_archetypes=5)
    model.fit(data)
    archetypes = model.get_archetypes()
"""

# Make modules accessible through the models namespace
import sys
import types
from typing import Any

from . import archetypes, base, biarchetypes, sparse_archetypes
from .archetypes import ArchetypeTracker, ImprovedArchetypalAnalysis

# Expose key classes at the models level
from .base import ArchetypalAnalysis
from .biarchetypes import BiarchetypalAnalysis
from .sparse_archetypes import SparseArchetypalAnalysis
