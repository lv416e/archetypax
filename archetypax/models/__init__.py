"""
Core model implementations for Archetypal Analysis.

This module provides various implementations of Archetypal Analysis algorithms,
including standard, improved, sparse, and bi-archetypal analysis variants.

Available Models:
    ArchetypalAnalysis: Base implementation of archetypal analysis
    ImprovedArchetypalAnalysis: Enhanced version with better initialization and convergence
    SparseArchetypalAnalysis: Implementation enforcing sparsity in archetype coefficients
    BiarchetypalAnalysis: Dual-direction archetypal analysis

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
