"""Core model implementations for Archetypal Analysis."""

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
