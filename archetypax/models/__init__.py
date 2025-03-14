"""Core model implementations for Archetypal Analysis."""

# Make modules accessible through the models namespace
import sys
import types
from typing import Any

from . import archetypes, base, biarchetypes
from .archetypes import ImprovedArchetypalAnalysis

# Expose key classes at the models level
from .base import ArchetypalAnalysis
from .biarchetypes import BiarchetypalAnalysis
