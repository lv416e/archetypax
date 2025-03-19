"""GPU-accelerated Archetypal Analysis implementation using JAX."""

__version__ = "0.1.0.dev2"

# Maintain backward compatibility with existing code
import sys
import types
from typing import Any

from . import logger, models, tools
from .logger import get_logger, get_message

# Direct imports for simplified usage
from .models.archetypes import ImprovedArchetypalAnalysis
from .models.base import ArchetypalAnalysis
from .models.biarchetypes import BiarchetypalAnalysis
from .models.sparse_archetypes import SparseArchetypalAnalysis
from .tools.evaluation import ArchetypalAnalysisEvaluator
from .tools.interpret import ArchetypalAnalysisInterpreter
from .tools.visualization import ArchetypalAnalysisVisualizer

# Register legacy import paths - with type annotations for mypy
if not isinstance(models.base, types.ModuleType):
    models.base = types.ModuleType("models.base")  # type: ignore
if not isinstance(models.archetypes, types.ModuleType):
    models.archetypes = types.ModuleType("models.archetypes")  # type: ignore
if not isinstance(models.biarchetypes, types.ModuleType):
    models.biarchetypes = types.ModuleType("models.biarchetypes")  # type: ignore
if not isinstance(tools.evaluation, types.ModuleType):
    tools.evaluation = types.ModuleType("tools.evaluation")  # type: ignore
if not isinstance(tools.visualization, types.ModuleType):
    tools.visualization = types.ModuleType("tools.visualization")  # type: ignore
if not isinstance(tools.interpret, types.ModuleType):
    tools.interpret = types.ModuleType("tools.interpret")  # type: ignore

sys.modules["archetypax.base"] = models.base
sys.modules["archetypax.archetypes"] = models.archetypes
sys.modules["archetypax.biarchetypes"] = models.biarchetypes
sys.modules["archetypax.evaluation"] = tools.evaluation
sys.modules["archetypax.visualization"] = tools.visualization
sys.modules["archetypax.interpret"] = tools.interpret
