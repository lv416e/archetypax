# ArchetypAX Architecture

## Package Structure

ArchetypAX is organized into a modular structure to enhance maintainability and extensibility:

```
archetypax/
├── __init__.py              # Package initialization and exports
├── models/                  # Core model implementations
│   ├── __init__.py          # Models namespace
│   ├── base.py              # Base archetypal analysis implementation
│   ├── archetypes.py        # Improved archetypal analysis implementation
│   ├── biarchetypes.py      # Biarchetypal analysis implementation
│   └── sparse_archetypes.py # Sparse archetypal analysis implementation
└── tools/                   # Utility modules
    ├── __init__.py          # Tools namespace
    ├── evaluation.py        # Evaluation metrics and analysis
    ├── interpret.py         # Interpretation methods
    ├── tracker.py           # Archetype optimization tracking utilities
    └── visualization.py     # Visualization utilities
```

## Design Philosophy

The package follows a modular design with clear separation of concerns:

1. **Core Models (`models/`)**: Contains the fundamental algorithmic implementations of archetypal analysis variants.
   - `base.py`: Provides the foundational implementation with essential functionality
   - `archetypes.py`: Extends the base with improved optimization and features
   - `biarchetypes.py`: Implements the dual-archetype approach for more expressive modeling
   - `sparse_archetypes.py`: Implements the sparse archetypal analysis

2. **Utility Tools (`tools/`)**: Houses supporting functionality for analysis and visualization.
   - `evaluation.py`: Metrics and methods to assess model quality and performance
   - `interpret.py`: Methods to interpret and explain model results
   - `tracker`: Archetypes tracking
   - `visualization.py`: Comprehensive visualization tools for model exploration

## Import Patterns

ArchetypAX supports multiple import patterns for flexibility:

### Direct Class Imports
```python
from archetypax import ArchetypalAnalysis, ImprovedArchetypalAnalysis
```

### Explicit Module Imports
```python
from archetypax.models.base import ArchetypalAnalysis
from archetypax.tools.evaluation import ArchetypalAnalysisEvaluator
```

### Module-Level Imports
```python
from archetypax.models import ArchetypalAnalysis
from archetypax.tools import ArchetypalAnalysisVisualizer
```

## Backward Compatibility

The package maintains backward compatibility with code written for earlier versions through a transparent module aliasing system. This ensures that existing code continues to function without modification while encouraging the adoption of the new, more organized structure for new development.
