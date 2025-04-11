# Migration Guide

This guide outlines major changes in ArchetypAX versions and provides recommendations for adapting your code.

## Version 0.2.0: Relocation of ArchetypeTracker

With version 0.2.0, we have refined our structure by relocating the `ArchetypeTracker` class to better reflect its purpose as an analytical tool rather than a core model.

### Key Changes

The `ArchetypeTracker` class has been moved from `archetypax.models.archetypes` to the new `archetypax.tools.tracker` module:

```
archetypax/
├── models/                  # Core model implementations
│   ├── base.py              # Base archetypal analysis implementation
│   ├── archetypes.py        # Improved archetypal analysis implementation (ArchetypeTracker relocated)
│   └── biarchetypes.py      # Biarchetypal analysis implementation
│   └── sparse_archetypes.py # Sparse archetypal analysis implementation
└── tools/                   # Utility and analysis modules
    ├── evaluation.py        # Evaluation metrics and analysis
    ├── interpret.py         # Interpretation methods
    ├── tracker.py           # Archetype optimization tracking utilities (NEW LOCATION)
    └── visualization.py     # Visualization utilities
```

This move better reflects `ArchetypeTracker`'s role as a specialized tool for analyzing and visualizing the optimization dynamics of archetypal models, rather than being a core model implementation itself.

### Backward Compatibility

**Important**: Full backward compatibility is maintained for this change. All existing code that imports `ArchetypeTracker` from its previous location (`archetypax.models.archetypes`) will continue to function without modification due to internal aliasing.

### Recommended Import Patterns for ArchetypeTracker

While existing import patterns will continue to work, we strongly recommend adopting the new patterns for clarity and future-proofing:

#### Legacy Import (Still Supported)

```python
# Old import path (still works but discouraged)
from archetypax.models.archetypes import ArchetypeTracker
```

#### Recommended New Imports

```python
# Option 1: Module-level import
from archetypax.tools import ArchetypeTracker

# Option 2: Explicit import (preferred for clarity)
from archetypax.tools.tracker import ArchetypeTracker
```

### Timeline for ArchetypeTracker

The legacy import path (`archetypax.models.archetypes.ArchetypeTracker`) will be maintained indefinitely via aliasing to ensure backward compatibility. However, we strongly encourage updating imports to the new path in all new and existing code for better clarity and alignment with the intended package structure.

---

## Version 0.1.0: Transition to Modular Structure

With version 0.1.0, ArchetypAX adopted a more modular structure to enhance maintainability and extensibility.

### Key Changes

The package was reorganized into a modular structure:

```
archetypax/
├── models/                  # Core model implementations
│   ├── base.py              # Base archetypal analysis implementation
│   ├── archetypes.py        # Improved archetypal analysis implementation
│   └── biarchetypes.py      # Biarchetypal analysis implementation
└── tools/                   # Utility modules
    ├── evaluation.py        # Evaluation metrics and analysis
    ├── visualization.py     # Visualization utilities
    └── interpret.py         # Interpretation methods
```

### Backward Compatibility

**Important**: Full backward compatibility is maintained. All existing code will continue to function without modification.

The package implements a transparent module aliasing system that ensures imports using the previous structure will continue to work seamlessly.

### Recommended Import Patterns

While existing import patterns will continue to work, we recommend adopting one of the following patterns for new code:

#### Direct Class Imports (Recommended for Most Users)

```python
from archetypax import ArchetypalAnalysis, ImprovedArchetypalAnalysis
```

This approach provides the simplest interface for most use cases.

#### Explicit Module Imports (Recommended for Advanced Users)

```python
from archetypax.models.base import ArchetypalAnalysis
from archetypax.tools.evaluation import ArchetypalAnalysisEvaluator
```

This approach provides explicit clarity about which module each component comes from.

#### Module-Level Imports

```python
from archetypax.models import ArchetypalAnalysis
from archetypax.tools import ArchetypalAnalysisVisualizer
```

This approach offers a balance between brevity and clarity.

### Examples

#### Before

```python
from archetypax import ArchetypalAnalysis
from archetypax.evaluation import ArchetypalAnalysisEvaluator
from archetypax.visualization import ArchetypalAnalysisVisualizer
```

#### After (Recommended)

```python
from archetypax import ArchetypalAnalysis
from archetypax.tools import ArchetypalAnalysisEvaluator, ArchetypalAnalysisVisualizer
```

Or for more explicit imports:

```python
from archetypax import ArchetypalAnalysis
from archetypax.tools.evaluation import ArchetypalAnalysisEvaluator
from archetypax.tools.visualization import ArchetypalAnalysisVisualizer
```

### Benefits of the Modular Structure

1. **Improved Organization**: Related functionality is grouped together
2. **Enhanced Maintainability**: Clearer separation of concerns
3. **Better Extensibility**: Easier to add new components
4. **Reduced Namespace Pollution**: More explicit imports reduce the risk of name collisions

### Timeline

The legacy import paths will be maintained indefinitely to ensure backward compatibility. However, we encourage adopting the new import patterns for all new code.
