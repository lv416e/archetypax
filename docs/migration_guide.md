# Migration Guide

## Transitioning to the Modular Structure

With version 0.1.0, ArchetypAX has adopted a more modular structure to enhance maintainability and extensibility. This guide outlines the changes and provides recommendations for transitioning existing code.

## Key Changes

The package has been reorganized into a modular structure:

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

## Backward Compatibility

**Important**: Full backward compatibility is maintained. All existing code will continue to function without modification.

The package implements a transparent module aliasing system that ensures imports using the previous structure will continue to work seamlessly.

## Recommended Import Patterns

While existing import patterns will continue to work, we recommend adopting one of the following patterns for new code:

### Direct Class Imports (Recommended for Most Users)

```python
from archetypax import ArchetypalAnalysis, ImprovedArchetypalAnalysis
```

This approach provides the simplest interface for most use cases.

### Explicit Module Imports (Recommended for Advanced Users)

```python
from archetypax.models.base import ArchetypalAnalysis
from archetypax.tools.evaluation import ArchetypalAnalysisEvaluator
```

This approach provides explicit clarity about which module each component comes from.

### Module-Level Imports

```python
from archetypax.models import ArchetypalAnalysis
from archetypax.tools import ArchetypalAnalysisVisualizer
```

This approach offers a balance between brevity and clarity.

## Examples

### Before

```python
from archetypax import ArchetypalAnalysis
from archetypax.evaluation import ArchetypalAnalysisEvaluator
from archetypax.visualization import ArchetypalAnalysisVisualizer
```

### After (Recommended)

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

## Benefits of the New Structure

1. **Improved Organization**: Related functionality is grouped together
2. **Enhanced Maintainability**: Clearer separation of concerns
3. **Better Extensibility**: Easier to add new components
4. **Reduced Namespace Pollution**: More explicit imports reduce the risk of name collisions

## Timeline

The legacy import paths will be maintained indefinitely to ensure backward compatibility. However, we encourage adopting the new import patterns for all new code.
