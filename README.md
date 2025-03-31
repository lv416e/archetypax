# ArchetypAX

**ArchetypAX** - Hardware-accelerated Archetypal Analysis implementation using JAX

<!--
Repository topics for better discoverability:
archetypal-analysis, jax, machine-learning, dimensionality-reduction, convex-hull-optimization
-->

> A Python library for Hardware-accelerated Archetypal Analysis using JAX.<br>Discover extreme patterns in your data with high-performance matrix factorization, convex hull optimization, and interpretable results.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/archetypax.svg?cache=no)](https://pypi.org/project/archetypax/)
[![Tests](https://github.com/lv416e/archetypax/actions/workflows/tests.yml/badge.svg)](https://github.com/lv416e/archetypax/actions/workflows/tests.yml)
[![Lint](https://github.com/lv416e/archetypax/actions/workflows/lint.yml/badge.svg)](https://github.com/lv416e/archetypax/actions/workflows/lint.yml)
[![Docs](https://github.com/lv416e/archetypax/actions/workflows/docs.yml/badge.svg)](https://github.com/lv416e/archetypax/actions/workflows/docs.yml)
[![Release](https://github.com/lv416e/archetypax/actions/workflows/release.yml/badge.svg)](https://github.com/lv416e/archetypax/actions/workflows/release.yml)

## Overview

`archetypax` is a high-performance implementation of Archetypal Analysis (AA) that leverages JAX for GPU acceleration.<br>
Archetypal Analysis is a matrix factorization technique that represents data points<br>
as convex combinations of extreme points (archetypes) found within the data's convex hull.<br>

Unlike traditional dimensionality reduction techniques like PCA which finds a basis of orthogonal components,<br>
AA finds interpretable extremal points that often correspond to meaningful prototypes in the data.<br>

## Features

- 🚀 **GPU/TPU Acceleration**: Utilizes JAX for fast computation on GPUs and TPUs
- 🔍 **Interpretable Results**: Finds meaningful archetypes that represent extremal patterns in data
- 🧠 **Smart Initialization**: Implements multiple strategic initialization methods including k-means++ and directional initialization
- 🛠️ **Numerical Stability**: Employs sophisticated techniques for improved convergence and stability
- 📊 **scikit-learn Compatible API**: Fully implements the familiar fit/transform interface for seamless integration
- 🔄 **Dual-perspective Analysis**: Offers BiarchetypalAnalysis for simultaneous row and column pattern discovery
- 📈 **Advanced Tracking**: Provides ArchetypeTracker for monitoring optimization trajectories and convergence dynamics
- 🎯 **Comprehensive Tooling**: Features extensive evaluation, interpretation, and visualization capabilities
- 📋 **Thorough Documentation**: Includes detailed rationales explaining why each method and parameter matters

## Related Projects and Techniques

ArchetypAX can be used alongside or compared with these related approaches:

- **PCA**: Principal Component Analysis finds orthogonal directions of maximum variance
- **NMF**: Non-negative Matrix Factorization decomposes data into non-negative components
- **k-means**: Clustering technique that partitions data into k clusters
- **JAX Ecosystem**: Compatible with JAX-based machine learning frameworks like Flax and Haiku
- **scikit-learn**: Follows similar API conventions, allowing easy integration

## Installation

```bash
# Using pip
pip install archetypax
```

or from GitHub:

```bash
pip install git+https://github.com/lv416e/archetypax.git
```

### Using uv

```bash
uv pip install archetypax
# or from GitHub
uv pip install git+https://github.com/lv416e/archetypax.git
```

### Using Poetry

``` bash
poetry add archetypax

# or from GitHub
poetry add git+https://github.com/lv416e/archetypax.git
```

### Requirements
- Python 3.10+
- JAX
- NumPy
- scikit-learn

## Quick Start

```python
import numpy as np
from archetypax import ArchetypalAnalysis

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 10)

# Initialize and fit the model
model = ArchetypalAnalysis(n_archetypes=5)
weights = model.fit_transform(X)

# Get the archetypes
archetypes = model.archetypes

# Reconstruct the data
X_reconstructed = model.reconstruct()

# Calculate reconstruction error
mse = np.mean((X - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {mse:.6f}")
```

## Import Patterns

ArchetypAX supports multiple import patterns for flexibility:

### Direct Class Imports (Recommended)

```python
from archetypax import ArchetypalAnalysis, ImprovedArchetypalAnalysis, BiarchetypalAnalysis, ArchetypeTracker
```

### Explicit Module Imports

```python
from archetypax.models.base import ArchetypalAnalysis
from archetypax.models.biarchetypes import BiarchetypalAnalysis
from archetypax.tools.evaluation import ArchetypalAnalysisEvaluator
from archetypax.tools.tracker import ArchetypeTracker
```

### Module-Level Imports

```python
from archetypax.models import ArchetypalAnalysis
from archetypax.tools import ArchetypalAnalysisVisualizer, ArchetypeTracker
```

## Documentation

### Parameters

#### ArchetypalAnalysis / ImprovedArchetypalAnalysis
- `n_archetypes`: Number of archetypes to find
- `max_iter`: Maximum number of iterations (default: 500)
- `tol`: Convergence tolerance (default: 1e-6)
- `random_seed`: Random seed for initialization (default: 42)
- `learning_rate`: Learning rate for optimizer (default: 0.001)
- `lambda_reg`: Regularization strength for weight distribution (default: 0.01)
- `normalize`: Whether to normalize features before fitting (default: False)
- `projection_method`: Method for projecting archetypes ("cbap", "convex_hull", "knn")
- `projection_alpha`: Blending coefficient for boundary projection (default: 0.1)
- `archetype_init_method`: Initialization strategy ("directional", "kmeans++", "qhull")

#### BiarchetypalAnalysis

- `n_row_archetypes`: Number of archetypes in observation space
- `n_col_archetypes`: Number of archetypes in feature space
- `max_iter`: Maximum number of iterations (default: 500)
- `tol`: Convergence tolerance (default: 1e-6)
- `random_seed`: Random seed for initialization (default: 42)
- `learning_rate`: Learning rate for optimizer (default: 0.001)
- `projection_method`: Method for projecting archetypes (default: "default")
- `lambda_reg`: Regularization strength for entropy terms (default: 0.01)

### Methods

- `fit(X)`: Fit the model to the data
- `transform(X)`: Transform new data to archetype weights
- `fit_transform(X)`: Fit the model and transform the data
- `reconstruct(X)`: Reconstruct data from archetype weights
- `get_loss_history()`: Get the loss history from training

## Examples

### Visualizing Archetypes in 2D Data

```python
import numpy as np
import matplotlib.pyplot as plt
from archetypax import ImprovedArchetypalAnalysis
from archetypax.tools.visualization import ArchetypalAnalysisVisualizer

# Generate some interesting 2D data (a triangle with points inside)
n_samples = 500
vertices = np.array([[0, 0], [1, 0], [0.5, 0.866]])
weights = np.random.dirichlet(np.ones(3), size=n_samples)
X = weights @ vertices

# Fit archetypal analysis with 3 archetypes
model = ImprovedArchetypalAnalysis(n_archetypes=3, archetype_init_method="directional")
model.fit(X)

# Plot original data and archetypes
plt.figure(figsize=(10, 8))
ArchetypalAnalysisVisualizer.plot_archetypes_2d(model, X)
plt.title("Archetypal Analysis of 2D Data")
plt.show()
```

### Using Biarchetypal Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from archetypax import BiarchetypalAnalysis
from archetypax.tools.visualization import ArchetypalAnalysisVisualizer

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(500, 5)

# Initialize and fit the model with row and column archetypes
model = BiarchetypalAnalysis(
    n_row_archetypes=2,   # Number of archetypes in observation space
    n_col_archetypes=2,   # Number of archetypes in feature space
    max_iter=500,
    random_seed=42
)
model.fit(X)

# Get both sets of archetypes
row_archetypes, col_archetypes = model.get_all_archetypes()
print("Row archetypes shape:", row_archetypes.shape)
print("Column archetypes shape:", col_archetypes.shape)

# Get both sets of weights
row_weights, col_weights = model.get_all_weights()
print("Row weights shape:", row_weights.shape)
print("Column weights shape:", col_weights.shape)

# Reconstruct data using biarchetypes
X_reconstructed = model.reconstruct()
mse = np.mean((X - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {mse:.6f}")
```

### Tracking Archetype Evolution

```python
import numpy as np
import matplotlib.pyplot as plt
from archetypax import ArchetypeTracker

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 10)

# Initialize the tracker
tracker = ArchetypeTracker(
    n_archetypes=3,
    max_iter=300,
    random_seed=42
)

# Fit the model while tracking archetype movement
tracker.fit(X)

# Visualize the archetype movement trajectory
tracker.visualize_movement()

# Visualize boundary proximity over iterations
tracker.visualize_boundary_proximity()
```

## How It Works

Archetypal Analysis solves the following optimization problem:

Given a data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ with n samples and d features, find k archetypes $\mathbf{A} \in \mathbb{R}^{k \times d}$ and weights $\mathbf{W} \in \mathbb{R}^{n \times k}$ such that:

$$
\text{minimize} \ \| \mathbf{X} - \mathbf{W} \cdot \mathbf{A} \|^2_{\text{F}}
$$

subject to:

- $\mathbf{W}$ is non-negative
- Each row of $\mathbf{W}$ sums to 1 (simplex constraint)
- $\mathbf{A}$ lies within the convex hull of $\mathbf{X}$

The biarchetypal extension solves a more complex factorization:

$$
\mathbf{X} \approx \mathbf{\alpha} \cdot \mathbf{\beta} \cdot \mathbf{X} \cdot \mathbf{\theta} \cdot \mathbf{\gamma}
$$

This implementation uses JAX's automatic differentiation and optimization tools to efficiently solve these problems on GPUs. It also incorporates several advanced enhancements:

1. **Strategic initialization methods** including directional initialization, k-means++ style, and convex hull approximation
2. **Intelligent regularization techniques** to promote interpretable weight distributions
3. **Advanced projection methods** including adaptive convex boundary approximation (CBAP)
4. **Sophisticated numerical stability safeguards** throughout the optimization process
5. **Comprehensive trajectory tracking** for monitoring convergence dynamics

## Citation

If you use this package in your research, please cite:

```
@software{archetypax2025,
  author = {mary},
  title = {archetypax: GPU-accelerated Archetypal Analysis using JAX},
  year = {2025},
  url = {https://github.com/lv416e/archetypax}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
