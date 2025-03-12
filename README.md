# ArchetypAX

**ArchetypAX** - Hardware-accelerated Archetypal Analysis implementation using JAX

[![PyPI version](https://badge.fury.io/py/archetypax.svg)](https://badge.fury.io/py/archetypax)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/lv416e/archetypax/actions/workflows/pytest.yml/badge.svg)](https://github.com/lv416e/archetypax/actions/workflows/pytest.yml)
[![Lint](https://github.com/lv416e/archetypax/actions/workflows/lint.yml/badge.svg)](https://github.com/lv416e/archetypax/actions/workflows/lint.yml)

## Overview

`archetypax` is a high-performance implementation of Archetypal Analysis (AA) that leverages JAX for GPU acceleration.<br>
Archetypal Analysis is a matrix factorization technique that represents data points<br>
as convex combinations of extreme points (archetypes) found within the data's convex hull.<br>

Unlike traditional dimensionality reduction techniques like PCA which finds a basis of orthogonal components, <br>
AA finds interpretable extremal points that often correspond to meaningful prototypes in the data.

## Features

- 🚀 **GPU/TPU Acceleration**: Utilizes JAX for fast computation on GPUs
- 🔍 **Interpretable Results**: Finds meaningful archetypes that represent extremal patterns in data
- 🧠 **Smart Initialization**: Uses k-means++ style initialization for better convergence
- 🛠️ **Numerical Stability**: Implements various techniques for improved stability
- 📊 **scikit-learn Compatible API**: Implements the familiar fit/transform interface

## Installation

```bash
pip install archetypax
```

### Requirements
- Python 3.12.8+
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

## Documentation

### Parameters

- `n_archetypes`: Number of archetypes to find
- `max_iter`: Maximum number of iterations (default: 500)
- `tol`: Convergence tolerance (default: 1e-6)
- `random_seed`: Random seed for initialization (default: 42)
- `learning_rate`: Learning rate for optimizer (default: 0.001)

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
from archetypax import ArchetypalAnalysis

# Generate some interesting 2D data (a triangle with points inside)
n_samples = 500
vertices = np.array([[0, 0], [1, 0], [0.5, 0.866]])
weights = np.random.dirichlet(np.ones(3), size=n_samples)
X = weights @ vertices

# Fit archetypal analysis with 3 archetypes
model = ArchetypalAnalysis(n_archetypes=3)
model.fit(X)

# Plot original data and archetypes
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Data points")
plt.scatter(
    model.archetypes[:, 0],
    model.archetypes[:, 1],
    color='red',
    s=100,
    marker='*',
    label="Archetypes"
)

# Draw the archetypal convex hull
hull_points = np.vstack([model.archetypes, model.archetypes[0]])
plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', label="Archetype convex hull")

plt.legend()
plt.title("Archetypal Analysis of 2D Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(alpha=0.3)
plt.show()
```

## How It Works

Archetypal Analysis solves the following optimization problem:

Given a data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ with n samples and d features, find k archetypes $\mathbf{Z} \in \mathbb{R}^{k \times p}$ and weights $\mathbf{w} \in \mathbb{R}^{n \times k}$ such that:

$$
\text{minimize} \| \mathbf{X} - \mathbf{w} \cdot \mathbf{Z} \|^2_{\text{F}}
$$

subject to:

- $\mathbf{w}$ is non-negative
- Each row of $\mathbf{w}$ sums to 1 (simplex constraint)
- $\mathbf{Z}$ lies within the convex hull of $\mathbf{X}$

This implementation uses JAX's automatic differentiation and optimization tools to efficiently solve this problem on GPUs. It also incorporates several enhancements:

1. **k-means++ style initialization** for better initial archetype positions
2. **Entropy regularization** to promote more uniform weight distributions
3. **Soft archetype projection** using k-nearest neighbors for improved stability
4. **Gradient clipping** to prevent numerical issues

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
