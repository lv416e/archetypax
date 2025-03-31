# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed
- Relocated `ArchetypeTracker` class from `models/archetypes.py` to `tools/tracker.py` for improved code organization while maintaining backward compatibility
- Enhanced documentation across core modules with "why-focused" docstrings that explain rationale and importance
- Improved module docstrings in `models/__init__.py` with detailed explanations of when to use each model variant
- Refined class and method docstrings in `ImprovedArchetypalAnalysis` with parameter impact explanations and usage guidance
- Enriched docstrings for key public methods (`transform`, `fit`, `fit_transform`, `project_archetypes`, `loss_function`) with clear explanations of their purpose and significance
- Upgraded documentation in `tools/` modules to better explain analysis and visualization capabilities
- Standardized docstring quality with native-level English throughout the codebase

### Fixed


## [0.1.0] - 2025/03/21

### Added
- Introduction of the innovative `ArchetypeTracker` class for precise monitoring of archetype evolution during optimization
- Comprehensive trajectory analysis tools for visualizing archetype movement patterns
- Boundary proximity tracking with historical metrics for evolutionary pattern analysis
- Time-series visualization of archetype convergence behaviors and stability indicators
- Interactive Jupyter notebook interfaces for real-time archetype evolution exploration
- Gradient flow visualization frameworks for optimization trajectory comprehension
- Multi-dimensional scaling techniques specifically adapted for high-dimensional archetype visualization

### Changed
- Dynamically adjusted blending coefficients that respond to optimization phase and loss trajectory
- Sophisticated parameter adaptation mechanisms based on iteration progress metrics
- Enhanced logging system with detailed diagnostics for optimization trajectory analysis
- Customized memory management systems for efficient archetype history tracking
- Refined gradient clipping strategies with progressive threshold adjustments

### Fixed
- Edge case handling for extreme archetype positioning in high-dimensional feature spaces
- Memory leakage issues in historical trajectory storage mechanisms
- Boundary condition anomalies during complex projection calculations
- Numerical precision challenges in gradient calculations for closely positioned archetypes
- Thread safety concerns in parallel computation environments


## [0.1.0.dev2] - 2025/03/17

### Added
- Initial implementation of ArchetypalAnalysis class
- JAX-based optimization with GPU support
- k-means++ style initialization for better convergence
- Entropy regularization for more uniform weight distributions
- Project structure and documentation
- Comprehensive CI/CD workflows for testing, documentation, and releases
- Improved test coverage with separation of slow and fast tests
- BiarchetypalAnalysis implementation for dual archetype modeling
- ImprovedArchetypalAnalysis with enhanced convergence properties
- ArchetypalAnalysisInterpreter for model interpretation and insights
- BiarchetypalAnalysisEvaluator for dual archetype model assessment
- BiarchetypalAnalysisVisualizer with specialized dual archetype visualization tools
- BiarchetypalAnalysisInterpreter for dual archetype model interpretation
- Elbow method for determining optimal number of archetypes
- Optimal biarchetype combination suggestion functionality
- Interpretability heatmap visualization for model comparison
- Dual membership heatmap visualization for biarchetypal analysis
- 2D visualization tools for biarchetypal models
- Dual simplex visualization for biarchetypal analysis
- Feature importance analysis for archetype characterization
- Weight diversity evaluation metrics
- Archetype separation assessment tools
- Dominant archetype purity evaluation
- Clustering metrics for model quality assessment
- Feature distinctiveness evaluation
- Sparsity coefficient calculation for model interpretability
- Cluster purity assessment for validation
- Multiple archetype projection methods (boundary, convex hull, KNN)
- L-BFGS optimization for weight transformation
- Scikit-learn compatible API with BaseEstimator and TransformerMixin
- Simultaneous row and column archetype learning in BiarchetypalAnalysis
- Regularization options for controlling model complexity
- Efficient JAX JIT compilation for all core computational functions
- Comprehensive model parameter validation and error handling

### Changed
- Refactored visualization tools for better performance
- Enhanced evaluation metrics for model assessment
- Updated documentation structure for better readability
- Improved comprehensive evaluation reporting functionality
- Enhanced evaluation metrics for biarchetypal models
- Upgraded mixture effect visualization capabilities
- Refined archetype profile visualization for better interpretability
- Optimized loss function with improved regularization terms
- Enhanced archetype initialization strategy for faster convergence
- Improved projection methods for better boundary representation
- Streamlined API for consistent model interaction across classes
- Upgraded optimization process with adaptive learning rates
- Enhanced numerical stability in matrix operations
- Improved memory efficiency for large dataset processing

### Fixed
- Error message consistency in test assertions
- Visualization issues in high-dimensional data
- Performance bottlenecks in weight calculation
- Handling of invalid reconstruction error metrics
- Consistency in error handling for unfitted models
- Reconstruction error calculation for biarchetypal models
- Display issues with dual membership heatmaps for high-dimensional data
- Numerical instability in simplex projection algorithm
- Convergence issues with certain initialization conditions
- Memory leaks in iterative optimization procedures
- Edge cases in archetype boundary projection
- Inconsistent behavior with zero-variance features
- Gradient calculation errors in specific corner cases
- Type conversion issues between NumPy and JAX arrays
- Thread safety issues in parallel computation


## [0.1.0.dev1] - 2025-03-11

- Initial pre-release

## Maintaining this Changelog

This CHANGELOG is automatically used by the GitHub Actions release workflow to generate release notes. To ensure proper categorization in release notes:

1. Always update the "Unreleased" section with your changes
2. Use appropriate subsections (Added, Changed, Fixed, etc.)
3. When creating a new release tag (e.g., `v0.1.0`), the release workflow will:
   - Create a new GitHub release
   - Generate release notes based on PR labels and this CHANGELOG
   - Publish the package to PyPI

When making PRs, use labels like `feature`, `enhancement`, `bug`, `documentation`, etc., to help with automatic release note generation.
