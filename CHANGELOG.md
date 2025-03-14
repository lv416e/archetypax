# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Initial implementation of ArchetypalAnalysis class
- JAX-based optimization with GPU support
- k-means++ style initialization for better convergence
- Entropy regularization for more uniform weight distributions
- Project structure and documentation
- Comprehensive CI/CD workflows for testing, documentation, and releases
- Improved test coverage with separation of slow and fast tests
- BiarchetypalAnalysis implementation for dual archetype modeling

### Changed
- Refactored visualization tools for better performance
- Enhanced evaluation metrics for model assessment
- Updated documentation structure for better readability

### Fixed
- Error message consistency in test assertions
- Visualization issues in high-dimensional data
- Performance bottlenecks in weight calculation

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
