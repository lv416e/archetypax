"""Unit tests for the ImprovedArchetypalAnalysis class.

This module contains comprehensive tests for the ImprovedArchetypalAnalysis implementation,
verifying its initialization, fitting, transformation, and reconstruction capabilities.
Tests ensure that the model adheres to archetypal analysis constraints and produces
mathematically valid results.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from archetypax import ImprovedArchetypalAnalysis


@pytest.fixture
def sample_data():
    """Generate synthetic data for testing purposes.

    Returns:
        numpy.ndarray: A synthetic dataset with 100 samples, 5 features, and 4 distinct clusters.
    """
    X, _ = make_blobs(n_samples=100, n_features=5, centers=4, cluster_std=1.0, random_state=42)
    return X


def test_initialization():
    """Verify proper initialization of the ImprovedArchetypalAnalysis model.

    This test ensures that all model parameters are correctly set during initialization
    and that uninitialized attributes are properly set to None.
    """
    model = ImprovedArchetypalAnalysis(n_archetypes=3)
    assert model.n_archetypes == 3
    assert model.max_iter == 500
    assert model.tol == 1e-6
    assert model.learning_rate == 0.001
    assert model.archetypes is None
    assert model.weights is None


def test_fit(sample_data):
    """Verify the model's fitting process and resulting attributes.

    This test ensures that after fitting:
    1. The model correctly identifies archetypes
    2. The weights matrix satisfies simplex constraints
    3. The dimensions of resulting matrices are correct

    Args:
        sample_data: Synthetic dataset fixture
    """
    model = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    model.fit(sample_data)

    # Verify that archetypes and weights are properly initialized
    assert model.archetypes is not None
    assert model.weights is not None

    # Verify correct dimensions of resulting matrices
    assert model.archetypes.shape == (3, 5)  # n_archetypes x n_features
    assert model.weights.shape == (100, 3)  # n_samples x n_archetypes
    assert len(model.loss_history) > 0

    # Verify simplex constraints: weights sum to 1 for each sample
    weight_sums = np.sum(model.weights, axis=1)
    np.testing.assert_allclose(weight_sums, np.ones(100), rtol=1e-5)

    # Verify non-negativity constraint on weights
    assert np.all(model.weights >= 0)


def test_transform(sample_data):
    """Verify the model's transform functionality on new data.

    This test ensures that transformation of new data:
    1. Produces weights of the correct shape
    2. Satisfies simplex constraints (non-negative, sum to 1)

    Args:
        sample_data: Synthetic dataset fixture
    """
    model = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    model.fit(sample_data)

    # Transform a subset of the original data
    new_data = sample_data[:10]
    weights = model.transform(new_data)

    # Verify correct dimensions of resulting weights
    assert weights.shape == (10, 3)

    # Verify simplex constraints: weights sum to 1 for each sample
    weight_sums = np.sum(weights, axis=1)
    np.testing.assert_allclose(weight_sums, np.ones(10), rtol=1e-5)

    # Verify non-negativity constraint on weights
    assert np.all(weights >= 0)


def test_transform_with_lbfgs(sample_data):
    """Verify the model's LBFGS-based transform functionality.

    This test ensures that the alternative LBFGS-based transformation:
    1. Produces weights of the correct shape
    2. Satisfies simplex constraints (non-negative, sum to 1)

    Args:
        sample_data: Synthetic dataset fixture
    """
    model = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    model.fit(sample_data)

    # Transform data using the LBFGS optimizer
    new_data = sample_data[:10]
    weights = model.transform_with_lbfgs(new_data)

    # Verify correct dimensions of resulting weights
    assert weights.shape == (10, 3)

    # Verify simplex constraints: weights sum to 1 for each sample
    weight_sums = np.sum(weights, axis=1)
    np.testing.assert_allclose(weight_sums, np.ones(10), rtol=1e-5)

    # Verify non-negativity constraint on weights
    assert np.all(weights >= 0)


def test_fit_transform(sample_data):
    """Verify the combined fit_transform method.

    This test ensures that the fit_transform method:
    1. Correctly fits the model
    2. Returns weights that satisfy simplex constraints

    Args:
        sample_data: Synthetic dataset fixture
    """
    model = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    weights = model.fit_transform(sample_data)

    # Verify correct dimensions of resulting weights
    assert weights.shape == (100, 3)

    # Verify simplex constraints: weights sum to 1 for each sample
    weight_sums = np.sum(weights, axis=1)
    np.testing.assert_allclose(weight_sums, np.ones(100), rtol=1e-5)

    # Verify that the model has been properly fitted
    assert model.archetypes is not None
    assert model.weights is not None


def test_normalize_option(sample_data):
    """Verify the effect of the normalization option during fitting.

    This test compares models fitted with and without normalization to ensure:
    1. Normalization parameters are correctly calculated
    2. The resulting archetypes differ when normalization is applied

    Args:
        sample_data: Synthetic dataset fixture
    """
    # Fit a model with normalization enabled
    model_with_norm = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    model_with_norm.fit(sample_data, normalize=True)

    # Fit a model with normalization disabled
    model_without_norm = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    model_without_norm.fit(sample_data, normalize=False)

    # Verify that normalization parameters are properly calculated
    assert model_with_norm.X_mean is not None
    assert model_with_norm.X_std is not None

    # Verify that normalization affects the resulting archetypes
    # Note: While not guaranteed, normalization typically produces different results
    assert not np.allclose(model_with_norm.archetypes, model_without_norm.archetypes)


def test_reconstruction(sample_data):
    """Verify data reconstruction capabilities.

    This test ensures that:
    1. Reconstructed data has the correct shape
    2. Reconstruction error is finite and non-zero

    Args:
        sample_data: Synthetic dataset fixture
    """
    model = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    model.fit(sample_data)

    # Reconstruct the original data from archetypes and weights
    reconstructed = np.matmul(model.weights, model.archetypes)

    # Verify correct dimensions of reconstructed data
    assert reconstructed.shape == sample_data.shape

    # Calculate and verify reconstruction error
    error = np.linalg.norm(sample_data - reconstructed, ord="fro")
    assert error > 0  # Perfect reconstruction is impossible with fewer archetypes than samples
    assert np.isfinite(error)  # Error should be a finite value


def test_error_before_fit():
    """Verify proper error handling when methods are called before fitting.

    This test ensures that appropriate ValueError exceptions are raised when
    transform methods are called on an unfitted model.
    """
    model = ImprovedArchetypalAnalysis(n_archetypes=3)
    X = np.random.rand(10, 5)

    # Verify that transform methods raise ValueError when called before fitting
    with pytest.raises(ValueError):
        model.transform(X)

    with pytest.raises(ValueError):
        model.transform_with_lbfgs(X)
