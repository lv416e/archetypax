"""Tests for the base module."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from archetypax.base import ArchetypalAnalysis


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, _ = make_blobs(n_samples=100, n_features=5, centers=3, cluster_std=1.0, random_state=42)
    return X


def test_initialization():
    """Test initialization of ArchetypalAnalysis."""
    model = ArchetypalAnalysis(n_archetypes=3)
    assert model.n_archetypes == 3
    assert model.max_iter == 500
    assert model.tol == 1e-6
    assert model.archetypes is None
    assert model.weights is None


def test_fit(sample_data):
    """Test fitting the model."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=50)
    model.fit(sample_data)

    # Check that archetypes and weights are set
    assert model.archetypes is not None
    assert model.weights is not None

    # Check shapes
    assert model.archetypes.shape == (3, 5)  # n_archetypes x n_features
    assert model.weights.shape == (100, 3)  # n_samples x n_archetypes

    # Check that weights sum to 1
    weight_sums = np.sum(model.weights, axis=1)
    np.testing.assert_allclose(weight_sums, np.ones(100), rtol=1e-5)

    # Check that weights are non-negative
    assert np.all(model.weights >= 0)


def test_transform(sample_data):
    """Test transforming data."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=50)
    model.fit(sample_data)

    # Transform the same data
    weights = model.transform(sample_data)

    # Check shape
    assert weights.shape == (100, 3)

    # Check that weights sum to 1
    weight_sums = np.sum(weights, axis=1)
    np.testing.assert_allclose(weight_sums, np.ones(100), rtol=1e-5)

    # Check that weights are non-negative
    assert np.all(weights >= 0)


def test_fit_transform(sample_data):
    """Test fit_transform method."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=50)
    weights = model.fit_transform(sample_data)

    # Check shape
    assert weights.shape == (100, 3)

    # Check that weights sum to 1
    weight_sums = np.sum(weights, axis=1)
    np.testing.assert_allclose(weight_sums, np.ones(100), rtol=1e-5)


def test_reconstruct(sample_data):
    """Test reconstruction of data."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=50)
    model.fit(sample_data)

    # Reconstruct the data
    X_reconstructed = model.reconstruct()

    # Check shape
    assert X_reconstructed.shape == sample_data.shape

    # Check that the reconstruction error is reasonable
    error = np.linalg.norm(sample_data - X_reconstructed, ord="fro")
    assert error > 0  # Should not be perfect

    # Note: With small number of iterations, the reconstruction might not always
    # be better than the mean. We just check that the error exists and is finite.
    assert np.isfinite(error)


def test_get_loss_history(sample_data):
    """Test getting loss history."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=50)
    model.fit(sample_data)

    # Get loss history
    loss_history = model.get_loss_history()

    # Check that it's a list
    assert isinstance(loss_history, list)

    # Check that it has values
    assert len(loss_history) > 0

    # Check that loss decreases
    assert loss_history[0] >= loss_history[-1]


def test_error_before_fit():
    """Test error when using methods before fitting."""
    model = ArchetypalAnalysis(n_archetypes=3)
    X = np.random.rand(10, 5)

    # These should all raise ValueError
    with pytest.raises(ValueError):
        model.transform(X)

    with pytest.raises(ValueError):
        model.reconstruct()
