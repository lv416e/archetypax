"""Unit tests for the visualization module.

This module contains tests for the ArchetypalAnalysisVisualizer class,
verifying that visualization methods function correctly with various inputs.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from archetypax import ImprovedArchetypalAnalysis
from archetypax.visualization import ArchetypalAnalysisVisualizer


@pytest.fixture
def fitted_model_and_data():
    """Generate sample data and a fitted model for testing.

    Returns:
        tuple: (fitted_model, data_matrix)
    """
    X, _ = make_blobs(n_samples=100, n_features=5, centers=3, cluster_std=1.0, random_state=42)
    model = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    model.fit(X)
    return model, X


@pytest.fixture
def fitted_model_and_data_2d():
    """Generate 2D sample data and a fitted model for testing.

    Returns:
        tuple: (fitted_model, data_matrix)
    """
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=42)
    model = ImprovedArchetypalAnalysis(n_archetypes=3, max_iter=10)
    model.fit(X)
    return model, X


def test_plot_loss(fitted_model_and_data, monkeypatch):
    """Test the plot_loss method."""
    model, _ = fitted_model_and_data

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Call the method
    ArchetypalAnalysisVisualizer.plot_loss(model)

    # Verify a figure was created
    assert plt.gcf() is not None
    plt.close()


def test_plot_archetypes_2d(fitted_model_and_data_2d, monkeypatch):
    """Test the plot_archetypes_2d method."""
    model, X = fitted_model_and_data_2d

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Call the method
    ArchetypalAnalysisVisualizer.plot_archetypes_2d(model, X)

    # Verify a figure was created
    assert plt.gcf() is not None
    plt.close()

    # Test with feature names
    feature_names = ["Feature 1", "Feature 2"]
    ArchetypalAnalysisVisualizer.plot_archetypes_2d(model, X, feature_names)
    assert plt.gcf() is not None
    plt.close()


def test_plot_archetypes_2d_errors(fitted_model_and_data, monkeypatch):
    """Test error handling in plot_archetypes_2d method."""
    model, X = fitted_model_and_data  # 5D data

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Should raise ValueError for non-2D data
    with pytest.raises(ValueError, match="only for 2D data"):
        ArchetypalAnalysisVisualizer.plot_archetypes_2d(model, X)

    # Create an unfitted model
    unfitted_model = ImprovedArchetypalAnalysis(n_archetypes=3)

    # Should raise ValueError for unfitted model
    with pytest.raises(ValueError, match="must be fitted"):
        ArchetypalAnalysisVisualizer.plot_archetypes_2d(unfitted_model, X[:, :2])


def test_plot_reconstruction_comparison(fitted_model_and_data_2d, monkeypatch):
    """Test the plot_reconstruction_comparison method."""
    model, X = fitted_model_and_data_2d  # Use 2D data

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Call the method
    ArchetypalAnalysisVisualizer.plot_reconstruction_comparison(model, X)

    # Verify a figure was created
    assert plt.gcf() is not None
    plt.close()


def test_plot_membership_weights(fitted_model_and_data, monkeypatch):
    """Test the plot_membership_weights method."""
    model, _ = fitted_model_and_data

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Call the method
    ArchetypalAnalysisVisualizer.plot_membership_weights(model)

    # Verify a figure was created
    assert plt.gcf() is not None
    plt.close()

    # Test with limited samples
    ArchetypalAnalysisVisualizer.plot_membership_weights(model, n_samples=10)
    assert plt.gcf() is not None
    plt.close()


def test_plot_archetype_profiles(fitted_model_and_data, monkeypatch):
    """Test the plot_archetype_profiles method."""
    model, _ = fitted_model_and_data

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Call the method
    ArchetypalAnalysisVisualizer.plot_archetype_profiles(model)

    # Verify a figure was created
    assert plt.gcf() is not None
    plt.close()

    # Test with feature names
    feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
    ArchetypalAnalysisVisualizer.plot_archetype_profiles(model, feature_names)
    assert plt.gcf() is not None
    plt.close()


def test_plot_archetype_distribution(fitted_model_and_data, monkeypatch):
    """Test the plot_archetype_distribution method."""
    model, _ = fitted_model_and_data

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Call the method
    ArchetypalAnalysisVisualizer.plot_archetype_distribution(model)

    # Verify a figure was created
    assert plt.gcf() is not None
    plt.close()


def test_plot_simplex_2d(fitted_model_and_data, monkeypatch):
    """Test the plot_simplex_2d method."""
    model, _ = fitted_model_and_data

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Call the method
    ArchetypalAnalysisVisualizer.plot_simplex_2d(model)

    # Verify a figure was created
    assert plt.gcf() is not None
    plt.close()

    # Test with limited samples
    ArchetypalAnalysisVisualizer.plot_simplex_2d(model, n_samples=10)
    assert plt.gcf() is not None
    plt.close()


def test_unfitted_model_errors():
    """Test error handling for unfitted models."""
    unfitted_model = ImprovedArchetypalAnalysis(n_archetypes=3)
    X_2d = np.random.rand(10, 2)

    # plot_loss doesn't raise an error, it just prints a message
    # So we don't test it here

    with pytest.raises(ValueError, match="must be fitted"):
        ArchetypalAnalysisVisualizer.plot_reconstruction_comparison(unfitted_model, X_2d)

    with pytest.raises(ValueError, match="must be fitted"):
        ArchetypalAnalysisVisualizer.plot_membership_weights(unfitted_model)

    with pytest.raises(ValueError, match="must be fitted"):
        ArchetypalAnalysisVisualizer.plot_archetype_profiles(unfitted_model)

    with pytest.raises(ValueError, match="must be fitted"):
        ArchetypalAnalysisVisualizer.plot_archetype_distribution(unfitted_model)

    with pytest.raises(ValueError, match="must be fitted"):
        ArchetypalAnalysisVisualizer.plot_simplex_2d(unfitted_model)

    with pytest.raises(ValueError, match="must be fitted"):
        ArchetypalAnalysisVisualizer.plot_archetypes_2d(unfitted_model, X_2d)
