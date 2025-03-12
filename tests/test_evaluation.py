"""Tests for the evaluation module."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from archetypax.base import ArchetypalAnalysis
from archetypax.evaluation import ArchetypalAnalysisEvaluator


@pytest.fixture
def fitted_model_and_data():
    """Generate sample data and a fitted model for testing."""
    X, _ = make_blobs(n_samples=100, n_features=5, centers=3, cluster_std=1.0, random_state=42)
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=50)
    model.fit(X)
    return model, X


def test_evaluator_initialization(fitted_model_and_data):
    """Test initialization of ArchetypalAnalysisEvaluator."""
    model, _ = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    # Check that model is set
    assert evaluator.model is model

    # Check that cached values are set
    assert evaluator.n_archetypes == 3
    assert evaluator.n_features == 5
    assert evaluator.dominant_archetypes.shape == (100,)


def test_reconstruction_error(fitted_model_and_data):
    """Test reconstruction error calculation."""
    model, X = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    # Test different metrics
    frobenius_error = evaluator.reconstruction_error(X, "frobenius")
    mae_error = evaluator.reconstruction_error(X, "mae")
    mse_error = evaluator.reconstruction_error(X, "mse")
    relative_error = evaluator.reconstruction_error(X, "relative")

    # Check that errors are positive
    assert frobenius_error > 0
    assert mae_error > 0
    assert mse_error > 0
    assert relative_error > 0  # Relative error should be positive

    # Check invalid metric
    with pytest.raises(ValueError):
        evaluator.reconstruction_error(X, "invalid_metric")


def test_explained_variance(fitted_model_and_data):
    """Test explained variance calculation."""
    model, X = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    explained_var = evaluator.explained_variance(X)

    # With small number of iterations, explained variance might be negative
    # Just check that it's finite
    assert np.isfinite(explained_var)


def test_dominant_archetype_purity(fitted_model_and_data):
    """Test dominant archetype purity calculation."""
    model, _ = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    purity_results = evaluator.dominant_archetype_purity()

    # Check that results contain expected keys
    assert "archetype_purity" in purity_results
    assert "overall_purity" in purity_results
    assert "purity_std" in purity_results
    assert "max_weights" in purity_results

    # Check that overall purity is between 0 and 1
    assert 0 <= purity_results["overall_purity"] <= 1

    # Check that max_weights has the right shape
    assert purity_results["max_weights"].shape == (100,)


def test_archetype_separation(fitted_model_and_data):
    """Test archetype separation calculation."""
    model, _ = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    separation_results = evaluator.archetype_separation()

    # Check that results contain expected keys
    assert "min_distance" in separation_results
    assert "max_distance" in separation_results
    assert "mean_distance" in separation_results
    assert "distance_ratio" in separation_results

    # Check that distances are positive
    assert separation_results["min_distance"] > 0
    assert separation_results["max_distance"] > 0
    assert separation_results["mean_distance"] > 0

    # Check that distance ratio is between 0 and 1
    assert 0 <= separation_results["distance_ratio"] <= 1


def test_clustering_metrics(fitted_model_and_data):
    """Test clustering metrics calculation."""
    model, X = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    clustering_results = evaluator.clustering_metrics(X)

    # Check that results contain expected keys
    assert "silhouette" in clustering_results
    assert "davies_bouldin" in clustering_results

    # Check that silhouette is between -1 and 1
    assert -1 <= clustering_results["silhouette"] <= 1

    # Check that davies_bouldin is positive
    assert clustering_results["davies_bouldin"] >= 0


def test_archetype_feature_importance(fitted_model_and_data):
    """Test archetype feature importance calculation."""
    model, _ = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    importance_df = evaluator.archetype_feature_importance()

    # Check shape
    assert importance_df.shape == (3, 5)  # n_archetypes x n_features

    # Check that values are positive
    assert np.all(importance_df.values >= 0)


def test_weight_diversity(fitted_model_and_data):
    """Test weight diversity calculation."""
    model, _ = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    diversity_results = evaluator.weight_diversity()

    # Check that results contain expected keys
    assert "mean_entropy" in diversity_results
    assert "mean_normalized_entropy" in diversity_results
    assert "entropy_std" in diversity_results
    assert "min_entropy" in diversity_results
    assert "max_entropy" in diversity_results

    # Check that normalized entropy is between 0 and 1
    assert 0 <= diversity_results["mean_normalized_entropy"] <= 1

    # Check that min_entropy <= mean_entropy <= max_entropy
    assert diversity_results["min_entropy"] <= diversity_results["mean_entropy"]
    assert diversity_results["mean_entropy"] <= diversity_results["max_entropy"]


def test_comprehensive_evaluation(fitted_model_and_data):
    """Test comprehensive evaluation."""
    model, X = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    results = evaluator.comprehensive_evaluation(X)

    # Check that results contain expected keys
    assert "reconstruction" in results
    assert "explained_variance" in results


# Add tests for visualization methods in the evaluator
def test_plot_methods(fitted_model_and_data, monkeypatch):
    """Test plotting methods in the evaluator."""
    model, X = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    # Mock plt.show to prevent actual display during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Test plot_feature_importance_heatmap
    evaluator.plot_feature_importance_heatmap()
    assert plt.gcf() is not None
    plt.close()

    # Test with feature names
    feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
    evaluator.plot_feature_importance_heatmap(feature_names=feature_names)
    assert plt.gcf() is not None
    plt.close()

    # Test plot_archetype_feature_comparison
    evaluator.plot_archetype_feature_comparison(top_n=3)
    assert plt.gcf() is not None
    plt.close()

    # Test with feature names
    evaluator.plot_archetype_feature_comparison(top_n=3, feature_names=feature_names)
    assert plt.gcf() is not None
    plt.close()

    # Test plot_weight_distributions
    evaluator.plot_weight_distributions()
    assert plt.gcf() is not None
    plt.close()

    # Test plot_purity_distribution
    evaluator.plot_purity_distribution()
    assert plt.gcf() is not None
    plt.close()

    # Test plot_distance_matrix
    evaluator.plot_distance_matrix()
    assert plt.gcf() is not None
    plt.close()

    # Test plot_entropy_vs_reconstruction
    evaluator.plot_entropy_vs_reconstruction(X)
    assert plt.gcf() is not None
    plt.close()


def test_additional_metrics(fitted_model_and_data):
    """Test additional metrics methods."""
    model, X = fitted_model_and_data
    evaluator = ArchetypalAnalysisEvaluator(model)

    # Test weight_diversity
    diversity = evaluator.weight_diversity()
    assert "mean_entropy" in diversity

    # Test archetype_separation
    separation = evaluator.archetype_separation()
    assert "min_distance" in separation

    # Test clustering_metrics
    clustering = evaluator.clustering_metrics(X)
    assert "silhouette" in clustering

    # Test comprehensive_evaluation
    results = evaluator.comprehensive_evaluation(X)
    assert "reconstruction" in results
    assert "explained_variance" in results
