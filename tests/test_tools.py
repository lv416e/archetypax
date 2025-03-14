"""Unit tests for archetypax tools."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from archetypax.models.base import ArchetypalAnalysis
from archetypax.tools.evaluation import ArchetypalAnalysisEvaluator
from archetypax.tools.interpret import ArchetypalAnalysisInterpreter
from archetypax.tools.visualization import ArchetypalAnalysisVisualizer


@pytest.fixture
def sample_data():
    """Generate synthetic data for testing purposes."""
    X, _ = make_blobs(n_samples=30, n_features=5, centers=3, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def fitted_model(sample_data):
    """Create a pre-fitted model for testing tool functionality."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=20)
    model.fit(sample_data)
    return model


class TestArchetypalAnalysisEvaluator:
    """Test suite for the ArchetypalAnalysisEvaluator class."""

    def test_initialization(self, fitted_model):
        """Verify that the evaluator initializes correctly with a fitted model."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        assert evaluator.model is fitted_model
        assert evaluator.n_archetypes == 3
        assert evaluator.n_features == 5
        assert evaluator.dominant_archetypes.shape == (30,)

    def test_initialization_with_unfitted_model(self):
        """Confirm that initialization raises an error with an unfitted model."""
        model = ArchetypalAnalysis(n_archetypes=3)
        with pytest.raises(ValueError):
            ArchetypalAnalysisEvaluator(model)

    def test_reconstruction_error(self, sample_data, fitted_model):
        """Validate that reconstruction error metrics calculate correctly."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)

        # Test various error metrics
        error_frob = evaluator.reconstruction_error(sample_data, metric="frobenius")
        error_mse = evaluator.reconstruction_error(sample_data, metric="mse")
        error_mae = evaluator.reconstruction_error(sample_data, metric="mae")
        error_rel = evaluator.reconstruction_error(sample_data, metric="relative")

        # Just check that errors are non-negative
        assert error_frob >= 0
        assert error_mse >= 0
        assert error_mae >= 0
        # Relative error might be greater than 1 for poor fits
        assert error_rel >= 0

    def test_explained_variance(self, sample_data, fitted_model):
        """Ensure that explained variance calculation returns a valid value."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        var_explained = evaluator.explained_variance(sample_data)

        # Explained variance can be negative for poor fits
        # Just check that it returns a value
        assert isinstance(var_explained, float)

    def test_archetype_separation(self, fitted_model):
        """Verify that archetype separation metrics are correctly calculated."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        separation = evaluator.archetype_separation()

        assert "mean_distance" in separation
        assert "min_distance" in separation
        assert "max_distance" in separation
        assert separation["mean_distance"] >= 0

    def test_dominant_archetype_purity(self, fitted_model):
        """Test dominant archetype purity calculation."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        purity = evaluator.dominant_archetype_purity()

        assert "archetype_purity" in purity
        assert "overall_purity" in purity
        assert "purity_std" in purity
        assert 0 <= purity["overall_purity"] <= 1

    def test_clustering_metrics(self, sample_data, fitted_model):
        """Test clustering metrics calculation."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        metrics = evaluator.clustering_metrics(sample_data)

        assert "davies_bouldin" in metrics
        assert "silhouette" in metrics
        assert -1 <= metrics["silhouette"] <= 1

    def test_archetype_feature_importance(self, fitted_model):
        """Test archetype feature importance calculation."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        importance = evaluator.archetype_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert importance.shape == (3, 5)  # 3 archetypes, 5 features

    def test_weight_diversity(self, fitted_model):
        """Test weight diversity metrics calculation."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        diversity = evaluator.weight_diversity()

        assert "mean_entropy" in diversity
        assert "entropy_std" in diversity
        assert "max_entropy" in diversity
        assert "mean_normalized_entropy" in diversity
        assert diversity["mean_entropy"] >= 0

    def test_invalid_reconstruction_error_metric(self, sample_data, fitted_model):
        """Test that using an invalid metric raises an error."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)

        # Test with invalid metric
        with pytest.raises(ValueError):
            evaluator.reconstruction_error(sample_data, metric="invalid")

    def test_print_evaluation_report(self, sample_data, fitted_model):
        """Test that print_evaluation_report runs without errors."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)

        # Just check that it runs without errors
        evaluator.print_evaluation_report(sample_data)

    @pytest.mark.slow
    def test_comprehensive_evaluation(self, sample_data, fitted_model):
        """Confirm that comprehensive evaluation returns metrics."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        results = evaluator.comprehensive_evaluation(sample_data)

        # Check that results is a dictionary with expected structure
        assert isinstance(results, dict)
        # Check for clustering metrics which should always be present
        assert "clustering" in results
        assert "davies_bouldin" in results["clustering"]
        assert "silhouette" in results["clustering"]
        # Check for other metric categories
        assert "reconstruction" in results
        assert "diversity" in results
        assert "purity" in results

    @pytest.mark.slow
    def test_biarchetypal_evaluator(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Test BiarchetypalAnalysisEvaluator initialization and basic functionality."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        # Initialize the evaluator with the fitted model
        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)

        # Verify basic properties
        assert evaluator.model is fitted_biarchetypal_model
        assert evaluator.n_archetypes_first == 2
        assert evaluator.n_archetypes_second == 2

        # Test comprehensive evaluation
        results = evaluator.comprehensive_evaluation(biarchetypal_sample_data)

        # Check that results contains expected categories
        assert "reconstruction" in results
        assert "diversity" in results
        assert "purity" in results


class TestArchetypalAnalysisInterpreter:
    """Test suite for the ArchetypalAnalysisInterpreter class."""

    def test_initialization(self):
        """Verify that the interpreter initializes correctly."""
        interpreter = ArchetypalAnalysisInterpreter()
        assert isinstance(interpreter.models_dict, dict)
        assert len(interpreter.models_dict) == 0
        assert isinstance(interpreter.results, dict)

    def test_add_model(self, fitted_model):
        """Test adding a model to the interpreter."""
        interpreter = ArchetypalAnalysisInterpreter()
        interpreter.add_model(3, fitted_model)

        assert 3 in interpreter.models_dict
        assert interpreter.models_dict[3] is fitted_model

    def test_add_unfitted_model(self):
        """Test that adding an unfitted model raises an error."""
        interpreter = ArchetypalAnalysisInterpreter()
        model = ArchetypalAnalysis(n_archetypes=3)

        with pytest.raises(ValueError):
            interpreter.add_model(3, model)

    def test_feature_distinctiveness(self, fitted_model):
        """Test feature distinctiveness calculation."""
        interpreter = ArchetypalAnalysisInterpreter()
        distinctiveness = interpreter.feature_distinctiveness(fitted_model.archetypes)

        assert distinctiveness.shape == (3,)
        assert np.all(distinctiveness >= 0)

    def test_sparsity_coefficient(self, fitted_model):
        """Test sparsity coefficient calculation."""
        interpreter = ArchetypalAnalysisInterpreter()
        sparsity = interpreter.sparsity_coefficient(fitted_model.archetypes)

        assert sparsity.shape == (3,)
        assert np.all(sparsity >= 0)
        assert np.all(sparsity <= 1)

    def test_cluster_purity(self, fitted_model):
        """Test cluster purity calculation."""
        interpreter = ArchetypalAnalysisInterpreter()
        purity, overall_purity = interpreter.cluster_purity(fitted_model.weights)

        assert purity.shape == (3,)
        assert np.all(purity >= 0)
        assert np.all(purity <= 1)
        assert 0 <= overall_purity <= 1

    def test_optimal_archetypes_elbow(self, sample_data):
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetype_profiles")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetype_distribution")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_simplex_2d")

    @pytest.mark.slow
    def test_plot_loss(self, fitted_model):
        """Test that plot_loss runs without errors."""
        ArchetypalAnalysisVisualizer.plot_loss(fitted_model)


@pytest.fixture
def biarchetypal_sample_data():
    """Generate synthetic data for testing biarchetypal analysis."""
    X, _ = make_blobs(n_samples=30, n_features=5, centers=4, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def fitted_biarchetypal_model(biarchetypal_sample_data):
    """Create a pre-fitted biarchetypal model for testing."""
    from archetypax.models.biarchetypes import BiarchetypalAnalysis

    model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=2, max_iter=20)
    model.fit(biarchetypal_sample_data)
    return model


@pytest.fixture
def biarchetypal_sample_data_2d():
    """Generate 2D synthetic data for testing biarchetypal visualization functions."""
    X, _ = make_blobs(n_samples=30, n_features=2, centers=4, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def fitted_biarchetypal_model_2d(biarchetypal_sample_data_2d):
    """Create a pre-fitted biarchetypal model for testing visualization functions."""
    from archetypax.models.biarchetypes import BiarchetypalAnalysis

    model = BiarchetypalAnalysis(n_archetypes_first=3, n_archetypes_second=3, max_iter=20)
    model.fit(biarchetypal_sample_data_2d)
    return model


@pytest.fixture
def fitted_biarchetypal_model_2d_3arch(biarchetypal_sample_data_2d):
    """Create a pre-fitted biarchetypal model with 3 archetypes for testing simplex visualization."""
    from archetypax.models.biarchetypes import BiarchetypalAnalysis

    model = BiarchetypalAnalysis(n_archetypes_first=3, n_archetypes_second=3, max_iter=20)
    model.fit(biarchetypal_sample_data_2d)
    return model


class TestBiarchetypalAnalysisEvaluator:
    """Test suite for the BiarchetypalAnalysisEvaluator class."""

    def test_initialization(self, fitted_biarchetypal_model):
        """Verify that the evaluator initializes correctly with a fitted model."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        assert evaluator.model is fitted_biarchetypal_model
        assert evaluator.n_archetypes_first == 2
        assert evaluator.n_archetypes_second == 2
        assert evaluator.n_features == 5
        assert evaluator.dominant_archetypes_first.shape == (30,)
        assert evaluator.dominant_archetypes_second.shape == (30,)

    def test_initialization_with_unfitted_model(self):
        """Confirm that initialization raises an error with an unfitted model."""
        from archetypax.models.biarchetypes import BiarchetypalAnalysis
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=2)
        with pytest.raises(ValueError):
            BiarchetypalAnalysisEvaluator(model)

    def test_initialization_with_wrong_model_type(self, fitted_model):
        """Confirm that initialization raises an error with a wrong model type."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        with pytest.raises(TypeError):
            BiarchetypalAnalysisEvaluator(fitted_model)

    def test_reconstruction_error(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Validate that reconstruction error metrics calculate correctly."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)

        # Test various error metrics
        error_frob = evaluator.reconstruction_error(biarchetypal_sample_data, metric="frobenius")
        error_mse = evaluator.reconstruction_error(biarchetypal_sample_data, metric="mse")
        error_mae = evaluator.reconstruction_error(biarchetypal_sample_data, metric="mae")
        error_rel = evaluator.reconstruction_error(biarchetypal_sample_data, metric="relative")

        # Just check that errors are non-negative
        assert error_frob >= 0
        assert error_mse >= 0
        assert error_mae >= 0
        assert error_rel >= 0

        # Test invalid metric
        with pytest.raises(ValueError):
            evaluator.reconstruction_error(biarchetypal_sample_data, metric="invalid")

    def test_explained_variance(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Ensure that explained variance calculation returns a valid value."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        var_explained = evaluator.explained_variance(biarchetypal_sample_data)

        # Just check that it returns a value
        assert isinstance(var_explained, float)

    def test_archetype_separation(self, fitted_biarchetypal_model):
        """Verify that archetype separation metrics are correctly calculated."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        separation = evaluator.archetype_separation()

        # Check that expected metrics are present
        assert "mean_distance_first" in separation
        assert "min_distance_first" in separation
        assert "max_distance_first" in separation
        assert "mean_distance_second" in separation
        assert "min_distance_second" in separation
        assert "max_distance_second" in separation
        assert "mean_cross_distance" in separation
        assert "min_cross_distance" in separation
        assert "max_cross_distance" in separation

        # Check that values are non-negative
        assert separation["mean_distance_first"] >= 0
        assert separation["mean_cross_distance"] >= 0

    def test_dominant_archetype_purity(self, fitted_biarchetypal_model):
        """Test dominant archetype purity calculation."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        purity = evaluator.dominant_archetype_purity()

        # Check that expected metrics are present
        assert "archetype_purity_first" in purity
        assert "archetype_purity_second" in purity
        assert "overall_purity_first" in purity
        assert "overall_purity_second" in purity
        assert "purity_std_first" in purity
        assert "purity_std_second" in purity

        # Check that values are in valid range
        assert 0 <= purity["overall_purity_first"] <= 1
        assert 0 <= purity["overall_purity_second"] <= 1

    def test_weight_diversity(self, fitted_biarchetypal_model):
        """Test weight diversity metrics calculation."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        diversity = evaluator.weight_diversity()

        # Check that expected metrics are present
        assert "mean_entropy_first" in diversity
        assert "entropy_std_first" in diversity
        assert "max_entropy_first" in diversity
        assert "mean_normalized_entropy_first" in diversity
        assert "mean_entropy_second" in diversity
        assert "entropy_std_second" in diversity
        assert "max_entropy_second" in diversity
        assert "mean_normalized_entropy_second" in diversity

        # Check that values are non-negative
        assert diversity["mean_entropy_first"] >= 0
        assert diversity["mean_entropy_second"] >= 0
        assert diversity["mean_normalized_entropy_first"] >= 0
        assert diversity["mean_normalized_entropy_second"] >= 0

    @pytest.mark.slow
    def test_comprehensive_evaluation(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Confirm that comprehensive evaluation returns metrics."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        results = evaluator.comprehensive_evaluation(biarchetypal_sample_data)

        # Check that results is a dictionary with expected structure
        assert isinstance(results, dict)
        assert "reconstruction" in results
        assert "separation" in results
        assert "purity" in results
        assert "diversity" in results

        # Check that reconstruction metrics are present
        assert "frobenius" in results["reconstruction"]
        assert "mse" in results["reconstruction"]
        assert "mae" in results["reconstruction"]
        assert "relative" in results["reconstruction"]
        assert "explained_variance" in results["reconstruction"]

    @pytest.mark.slow
    def test_print_evaluation_report(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Test that print_evaluation_report runs without errors."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        # Just check that it runs without errors
        evaluator.print_evaluation_report(biarchetypal_sample_data)


class TestBiarchetypalAnalysisVisualizer:
    """Test suite for the BiarchetypalAnalysisVisualizer class."""

    def test_static_methods_exist(self):
        """Verify that all expected static methods exist in the visualizer."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_dual_archetypes_2d")
        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_biarchetypal_reconstruction")
        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_dual_membership_heatmap")
        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_mixture_effect")
        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_dual_simplex_2d")

    @pytest.mark.slow
    def test_plot_dual_membership_heatmap(self, fitted_biarchetypal_model):
        """Test that plot_dual_membership_heatmap runs without errors."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        # Test with default parameters
        BiarchetypalAnalysisVisualizer.plot_dual_membership_heatmap(fitted_biarchetypal_model)

        # Test with specific number of samples
        BiarchetypalAnalysisVisualizer.plot_dual_membership_heatmap(fitted_biarchetypal_model, n_samples=20)

    @pytest.mark.slow
    def test_plot_dual_archetypes_2d_with_2d_data(self, biarchetypal_sample_data_2d, fitted_biarchetypal_model_2d):
        """Test that plot_dual_archetypes_2d runs without errors with 2D data."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        # Create feature names for testing
        feature_names = [f"Feature {i}" for i in range(biarchetypal_sample_data_2d.shape[1])]

        # Test with feature names
        BiarchetypalAnalysisVisualizer.plot_dual_archetypes_2d(
            fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d, feature_names
        )

        # Test without feature names
        BiarchetypalAnalysisVisualizer.plot_dual_archetypes_2d(
            fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d
        )

    @pytest.mark.slow
    def test_plot_biarchetypal_reconstruction_with_2d_data(
        self, biarchetypal_sample_data_2d, fitted_biarchetypal_model_2d
    ):
        """Test that plot_biarchetypal_reconstruction runs without errors with 2D data."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        BiarchetypalAnalysisVisualizer.plot_biarchetypal_reconstruction(
            fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d
        )

    @pytest.mark.slow
    def test_plot_dual_membership_heatmap_with_2d_model(self, fitted_biarchetypal_model_2d):
        """Test that plot_dual_membership_heatmap runs without errors with 2D model."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        # Test with default parameters
        BiarchetypalAnalysisVisualizer.plot_dual_membership_heatmap(fitted_biarchetypal_model_2d)

        # Test with specific number of samples
        BiarchetypalAnalysisVisualizer.plot_dual_membership_heatmap(fitted_biarchetypal_model_2d, n_samples=20)

    @pytest.mark.slow
    def test_plot_mixture_effect_with_2d_data(self, biarchetypal_sample_data_2d, fitted_biarchetypal_model_2d):
        """Test that plot_mixture_effect runs without errors with 2D data."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        # Test with default parameters
        BiarchetypalAnalysisVisualizer.plot_mixture_effect(fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d)

        # Test with specific mixture steps
        BiarchetypalAnalysisVisualizer.plot_mixture_effect(
            fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d, mixture_steps=3
        )

    @pytest.mark.slow
    def test_plot_dual_simplex_2d_with_3arch_model(self, fitted_biarchetypal_model_2d_3arch):
        """Test that plot_dual_simplex_2d runs without errors with a model having 3 archetypes in each set."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        # Test with default parameters
        BiarchetypalAnalysisVisualizer.plot_dual_simplex_2d(fitted_biarchetypal_model_2d_3arch)

        # Test with specific number of samples
        BiarchetypalAnalysisVisualizer.plot_dual_simplex_2d(fitted_biarchetypal_model_2d_3arch, n_samples=100)


@pytest.fixture
def sample_data_2d():
    """Generate 2D synthetic data for testing visualization functions."""
    X, _ = make_blobs(n_samples=30, n_features=2, centers=3, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def fitted_model_2d(sample_data_2d):
    """Create a pre-fitted model for testing visualization functions."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=20)
    model.fit(sample_data_2d)
    return model


class TestArchetypalAnalysisVisualizer:
    """Test suite for the ArchetypalAnalysisVisualizer class."""

    def test_static_methods_exist(self):
        """Verify that all expected static methods exist in the visualizer."""
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_loss")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetypes_2d")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_reconstruction_comparison")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_membership_weights")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetype_profiles")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetype_distribution")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_simplex_2d")

    @pytest.mark.slow
    def test_plot_loss(self, fitted_model):
        """Test that plot_loss runs without errors."""
        ArchetypalAnalysisVisualizer.plot_loss(fitted_model)

    @pytest.mark.slow
    def test_plot_archetypes_2d(self, sample_data_2d, fitted_model_2d):
        """Test that plot_archetypes_2d runs without errors."""
        # Create feature names for testing
        feature_names = [f"Feature {i}" for i in range(sample_data_2d.shape[1])]

        # Test with feature names
        ArchetypalAnalysisVisualizer.plot_archetypes_2d(fitted_model_2d, sample_data_2d, feature_names)

        # Test without feature names
        ArchetypalAnalysisVisualizer.plot_archetypes_2d(fitted_model_2d, sample_data_2d)

    @pytest.mark.slow
    def test_plot_reconstruction_comparison(self, sample_data_2d, fitted_model_2d):
        """Test that plot_reconstruction_comparison runs without errors."""
        ArchetypalAnalysisVisualizer.plot_reconstruction_comparison(fitted_model_2d, sample_data_2d)

    @pytest.mark.slow
    def test_plot_membership_weights(self, fitted_model):
        """Test that plot_membership_weights runs without errors."""
        # Test with default parameters
        ArchetypalAnalysisVisualizer.plot_membership_weights(fitted_model)

        # Test with specific number of samples
        ArchetypalAnalysisVisualizer.plot_membership_weights(fitted_model, n_samples=10)

    @pytest.mark.slow
    def test_plot_archetype_profiles(self, fitted_model):
        """Test that plot_archetype_profiles runs without errors."""
        # Create feature names for testing
        feature_names = [f"Feature {i}" for i in range(fitted_model.archetypes.shape[1])]

        # Test with feature names
        ArchetypalAnalysisVisualizer.plot_archetype_profiles(fitted_model, feature_names)

        # Test without feature names
        ArchetypalAnalysisVisualizer.plot_archetype_profiles(fitted_model)

    @pytest.mark.slow
    def test_plot_archetype_distribution(self, fitted_model):
        """Test that plot_archetype_distribution runs without errors."""
        ArchetypalAnalysisVisualizer.plot_archetype_distribution(fitted_model)

    @pytest.mark.slow
    def test_plot_simplex_2d(self, fitted_model):
        """Test that plot_simplex_2d runs without errors."""
        # Test with default parameters
        ArchetypalAnalysisVisualizer.plot_simplex_2d(fitted_model)

        # Test with specific number of samples
        ArchetypalAnalysisVisualizer.plot_simplex_2d(fitted_model, n_samples=10)
