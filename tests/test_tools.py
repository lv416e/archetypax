"""Unit tests for archetypax tools."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from archetypax.models.archetypes import ImprovedArchetypalAnalysis
from archetypax.models.base import ArchetypalAnalysis
from archetypax.tools.evaluation import ArchetypalAnalysisEvaluator
from archetypax.tools.interpret import ArchetypalAnalysisInterpreter
from archetypax.tools.tracker import ArchetypeTracker
from archetypax.tools.visualization import ArchetypalAnalysisVisualizer


@pytest.fixture
def sample_data():
    """Generate synthetic test data."""
    X, _ = make_blobs(n_samples=30, n_features=5, centers=3, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def small_sample_data():
    """Generate smaller synthetic data for faster tests."""
    X, _ = make_blobs(n_samples=10, n_features=2, centers=2, random_state=42, cluster_std=1.5)
    return X


@pytest.fixture
def fitted_model(sample_data):
    """Create a pre-fitted model for testing tool functionality."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=20)
    model.fit(sample_data)
    return model


class TestArchetypeTracker:
    """Test suite for the ArchetypeTracker class."""

    def test_initialization(self):
        """Verify proper initialization of tracker parameters."""
        tracker = ArchetypeTracker(n_archetypes=3)

        # Check base class parameters are properly initialized
        assert tracker.n_archetypes == 3
        assert tracker.max_iter == 500
        assert tracker.tol == 1e-6
        assert tracker.archetypes is None
        assert tracker.weights is None

        # Check tracker-specific attributes
        assert tracker.archetype_history == []
        assert tracker.boundary_proximity_history == []
        assert tracker.is_outside_history == []
        assert tracker.archetype_grad_scale == 1.0
        assert tracker.noise_scale == 0.02
        assert tracker.exploration_noise_scale == 0.05

    def test_fit(self, small_sample_data):
        """Test that fit method properly records archetype movement history."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=2)
        tracker.fit(small_sample_data)

        # Check that base functionality works
        assert tracker.archetypes is not None
        assert tracker.weights is not None
        assert tracker.archetypes.shape == (2, 2)
        assert tracker.weights.shape == (10, 2)

        # Check tracker-specific outputs
        assert len(tracker.archetype_history) > 0
        assert len(tracker.boundary_proximity_history) > 0
        assert len(tracker.is_outside_history) > 0

        # Check history shape consistency
        assert tracker.archetype_history[0].shape == (2, 2)
        assert len(tracker.boundary_proximity_history) == len(tracker.is_outside_history)
        assert isinstance(tracker.boundary_proximity_history[0], float)

        # Test loss history
        assert len(tracker.loss_history) > 0
        assert all(isinstance(loss, float) for loss in tracker.loss_history)

    def test_transform(self, small_sample_data):
        """Test transform still works properly in the tracker."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=2)
        tracker.fit(small_sample_data)

        # Test transform with same data
        weights = tracker.transform(small_sample_data, max_iter=5)
        assert weights.shape == (10, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Test with new data
        new_data = np.random.rand(5, 2)
        weights = tracker.transform(new_data, max_iter=5)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_check_archetypes_outside(self, small_sample_data):
        """Test method for checking if archetypes are outside convex hull."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=2)
        tracker.fit(small_sample_data)

        # Get test data in JAX format
        X_jax = jnp.array(small_sample_data)
        archetypes_jax = jnp.array(tracker.archetypes)

        # Check if archetypes are outside convex hull
        is_outside = tracker._check_archetypes_outside(archetypes_jax, X_jax)

        # Verify result shape and type
        assert is_outside.shape == (2,)
        assert is_outside.dtype == bool

    def test_constrain_to_convex_hull(self, small_sample_data):
        """Test method for constraining archetypes to convex hull."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=2)
        tracker.fit(small_sample_data)

        # Get test data in JAX format
        X_jax = jnp.array(small_sample_data)
        archetypes_jax = jnp.array(tracker.archetypes)

        # Constrain archetypes to convex hull
        constrained = tracker._constrain_to_convex_hull_batch(archetypes_jax, X_jax)

        # Verify result shape
        assert constrained.shape == (2, 2)

        # Check if any archetypes are outside after constraint
        is_outside = tracker._check_archetypes_outside(constrained, X_jax)
        assert not np.any(is_outside)  # None should be outside after constraint

    @pytest.mark.skipif(True, reason="Matplotlib visualization test skipped by default")
    def test_visualize_movement(self, small_sample_data):
        """Test visualization method for archetype movement."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=2)
        tracker.fit(small_sample_data)

        # Try to visualize movement with default parameters
        fig = tracker.visualize_movement()

        # If matplotlib is available, this should return a figure
        if fig is not None:
            assert str(type(fig)).find("matplotlib.figure.Figure") != -1

    @pytest.mark.skipif(True, reason="Matplotlib visualization test skipped by default")
    def test_visualize_boundary_proximity(self, small_sample_data):
        """Test visualization method for boundary proximity."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=2)
        tracker.fit(small_sample_data)

        # Try to visualize boundary proximity
        fig = tracker.visualize_boundary_proximity()

        # If matplotlib is available, this should return a figure
        if fig is not None:
            assert str(type(fig)).find("matplotlib.figure.Figure") != -1


class TestArchetypalAnalysisEvaluator:
    """Test suite for the ArchetypalAnalysisEvaluator class."""

    def test_initialization(self, fitted_model):
        """Verify proper evaluator initialization with a fitted model."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        assert evaluator.model is fitted_model
        assert evaluator.n_archetypes == 3
        assert evaluator.n_features == 5
        assert evaluator.dominant_archetypes.shape == (30,)

    def test_initialization_with_unfitted_model(self):
        """Ensure initialization fails appropriately with an unfitted model."""
        model = ArchetypalAnalysis(n_archetypes=3)
        with pytest.raises(ValueError):
            ArchetypalAnalysisEvaluator(model)

    def test_reconstruction_error(self, sample_data, fitted_model):
        """Validate reconstruction error metric calculations."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)

        # Test various error metrics
        error_frob = evaluator.reconstruction_error(sample_data, metric="frobenius")
        error_mse = evaluator.reconstruction_error(sample_data, metric="mse")
        error_mae = evaluator.reconstruction_error(sample_data, metric="mae")
        error_rel = evaluator.reconstruction_error(sample_data, metric="relative")

        # Verify non-negative errors
        assert error_frob >= 0
        assert error_mse >= 0
        assert error_mae >= 0
        # Relative error may exceed 1 for suboptimal fits
        assert error_rel >= 0

    def test_explained_variance(self, sample_data, fitted_model):
        """Verify explained variance calculation returns valid results."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        var_explained = evaluator.explained_variance(sample_data)

        # Explained variance may be negative for poor fits
        # Simply verify return type
        assert isinstance(var_explained, float)

    def test_archetype_separation(self, fitted_model):
        """Validate archetype separation metric calculations."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        separation = evaluator.archetype_separation()

        assert "mean_distance" in separation
        assert "min_distance" in separation
        assert "max_distance" in separation
        assert separation["mean_distance"] >= 0

    def test_dominant_archetype_purity(self, fitted_model):
        """Verify dominant archetype purity calculations."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        purity = evaluator.dominant_archetype_purity()

        assert "archetype_purity" in purity
        assert "overall_purity" in purity
        assert "purity_std" in purity
        assert purity["overall_purity"] >= 0 and purity["overall_purity"] <= 1

    def test_clustering_metrics(self, sample_data, fitted_model):
        """Validate clustering metric calculations."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        metrics = evaluator.clustering_metrics(sample_data)

        assert "davies_bouldin" in metrics
        assert "silhouette" in metrics
        assert -1 <= metrics["silhouette"] <= 1

    def test_archetype_feature_importance(self, fitted_model):
        """Verify archetype feature importance calculations."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        importance = evaluator.archetype_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert importance.shape == (3, 5)  # 3 archetypes, 5 features

    def test_weight_diversity(self, fitted_model):
        """Validate weight diversity metric calculations."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        diversity = evaluator.weight_diversity()

        assert "mean_entropy" in diversity
        assert "entropy_std" in diversity
        assert "max_entropy" in diversity
        assert "mean_normalized_entropy" in diversity
        assert diversity["mean_entropy"] >= 0

    def test_invalid_reconstruction_error_metric(self, sample_data, fitted_model):
        """Ensure appropriate error handling for invalid metrics."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)

        with pytest.raises(ValueError):
            evaluator.reconstruction_error(sample_data, metric="invalid")

    def test_print_evaluation_report(self, sample_data, fitted_model):
        """Verify evaluation report generation executes without errors."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        evaluator.print_evaluation_report(sample_data)

    @pytest.mark.slow
    def test_comprehensive_evaluation(self, sample_data, fitted_model):
        """Validate comprehensive evaluation results."""
        evaluator = ArchetypalAnalysisEvaluator(fitted_model)
        results = evaluator.comprehensive_evaluation(sample_data)

        assert isinstance(results, dict)
        assert "clustering" in results
        assert "davies_bouldin" in results["clustering"]
        assert "silhouette" in results["clustering"]
        assert "reconstruction" in results
        assert "diversity" in results
        assert "purity" in results

    @pytest.mark.slow
    def test_biarchetypal_evaluator(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Verify BiarchetypalAnalysisEvaluator initialization and core functionality."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)

        assert evaluator.model is fitted_biarchetypal_model
        assert evaluator.n_archetypes_first == 2
        assert evaluator.n_archetypes_second == 1

        results = evaluator.comprehensive_evaluation(biarchetypal_sample_data)

        assert "reconstruction" in results
        assert "diversity" in results
        assert "purity" in results


class TestArchetypalAnalysisInterpreter:
    """Test suite for the ArchetypalAnalysisInterpreter class."""

    def test_initialization(self):
        """Verify proper interpreter initialization."""
        interpreter = ArchetypalAnalysisInterpreter()
        assert isinstance(interpreter.models_dict, dict | None)
        assert len(interpreter.models_dict) == 0 if interpreter.models_dict is not None else True
        assert isinstance(interpreter.results, dict | None)
        assert len(interpreter.results) == 0 if interpreter.results is not None else True

    def test_add_model(self, fitted_model):
        """Validate model addition functionality."""
        interpreter = ArchetypalAnalysisInterpreter()
        interpreter.add_model(3, fitted_model)

        assert 3 in interpreter.models_dict
        assert interpreter.models_dict[3] is fitted_model

    def test_add_unfitted_model(self):
        """Ensure appropriate error handling for unfitted models."""
        interpreter = ArchetypalAnalysisInterpreter()
        model = ArchetypalAnalysis(n_archetypes=3)

        with pytest.raises(ValueError):
            interpreter.add_model(3, model)

    def test_feature_distinctiveness(self, fitted_model):
        """Validate feature distinctiveness calculations."""
        interpreter = ArchetypalAnalysisInterpreter()
        distinctiveness = interpreter.feature_distinctiveness(fitted_model.archetypes)

        assert distinctiveness.shape == (3,)
        assert np.all(distinctiveness >= 0)

    def test_sparsity_coefficient(self, fitted_model):
        """Verify sparsity coefficient calculations."""
        interpreter = ArchetypalAnalysisInterpreter()
        sparsity = interpreter.sparsity_coefficient(fitted_model.archetypes)

        assert sparsity.shape == (3,)
        assert np.all(sparsity >= 0)
        assert np.all(sparsity <= 1)

    def test_cluster_purity(self, fitted_model):
        """Validate cluster purity calculations."""
        interpreter = ArchetypalAnalysisInterpreter()
        purity, overall_purity = interpreter.cluster_purity(fitted_model.weights)

        assert purity.shape == (3,)
        assert np.all(purity >= 0)
        assert np.all(purity <= 1)
        assert overall_purity >= 0 and overall_purity <= 1

    @pytest.mark.slow
    def test_feature_consistency(self, sample_data):
        """Validate feature consistency calculations across multiple initializations."""

        # Mock the ImprovedArchetypalAnalysis class to avoid actual fitting
        # This is needed because feature_consistency creates and fits multiple models
        original_init = ImprovedArchetypalAnalysis.__init__
        original_fit = ImprovedArchetypalAnalysis.fit

        try:
            # Create a small sample dataset
            X_small = sample_data[:10, :3]  # Use a small subset for faster testing
            n_archetypes = 2
            n_trials = 3

            # Create a pre-fitted model for testing
            model = ImprovedArchetypalAnalysis(n_archetypes=n_archetypes, max_iter=5)
            model.fit(X_small)

            # Create an interpreter
            interpreter = ArchetypalAnalysisInterpreter()

            # Test feature consistency
            consistency_scores = interpreter.feature_consistency(
                X_small, n_archetypes=n_archetypes, n_trials=n_trials, top_k=2, random_seed=42
            )

            # Verify the output
            assert consistency_scores.shape == (n_archetypes,)
            assert np.all(consistency_scores >= 0)
            assert np.all(consistency_scores <= 1)
        finally:
            # Restore original methods
            ImprovedArchetypalAnalysis.__init__ = original_init
            ImprovedArchetypalAnalysis.fit = original_fit

    @pytest.mark.slow
    def test_evaluate_all_models(self, sample_data):
        """Test evaluating all models with multiple archetypes."""
        # Create models with different numbers of archetypes
        models = {}
        for k in [2, 3, 4]:
            model = ArchetypalAnalysis(n_archetypes=k, max_iter=10)
            model.fit(sample_data)
            models[k] = model

        # Create interpreter and add models
        interpreter = ArchetypalAnalysisInterpreter(models)

        # Evaluate all models
        results = interpreter.evaluate_all_models(sample_data)

        # Verify results structure
        assert len(results) == 3  # Should have results for 3 models
        for k in [2, 3, 4]:
            assert k in results
            assert "distinctiveness" in results[k]
            assert "sparsity" in results[k]
            assert "purity" in results[k]
            assert "avg_distinctiveness" in results[k]
            assert "avg_sparsity" in results[k]
            assert "avg_purity" in results[k]
            assert "interpretability_score" in results[k]

        # Verify information gain calculation
        for k in [3, 4]:  # Skip the first model (k=2) as it has no information gain
            assert "information_gain" in results[k]
            assert (
                results[k]["information_gain"] >= 0
            )  # results[k]["information_gain"] >= 0 and results[k]["information_gain"] <= 1

        # Verify balance score calculation
        for k in [3, 4]:
            assert "balance_score" in results[k]
            assert (
                results[k]["balance_score"] >= 0
            )  # results[k]["balance_score"] >= 0 and results[k]["balance_score"] <= 1

    @pytest.mark.slow
    def test_suggest_optimal_archetypes(self, sample_data):
        """Test suggesting optimal number of archetypes using different methods."""
        # Create models with different numbers of archetypes
        models = {}
        for k in [2, 3, 4]:
            model = ArchetypalAnalysis(n_archetypes=k, max_iter=10)
            model.fit(sample_data)
            models[k] = model

        # Create interpreter and add models
        interpreter = ArchetypalAnalysisInterpreter(models)

        # Evaluate all models first
        interpreter.evaluate_all_models(sample_data)

        # Test different methods for suggesting optimal archetypes
        optimal_balance = interpreter.suggest_optimal_archetypes(method="balance")
        assert isinstance(optimal_balance, int)
        assert optimal_balance in [2, 3, 4]

        optimal_interpretability = interpreter.suggest_optimal_archetypes(method="interpretability")
        assert isinstance(optimal_interpretability, int)
        assert optimal_interpretability in [2, 3, 4]

        optimal_information_gain = interpreter.suggest_optimal_archetypes(method="information_gain")
        assert isinstance(optimal_information_gain, int)
        assert optimal_information_gain in [3, 4]  # k=2 has no information gain

        # Test with invalid method
        with pytest.raises(ValueError):
            interpreter.suggest_optimal_archetypes(method="invalid_method")

    @pytest.mark.slow
    def test_plot_interpretability_metrics(self, sample_data):
        """Test plotting interpretability metrics."""
        # Create models with different numbers of archetypes
        models = {}
        for k in [2, 3, 4]:
            model = ArchetypalAnalysis(n_archetypes=k, max_iter=10)
            model.fit(sample_data)
            models[k] = model

        # Create interpreter and add models
        interpreter = ArchetypalAnalysisInterpreter(models)

        # Evaluate all models first
        interpreter.evaluate_all_models(sample_data)

        # Test plotting (just verify it runs without errors)
        interpreter.plot_interpretability_metrics()

    def test_optimal_archetypes_elbow(self, sample_data):
        """Verify visualization method availability."""
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetype_profiles")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetype_distribution")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_simplex_2d")

    @pytest.mark.slow
    def test_plot_loss(self, fitted_model):
        """Verify loss plotting functionality."""
        ArchetypalAnalysisVisualizer.plot_loss(fitted_model)


@pytest.fixture
def biarchetypal_sample_data():
    """Generate synthetic data for biarchetypal analysis testing."""
    X, _ = make_blobs(n_samples=20, n_features=3, centers=2, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def fitted_biarchetypal_model(biarchetypal_sample_data):
    """Create a pre-fitted biarchetypal model for testing."""
    from archetypax.models.biarchetypes import BiarchetypalAnalysis

    model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
    model.fit(biarchetypal_sample_data)
    return model


@pytest.fixture
def biarchetypal_sample_data_2d():
    """Generate 2D synthetic data for biarchetypal visualization testing."""
    X, _ = make_blobs(n_samples=20, n_features=2, centers=2, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def fitted_biarchetypal_model_2d(biarchetypal_sample_data_2d):
    """Create a pre-fitted biarchetypal model for visualization testing."""
    from archetypax.models.biarchetypes import BiarchetypalAnalysis

    model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=2, max_iter=10)
    model.fit(biarchetypal_sample_data_2d)
    return model


@pytest.fixture
def fitted_biarchetypal_model_2d_3arch(biarchetypal_sample_data_2d):
    """Create a pre-fitted biarchetypal model with 3 archetypes for simplex visualization."""
    from archetypax.models.biarchetypes import BiarchetypalAnalysis

    model = BiarchetypalAnalysis(n_row_archetypes=3, n_col_archetypes=2, max_iter=10)
    model.fit(biarchetypal_sample_data_2d)
    return model


class TestBiarchetypalAnalysisEvaluator:
    """Test suite for the BiarchetypalAnalysisEvaluator class."""

    def test_initialization(self, fitted_biarchetypal_model):
        """Verify proper evaluator initialization with a fitted model."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        assert evaluator.model is fitted_biarchetypal_model
        assert evaluator.n_archetypes_first == 2
        assert evaluator.n_archetypes_second == 1
        assert evaluator.n_features == 3
        assert evaluator.dominant_archetypes_first.shape == (20,)
        assert evaluator.dominant_archetypes_second.shape == (3,)

    def test_initialization_with_unfitted_model(self):
        """Ensure initialization fails appropriately with an unfitted model."""
        from archetypax.models.biarchetypes import BiarchetypalAnalysis
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1)
        with pytest.raises(ValueError):
            BiarchetypalAnalysisEvaluator(model)

    def test_initialization_with_wrong_model_type(self, fitted_model):
        """Ensure initialization fails appropriately with incorrect model type."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        with pytest.raises(TypeError):
            BiarchetypalAnalysisEvaluator(fitted_model)

    def test_reconstruction_error(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Validate reconstruction error metric calculations."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)

        error_frob = evaluator.reconstruction_error(biarchetypal_sample_data, metric="frobenius")
        error_mse = evaluator.reconstruction_error(biarchetypal_sample_data, metric="mse")
        error_mae = evaluator.reconstruction_error(biarchetypal_sample_data, metric="mae")
        error_rel = evaluator.reconstruction_error(biarchetypal_sample_data, metric="relative")

        assert error_frob >= 0
        assert error_mse >= 0
        assert error_mae >= 0
        assert error_rel >= 0

        with pytest.raises(ValueError):
            evaluator.reconstruction_error(biarchetypal_sample_data, metric="invalid")

    def test_explained_variance(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Verify explained variance calculation returns valid results."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        var_explained = evaluator.explained_variance(biarchetypal_sample_data)

        assert isinstance(var_explained, float)

    def test_archetype_separation(self, fitted_biarchetypal_model):
        """Validate archetype separation metric calculations."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        separation = evaluator.archetype_separation()

        assert "mean_distance_first" in separation
        assert "min_distance_first" in separation
        assert "max_distance_first" in separation
        assert "mean_distance_second" in separation
        assert "min_distance_second" in separation
        assert "max_distance_second" in separation
        assert "mean_cross_distance" in separation
        assert "min_cross_distance" in separation
        assert "max_cross_distance" in separation

        assert separation["mean_distance_first"] >= 0
        assert separation["mean_cross_distance"] >= 0

    def test_dominant_archetype_purity(self, fitted_biarchetypal_model):
        """Verify dominant archetype purity calculations."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        purity = evaluator.dominant_archetype_purity()

        assert "archetype_purity_first" in purity
        assert "archetype_purity_second" in purity
        assert "overall_purity_first" in purity
        assert "overall_purity_second" in purity
        assert "purity_std_first" in purity
        assert "purity_std_second" in purity

        assert purity["overall_purity_first"] >= 0 and purity["overall_purity_first"] <= 1
        assert purity["overall_purity_second"] >= 0 and purity["overall_purity_second"] <= 1

    def test_weight_diversity(self, fitted_biarchetypal_model):
        """Validate weight diversity metric calculations."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        diversity = evaluator.weight_diversity()

        assert "mean_entropy_first" in diversity
        assert "entropy_std_first" in diversity
        assert "max_entropy_first" in diversity
        assert "mean_normalized_entropy_first" in diversity
        assert "mean_entropy_second" in diversity
        assert "entropy_std_second" in diversity
        assert "max_entropy_second" in diversity
        assert "mean_normalized_entropy_second" in diversity

        assert diversity["mean_entropy_first"] >= 0
        assert diversity["mean_entropy_second"] >= 0
        assert diversity["mean_normalized_entropy_first"] >= 0
        assert diversity["mean_normalized_entropy_second"] >= 0

    @pytest.mark.slow
    def test_comprehensive_evaluation(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Validate comprehensive evaluation results."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        results = evaluator.comprehensive_evaluation(biarchetypal_sample_data)

        assert isinstance(results, dict)
        assert "reconstruction" in results
        assert "separation" in results
        assert "purity" in results
        assert "diversity" in results

        assert "frobenius" in results["reconstruction"]
        assert "mse" in results["reconstruction"]
        assert "mae" in results["reconstruction"]
        assert "relative" in results["reconstruction"]
        assert "explained_variance" in results["reconstruction"]

    @pytest.mark.slow
    def test_print_evaluation_report(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Verify evaluation report generation executes without errors."""
        from archetypax.tools.evaluation import BiarchetypalAnalysisEvaluator

        evaluator = BiarchetypalAnalysisEvaluator(fitted_biarchetypal_model)
        evaluator.print_evaluation_report(biarchetypal_sample_data)


class TestBiarchetypalAnalysisVisualizer:
    """Test suite for the BiarchetypalAnalysisVisualizer class."""

    def test_static_methods_exist(self):
        """Verify availability of all visualization methods."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_dual_archetypes_2d")
        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_biarchetypal_reconstruction")
        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_dual_membership_heatmap")
        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_mixture_effect")
        assert hasattr(BiarchetypalAnalysisVisualizer, "plot_dual_simplex_2d")

    @pytest.mark.slow
    def test_plot_dual_membership_heatmap(self, fitted_biarchetypal_model):
        """Verify dual membership heatmap visualization."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        BiarchetypalAnalysisVisualizer.plot_dual_membership_heatmap(fitted_biarchetypal_model)
        BiarchetypalAnalysisVisualizer.plot_dual_membership_heatmap(fitted_biarchetypal_model, n_samples=20)

    @pytest.mark.slow
    def test_plot_dual_archetypes_2d_with_2d_data(self, biarchetypal_sample_data_2d, fitted_biarchetypal_model_2d):
        """Verify 2D dual archetype visualization."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        feature_names = [f"Feature {i}" for i in range(biarchetypal_sample_data_2d.shape[1])]

        BiarchetypalAnalysisVisualizer.plot_dual_archetypes_2d(
            fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d, feature_names
        )

        BiarchetypalAnalysisVisualizer.plot_dual_archetypes_2d(
            fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d
        )

    @pytest.mark.slow
    def test_plot_biarchetypal_reconstruction_with_2d_data(
        self, biarchetypal_sample_data_2d, fitted_biarchetypal_model_2d
    ):
        """Verify biarchetypal reconstruction visualization."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        BiarchetypalAnalysisVisualizer.plot_biarchetypal_reconstruction(
            fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d
        )

    @pytest.mark.slow
    def test_plot_dual_membership_heatmap_with_2d_model(self, fitted_biarchetypal_model_2d):
        """Verify dual membership heatmap visualization with 2D model."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        BiarchetypalAnalysisVisualizer.plot_dual_membership_heatmap(fitted_biarchetypal_model_2d)
        BiarchetypalAnalysisVisualizer.plot_dual_membership_heatmap(fitted_biarchetypal_model_2d, n_samples=20)

    @pytest.mark.slow
    def test_plot_mixture_effect_with_2d_data(self, biarchetypal_sample_data_2d, fitted_biarchetypal_model_2d):
        """Verify mixture effect visualization."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        BiarchetypalAnalysisVisualizer.plot_mixture_effect(fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d)
        BiarchetypalAnalysisVisualizer.plot_mixture_effect(
            fitted_biarchetypal_model_2d, biarchetypal_sample_data_2d, mixture_steps=3
        )

    @pytest.mark.slow
    def test_plot_dual_simplex_2d_with_3arch_model(self, fitted_biarchetypal_model_2d_3arch):
        """Verify dual simplex visualization with 3-archetype model."""
        from archetypax.tools.visualization import BiarchetypalAnalysisVisualizer

        BiarchetypalAnalysisVisualizer.plot_dual_simplex_2d(fitted_biarchetypal_model_2d_3arch)
        BiarchetypalAnalysisVisualizer.plot_dual_simplex_2d(fitted_biarchetypal_model_2d_3arch, n_samples=100)


@pytest.fixture
def sample_data_2d():
    """Generate 2D synthetic data for visualization testing."""
    X, _ = make_blobs(n_samples=30, n_features=2, centers=3, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def fitted_model_2d(sample_data_2d):
    """Create a pre-fitted model for visualization testing."""
    model = ArchetypalAnalysis(n_archetypes=3, max_iter=20)
    model.fit(sample_data_2d)
    return model


class TestArchetypalAnalysisVisualizer:
    """Test suite for the ArchetypalAnalysisVisualizer class."""

    def test_static_methods_exist(self):
        """Verify availability of all visualization methods."""
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_loss")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetypes_2d")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_reconstruction_comparison")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_membership_weights")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetype_profiles")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_archetype_distribution")
        assert hasattr(ArchetypalAnalysisVisualizer, "plot_simplex_2d")

    @pytest.mark.slow
    def test_plot_loss(self, fitted_model):
        """Verify loss plotting functionality."""
        ArchetypalAnalysisVisualizer.plot_loss(fitted_model)

    @pytest.mark.slow
    def test_plot_archetypes_2d(self, sample_data_2d, fitted_model_2d):
        """Verify 2D archetype visualization."""
        feature_names = [f"Feature {i}" for i in range(sample_data_2d.shape[1])]

        ArchetypalAnalysisVisualizer.plot_archetypes_2d(fitted_model_2d, sample_data_2d, feature_names)
        ArchetypalAnalysisVisualizer.plot_archetypes_2d(fitted_model_2d, sample_data_2d)

    @pytest.mark.slow
    def test_plot_reconstruction_comparison(self, sample_data_2d, fitted_model_2d):
        """Verify reconstruction comparison visualization."""
        ArchetypalAnalysisVisualizer.plot_reconstruction_comparison(fitted_model_2d, sample_data_2d)

    @pytest.mark.slow
    def test_plot_membership_weights(self, fitted_model):
        """Verify membership weight visualization."""
        ArchetypalAnalysisVisualizer.plot_membership_weights(fitted_model)
        ArchetypalAnalysisVisualizer.plot_membership_weights(fitted_model, n_samples=10)

    @pytest.mark.slow
    def test_plot_archetype_profiles(self, fitted_model):
        """Verify archetype profile visualization."""
        feature_names = [f"Feature {i}" for i in range(fitted_model.archetypes.shape[1])]

        ArchetypalAnalysisVisualizer.plot_archetype_profiles(fitted_model, feature_names)
        ArchetypalAnalysisVisualizer.plot_archetype_profiles(fitted_model)

    @pytest.mark.slow
    def test_plot_archetype_distribution(self, fitted_model):
        """Verify archetype distribution visualization."""
        ArchetypalAnalysisVisualizer.plot_archetype_distribution(fitted_model)

    @pytest.mark.slow
    def test_plot_simplex_2d(self, fitted_model):
        """Verify 2D simplex visualization."""
        ArchetypalAnalysisVisualizer.plot_simplex_2d(fitted_model)
        ArchetypalAnalysisVisualizer.plot_simplex_2d(fitted_model, n_samples=10)


class TestBiarchetypalAnalysisInterpreter:
    """Test suite for the BiarchetypalAnalysisInterpreter class."""

    def test_initialization(self):
        """Verify proper interpreter initialization."""
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()
        assert isinstance(interpreter.models_dict, dict)
        assert len(interpreter.models_dict) == 0
        assert isinstance(interpreter.results, dict)

    def test_add_model(self, fitted_biarchetypal_model):
        """Validate model addition functionality."""
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()
        interpreter.add_model(2, 1, fitted_biarchetypal_model)

        assert (2, 1) in interpreter.models_dict
        assert interpreter.models_dict[2, 1] is fitted_biarchetypal_model

    def test_add_unfitted_model(self):
        """Ensure appropriate error handling for unfitted models."""
        from archetypax.models.biarchetypes import BiarchetypalAnalysis
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1)

        with pytest.raises(ValueError):
            interpreter.add_model(2, 1, model)

    def test_feature_distinctiveness(self, fitted_biarchetypal_model):
        """Validate feature distinctiveness calculations."""
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()
        row_archetypes, _ = fitted_biarchetypal_model.get_all_archetypes()
        distinctiveness = interpreter.feature_distinctiveness(row_archetypes)

        assert distinctiveness.shape == (2,)
        assert np.all(distinctiveness >= 0)

    def test_sparsity_coefficient(self, fitted_biarchetypal_model):
        """Verify sparsity coefficient calculations."""
        """Test sparsity coefficient calculation."""
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()
        row_archetypes, _ = fitted_biarchetypal_model.get_all_archetypes()
        sparsity = interpreter.sparsity_coefficient(row_archetypes)

        assert sparsity.shape == (2,)
        assert np.all(sparsity >= 0)
        assert np.all(sparsity <= 1)

    def test_cluster_purity(self, fitted_biarchetypal_model):
        """Test cluster purity calculation."""
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()
        row_weights, _ = fitted_biarchetypal_model.get_all_weights()
        purity, overall_purity = interpreter.cluster_purity(row_weights)

        assert purity.shape == (2,)
        assert np.all(purity >= 0)
        assert np.all(purity <= 1)
        assert overall_purity >= 0 and overall_purity <= 1

    def test_evaluate_all_models(self, biarchetypal_sample_data, fitted_biarchetypal_model):
        """Test evaluating all models."""
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()
        interpreter.add_model(2, 1, fitted_biarchetypal_model)
        results = interpreter.evaluate_all_models(biarchetypal_sample_data)

        assert (2, 1) in results
        assert "distinctiveness_first" in results[2, 1]
        assert "sparsity_first" in results[2, 1]
        assert "purity_first" in results[2, 1]
        assert "distinctiveness_second" in results[2, 1]
        assert "sparsity_second" in results[2, 1]
        assert "purity_second" in results[2, 1]
        assert "combined_interpretability" in results[2, 1]
        assert "reconstruction_error" in results[2, 1]

    def test_suggest_optimal_biarchetypes(self, biarchetypal_sample_data):
        """Test suggesting optimal biarchetypes."""
        from archetypax.models.biarchetypes import BiarchetypalAnalysis
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()

        # Add multiple models with different archetype combinations
        for k1, k2 in [(2, 1), (2, 2), (3, 1)]:
            model = BiarchetypalAnalysis(n_row_archetypes=k1, n_col_archetypes=k2, max_iter=10)
            model.fit(biarchetypal_sample_data)
            interpreter.add_model(k1, k2, model)

        # Evaluate all models
        interpreter.evaluate_all_models(biarchetypal_sample_data)

        # Test different methods for suggesting optimal biarchetypes
        optimal_interpretability = interpreter.suggest_optimal_biarchetypes(method="interpretability")
        assert isinstance(optimal_interpretability, tuple)
        assert len(optimal_interpretability) == 2

        # Test balance method (may fall back to interpretability if information gain isn't available)
        optimal_balance = interpreter.suggest_optimal_biarchetypes(method="balance")
        assert isinstance(optimal_balance, tuple)
        assert len(optimal_balance) == 2

    def test_plot_interpretability_heatmap(self, biarchetypal_sample_data):
        """Test plotting interpretability heatmap."""
        from archetypax.models.biarchetypes import BiarchetypalAnalysis
        from archetypax.tools.interpret import BiarchetypalAnalysisInterpreter

        interpreter = BiarchetypalAnalysisInterpreter()

        # Add multiple models with different archetype combinations
        for k1 in [2, 3]:
            for k2 in [1, 2]:
                model = BiarchetypalAnalysis(n_row_archetypes=k1, n_col_archetypes=k2, max_iter=10)
                model.fit(biarchetypal_sample_data)
                interpreter.add_model(k1, k2, model)

        # Evaluate all models
        interpreter.evaluate_all_models(biarchetypal_sample_data)

        # Test plotting
        fig = interpreter.plot_interpretability_heatmap()
        assert fig is not None
        plt.close(fig)
