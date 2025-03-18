"""Unit tests for archetypax models."""

import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from archetypax.models.archetypes import ArchetypeTracker, ImprovedArchetypalAnalysis
from archetypax.models.base import ArchetypalAnalysis
from archetypax.models.biarchetypes import BiarchetypalAnalysis


@pytest.fixture
def sample_data():
    """Generate synthetic data for testing purposes."""
    X, _ = make_blobs(n_samples=50, n_features=5, centers=3, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture
def small_sample_data():
    """Generate smaller synthetic data for faster tests."""
    X, _ = make_blobs(n_samples=20, n_features=3, centers=2, random_state=42, cluster_std=2.0)
    return X


@pytest.fixture(
    params=[
        (ArchetypalAnalysis, {"n_archetypes": 2}),
        (ImprovedArchetypalAnalysis, {"n_archetypes": 2}),
        (BiarchetypalAnalysis, {"n_row_archetypes": 2, "n_col_archetypes": 1}),
    ]
)
def model_class_and_params(request):
    """Parametrized fixture providing model classes and their initialization parameters."""
    return request.param


class TestArchetypalAnalysis:
    """Test suite for the base ArchetypalAnalysis class."""

    def test_initialization(self):
        """Verify proper initialization of model parameters."""
        model = ArchetypalAnalysis(n_archetypes=3)
        assert model.n_archetypes == 3
        assert model.max_iter == 500
        assert model.tol == 1e-6
        assert model.archetypes is None
        assert model.weights is None
        assert len(model.loss_history) == 0

    @pytest.mark.slow
    def test_fit(self, sample_data):
        """Validate model fitting and output characteristics."""
        model = ArchetypalAnalysis(n_archetypes=3, max_iter=20)
        model.fit(sample_data)

        # Ensure proper attribute initialization post-fitting
        assert model.archetypes is not None
        assert model.weights is not None
        assert len(model.loss_history) > 0

        # Validate matrix dimensions
        assert model.archetypes.shape == (3, 5)
        assert model.weights.shape == (50, 3)

        # Confirm simplex constraint adherence
        assert np.allclose(np.sum(model.weights, axis=1), 1.0)

    def test_transform(self, small_sample_data):
        """Ensure transform operation yields valid weight matrices."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        weights = model.transform(small_sample_data)
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_fit_transform(self, small_sample_data):
        """Validate combined fit and transform functionality."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        weights = model.fit_transform(small_sample_data)

        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_reconstruct(self, small_sample_data):
        """Verify reconstruction dimensionality matches input data."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    def test_get_loss_history(self, small_sample_data):
        """Examine loss history characteristics after model fitting."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        loss_history = model.get_loss_history()
        assert isinstance(loss_history, list)
        assert len(loss_history) > 0
        assert all(isinstance(loss, float) for loss in loss_history)

    @pytest.mark.slow
    def test_fit_with_normalization(self, small_sample_data):
        """Evaluate model performance with data normalization."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        # Verify normalization parameters
        assert model.X_mean is not None
        assert model.X_std is not None

        # Confirm proper attribute initialization
        assert model.archetypes is not None
        assert model.weights is not None

        # Validate reconstruction dimensions
        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    def test_transform_new_data(self, small_sample_data):
        """Assess transformation of previously unseen data."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Generate novel test data
        new_data = np.random.rand(5, 3)

        weights = model.transform(new_data)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_reconstruct_new_data(self, small_sample_data):
        """Evaluate reconstruction of previously unseen data."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Generate novel test data
        new_data = np.random.rand(5, 3)

        reconstructed = model.reconstruct(new_data)
        assert reconstructed.shape == new_data.shape

    def test_error_before_fit(self):
        """Verify appropriate error handling for operations on unfitted model."""
        model = ArchetypalAnalysis(n_archetypes=3)

        # Generate test data
        X = np.random.rand(10, 5)

        # Validate transform error
        with pytest.raises(ValueError, match="Model must be fitted before transform"):
            model.transform(X)

        # Validate reconstruct error
        with pytest.raises(ValueError, match="Model must be fitted before reconstruct"):
            model.reconstruct()


class TestImprovedArchetypalAnalysis:
    """Test suite for the ImprovedArchetypalAnalysis class."""

    def test_initialization(self):
        """Verify proper initialization of improved model parameters."""
        model = ImprovedArchetypalAnalysis(n_archetypes=3)
        assert model.n_archetypes == 3
        assert model.max_iter == 500
        assert model.archetypes is None
        assert model.weights is None

    @pytest.mark.slow
    def test_fit(self, small_sample_data):
        """Validate improved model fitting and output characteristics."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Ensure proper attribute initialization
        assert model.archetypes is not None
        assert model.weights is not None

        # Validate matrix dimensions
        assert model.archetypes.shape == (2, 3)
        assert model.weights.shape == (20, 2)

        # Confirm simplex constraint adherence
        assert np.allclose(np.sum(model.weights, axis=1), 1.0)

    def test_transform(self, small_sample_data):
        """Ensure transform operation yields valid weight matrices."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        weights = model.transform(small_sample_data)
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_transform_new_data(self, small_sample_data):
        """Assess transformation of previously unseen data."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Generate novel test data
        new_data = np.random.rand(5, 3)

        weights = model.transform(new_data)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_transform_with_adam(self, small_sample_data):
        """Test transform with Adam optimizer."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        weights = model.transform(small_sample_data, method="adam", max_iter=30)
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Test with new data
        new_data = np.random.rand(5, 3)
        weights = model.transform(new_data, method="adam", max_iter=30)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_transform_with_sgd(self, small_sample_data):
        """Test transform with SGD optimizer."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        weights = model.transform(small_sample_data, method="sgd", max_iter=50)
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Test with new data
        new_data = np.random.rand(5, 3)
        weights = model.transform(new_data, method="sgd", max_iter=50)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_transform_with_lbfgs(self, small_sample_data):
        """Test transform with L-BFGS optimizer."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        weights = model.transform(small_sample_data, method="lbfgs", max_iter=20)
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Test with new data
        new_data = np.random.rand(5, 3)
        weights = model.transform(new_data, method="lbfgs", max_iter=20)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_transform_with_adaptive(self, small_sample_data):
        """Test transform with adaptive method selection."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Small dataset - should choose lbfgs
        weights = model.transform(small_sample_data, method="adaptive")
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Verify different size datasets use different methods
        # Create medium and large datasets for testing adaptive selection
        medium_data = np.random.rand(1500, 3)
        large_data = np.random.rand(15000, 3)

        # These should exercise different code paths
        # Medium dataset should select adam
        weights_medium = model.transform(medium_data, method="adaptive", max_iter=5)
        assert weights_medium.shape == (1500, 2)
        assert np.allclose(np.sum(weights_medium, axis=1), 1.0)

        # Large dataset should select sgd
        weights_large = model.transform(large_data, method="adaptive", max_iter=5)
        assert weights_large.shape == (15000, 2)
        assert np.allclose(np.sum(weights_large, axis=1), 1.0)

    def test_transform_convergence(self, small_sample_data):
        """Test early convergence in transform methods."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Test with very loose tolerance that should converge quickly
        weights_loose = model.transform(small_sample_data, method="adam", tol=1e-1, max_iter=100)
        assert weights_loose.shape == (20, 2)

        # Test with very strict tolerance that should require more iterations
        weights_strict = model.transform(small_sample_data, method="adam", tol=1e-10, max_iter=100)
        assert weights_strict.shape == (20, 2)

        # Both should satisfy simplex constraints regardless of convergence
        assert np.allclose(np.sum(weights_loose, axis=1), 1.0)
        assert np.allclose(np.sum(weights_strict, axis=1), 1.0)

    def test_kwargs_passing(self, small_sample_data):
        """Test kwargs are properly passed through the API."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)

        # Test fit_transform with kwargs
        weights = model.fit_transform(small_sample_data, normalize=True, method="adam", max_iter=15, tol=1e-4)

        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Test separate fit and transform with kwargs
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        weights = model.transform(small_sample_data, method="lbfgs", max_iter=15, tol=1e-4)

        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    @pytest.mark.slow
    def test_fit_transform(self, small_sample_data):
        """Validate combined fit and transform functionality."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        weights = model.fit_transform(small_sample_data)

        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Verify model fitting
        assert model.archetypes is not None
        assert model.weights is not None

    def test_reconstruct(self, small_sample_data):
        """Verify reconstruction dimensionality matches input data."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    def test_reconstruct_new_data(self, small_sample_data):
        """Evaluate reconstruction of previously unseen data."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Generate novel test data
        new_data = np.random.rand(5, 3)

        reconstructed = model.reconstruct(new_data)
        assert reconstructed.shape == new_data.shape

    @pytest.mark.slow
    def test_fit_with_normalization(self, small_sample_data):
        """Evaluate model performance with data normalization."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        # Verify normalization parameters
        assert model.X_mean is not None
        assert model.X_std is not None

        # Confirm proper attribute initialization
        assert model.archetypes is not None
        assert model.weights is not None

        # Validate reconstruction dimensions
        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    def test_error_before_fit(self):
        """Verify appropriate error handling for operations on unfitted model."""
        model = ImprovedArchetypalAnalysis(n_archetypes=3)

        # Generate test data
        X = np.random.rand(10, 5)

        # Validate transform error
        with pytest.raises(ValueError, match="Model must be fitted before transform"):
            model.transform(X)

        # Validate reconstruct error
        with pytest.raises(ValueError, match="Model must be fitted before reconstruct"):
            model.reconstruct()


class TestBiarchetypalAnalysis:
    """Test suite for the BiarchetypalAnalysis class."""

    def test_initialization(self):
        """Verify proper initialization of biarchetypal model parameters."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1)
        assert model.n_row_archetypes == 2
        assert model.n_col_archetypes == 1
        assert model.max_iter == 500
        assert model.archetypes is None
        assert model.weights is None

    @pytest.mark.slow
    def test_fit(self, small_sample_data):
        """Validate biarchetypal model fitting and output characteristics."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data)

        # Ensure proper attribute initialization
        assert model.archetypes is not None
        assert model.weights is not None

        # Extract archetype components
        row_archetypes = model.get_row_archetypes()
        col_archetypes = model.get_col_archetypes()
        row_weights = model.get_row_weights()
        col_weights = model.get_col_weights()

        # Validate matrix dimensions
        assert row_archetypes.shape == (2, 3)
        assert col_archetypes.shape[1] == 3
        assert row_weights.shape == (20, 2)
        assert col_weights.shape == (1, 3)

    def test_transform(self, small_sample_data):
        """Ensure transform operation yields valid weight matrices."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data)

        alpha, gamma = model.transform(small_sample_data)

        # Validate dimensions
        assert alpha.shape == (20, 2)
        assert gamma.shape[0] == 1

        # Confirm simplex constraint adherence
        assert np.allclose(np.sum(alpha, axis=1), 1.0)
        assert np.allclose(np.sum(gamma, axis=0), 1.0)

    @pytest.mark.slow
    def test_fit_transform(self, small_sample_data):
        """Validate combined fit and transform functionality."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        alpha, gamma = model.fit_transform(small_sample_data)

        # Validate dimensions
        assert alpha.shape == (20, 2)
        assert gamma.shape[0] == 1

        # Confirm simplex constraint adherence
        assert np.allclose(np.sum(alpha, axis=1), 1.0)
        assert np.allclose(np.sum(gamma, axis=0), 1.0)

        # Verify model fitting
        assert model.archetypes is not None
        assert model.weights is not None
        assert model.biarchetypes is not None
        assert model.beta is not None
        assert model.theta is not None

    def test_reconstruct(self, small_sample_data):
        """Verify reconstruction functionality."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data)

        # Test default reconstruction
        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

        # Test reconstruction with new data
        new_data = np.random.rand(5, small_sample_data.shape[1])
        reconstructed_new = model.reconstruct(new_data)
        assert reconstructed_new.shape == new_data.shape

    def test_get_row_archetypes(self, small_sample_data):
        """Validate row archetype extraction."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data)

        row_archetypes = model.get_row_archetypes()
        assert row_archetypes.shape == (2, 3)

    def test_get_col_archetypes(self, small_sample_data):
        """Validate column archetype extraction."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data)

        col_archetypes = model.get_col_archetypes()
        assert col_archetypes.shape[1] == 3

    def test_get_row_weights(self, small_sample_data):
        """Validate row weight extraction and constraints."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data)

        row_weights = model.get_row_weights()
        assert row_weights.shape == (20, 2)
        assert np.allclose(np.sum(row_weights, axis=1), 1.0)

    def test_get_col_weights(self, small_sample_data):
        """Validate column weight extraction and constraints."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data)

        col_weights = model.get_col_weights()
        assert col_weights.shape == (1, 3)
        assert np.allclose(np.sum(col_weights, axis=0), 1.0)

    def test_error_before_fit(self):
        """Verify appropriate error handling for operations on unfitted model."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1)
        X = np.random.rand(10, 3)

        with pytest.raises(ValueError, match="Model must be fitted before getting row archetypes"):
            model.get_row_archetypes()

        with pytest.raises(ValueError, match="Model must be fitted before getting column archetypes"):
            model.get_col_archetypes()

        with pytest.raises(ValueError, match="Model must be fitted before getting row weights"):
            model.get_row_weights()

        with pytest.raises(ValueError, match="Model must be fitted before getting column weights"):
            model.get_col_weights()

        with pytest.raises(ValueError, match="Model must be fitted before reconstruction"):
            model.reconstruct()

        with pytest.raises(ValueError, match="Model must be fitted before transform"):
            model.transform(X)

    @pytest.mark.slow
    def test_fit_with_normalization(self, small_sample_data):
        """Evaluate model performance with data normalization."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        # Verify normalization parameters
        assert model.X_mean is not None
        assert model.X_std is not None

        # Confirm proper attribute initialization
        assert model.archetypes is not None
        assert model.weights is not None

        # Validate reconstruction dimensions
        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    @pytest.mark.slow
    def test_transform_with_normalization(self, small_sample_data):
        """Evaluate transformation with normalized data."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        # Transform original data
        alpha, gamma = model.transform(small_sample_data)

        # Validate dimensions and constraints
        assert alpha.shape == (20, 2)
        assert gamma.shape == (1, 3)
        assert np.allclose(np.sum(alpha, axis=1), 1.0)
        assert np.allclose(np.sum(gamma, axis=0), 1.0)

        # Test with new data
        new_data = np.random.rand(5, small_sample_data.shape[1])
        alpha_new, gamma_new = model.transform(new_data)

        # Validate dimensions and constraints for new data
        assert alpha_new.shape == (5, 2)
        assert gamma_new.shape == (1, 3)
        assert np.allclose(np.sum(alpha_new, axis=1), 1.0)
        assert np.allclose(np.sum(gamma_new, axis=0), 1.0)

    def test_transform_new_data(self, small_sample_data):
        """Assess transformation of previously unseen data."""
        model = BiarchetypalAnalysis(n_row_archetypes=2, n_col_archetypes=1, max_iter=10)
        model.fit(small_sample_data)

        # Generate novel test data
        new_data = np.random.rand(5, small_sample_data.shape[1])

        alpha_new, gamma_new = model.transform(new_data)

        # Validate dimensions
        assert alpha_new.shape == (5, 2)
        assert gamma_new.shape == (1, 3)

        # Confirm simplex constraint adherence
        assert np.allclose(np.sum(alpha_new, axis=1), 1.0)
        assert np.allclose(np.sum(gamma_new, axis=0), 1.0)


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
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=10)
        tracker.fit(small_sample_data)

        # Check that base functionality works
        assert tracker.archetypes is not None
        assert tracker.weights is not None
        assert tracker.archetypes.shape == (2, 3)
        assert tracker.weights.shape == (20, 2)

        # Check tracker-specific outputs
        assert len(tracker.archetype_history) > 0
        assert len(tracker.boundary_proximity_history) > 0
        assert len(tracker.is_outside_history) > 0

        # Check history shape consistency
        assert tracker.archetype_history[0].shape == (2, 3)
        assert len(tracker.boundary_proximity_history) == len(tracker.is_outside_history)
        assert isinstance(tracker.boundary_proximity_history[0], float)

        # Test loss history
        assert len(tracker.loss_history) > 0
        assert all(isinstance(loss, float) for loss in tracker.loss_history)

    def test_transform(self, small_sample_data):
        """Test transform still works properly in the tracker."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=10)
        tracker.fit(small_sample_data)

        # Test transform with same data
        weights = tracker.transform(small_sample_data)
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Test with new data
        new_data = np.random.rand(5, 3)
        weights = tracker.transform(new_data)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_calculate_weights(self, small_sample_data):
        """Test internal _calculate_weights method."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=10)
        tracker.fit(small_sample_data)

        # Get archetypes in JAX format
        X_jax = jnp.array(small_sample_data)
        archetypes_jax = jnp.array(tracker.archetypes)

        # Calculate weights
        weights = tracker._calculate_weights(X_jax, archetypes_jax)

        # Check shape and constraints
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_check_archetypes_outside(self, small_sample_data):
        """Test method for checking if archetypes are outside convex hull."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=10)
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
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=10)
        tracker.fit(small_sample_data)

        # Get test data in JAX format
        X_jax = jnp.array(small_sample_data)
        archetypes_jax = jnp.array(tracker.archetypes)

        # Constrain archetypes to convex hull
        constrained = tracker._constrain_to_convex_hull_batch(archetypes_jax, X_jax)

        # Verify result shape
        assert constrained.shape == (2, 3)

        # Check if any archetypes are outside after constraint
        is_outside = tracker._check_archetypes_outside(constrained, X_jax)
        assert not np.any(is_outside)  # None should be outside after constraint

    @pytest.mark.skipif(True, reason="Matplotlib visualization test skipped by default")
    def test_visualize_movement(self, small_sample_data):
        """Test visualization method for archetype movement."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=10)
        tracker.fit(small_sample_data)

        # Try to visualize movement with default parameters
        fig = tracker.visualize_movement()

        # If matplotlib is available, this should return a figure
        if fig is not None:
            assert str(type(fig)).find("matplotlib.figure.Figure") != -1

    @pytest.mark.skipif(True, reason="Matplotlib visualization test skipped by default")
    def test_visualize_boundary_proximity(self, small_sample_data):
        """Test visualization method for boundary proximity."""
        tracker = ArchetypeTracker(n_archetypes=2, max_iter=10)
        tracker.fit(small_sample_data)

        # Try to visualize boundary proximity
        fig = tracker.visualize_boundary_proximity()

        # If matplotlib is available, this should return a figure
        if fig is not None:
            assert str(type(fig)).find("matplotlib.figure.Figure") != -1


class TestCommonModelFunctionality:
    """Parameterized tests for common model functionality across all implementations."""

    def test_basic_initialization(self, model_class_and_params):
        """Verify consistent initialization behavior across model variants."""
        model_class, params = model_class_and_params
        model = model_class(**params)

        # Validate common attributes
        assert model.max_iter == 500
        assert model.tol == 1e-6
        assert model.archetypes is None
        assert model.weights is None
