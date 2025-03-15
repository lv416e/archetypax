"""Unit tests for archetypax models."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from archetypax.models.archetypes import ImprovedArchetypalAnalysis
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
