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
        (BiarchetypalAnalysis, {"n_archetypes_first": 2, "n_archetypes_second": 1}),
    ]
)
def model_class_and_params(request):
    """Parametrized fixture for model classes and their initialization parameters."""
    return request.param


class TestArchetypalAnalysis:
    """Test suite for the base ArchetypalAnalysis class."""

    def test_initialization(self):
        """Verify that the model initializes with correct default parameters."""
        model = ArchetypalAnalysis(n_archetypes=3)
        assert model.n_archetypes == 3
        assert model.max_iter == 500
        assert model.tol == 1e-6
        assert model.archetypes is None
        assert model.weights is None
        assert len(model.loss_history) == 0

    @pytest.mark.slow
    def test_fit(self, sample_data):
        """Ensure that the model fits correctly to the provided data."""
        model = ArchetypalAnalysis(n_archetypes=3, max_iter=20)
        model.fit(sample_data)

        # Verify that attributes are properly set after fitting
        assert model.archetypes is not None
        assert model.weights is not None
        assert len(model.loss_history) > 0

        # Confirm expected shapes of output matrices
        assert model.archetypes.shape == (3, 5)
        assert model.weights.shape == (50, 3)

        # Validate that weights satisfy the simplex constraint
        assert np.allclose(np.sum(model.weights, axis=1), 1.0)

    def test_transform(self, small_sample_data):
        """Confirm that transform method returns weights with proper constraints."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        weights = model.transform(small_sample_data)
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_fit_transform(self, small_sample_data):
        """Verify that fit_transform correctly fits the model and returns weights."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        weights = model.fit_transform(small_sample_data)

        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_reconstruct(self, small_sample_data):
        """Ensure that reconstruction produces output with the same shape as input data."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    def test_get_loss_history(self, small_sample_data):
        """Test that get_loss_history returns the loss history after fitting."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        loss_history = model.get_loss_history()
        assert isinstance(loss_history, list)
        assert len(loss_history) > 0
        assert all(isinstance(loss, float) for loss in loss_history)

    @pytest.mark.slow
    def test_fit_with_normalization(self, small_sample_data):
        """Test that the model fits correctly with normalization enabled."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        # Verify that normalization parameters are set
        assert model.X_mean is not None
        assert model.X_std is not None

        # Verify that attributes are properly set after fitting
        assert model.archetypes is not None
        assert model.weights is not None

        # Test reconstruction with normalized data
        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    def test_transform_new_data(self, small_sample_data):
        """Test transforming new data after fitting."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Create new data with same number of features
        new_data = np.random.rand(5, 3)

        weights = model.transform(new_data)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_reconstruct_new_data(self, small_sample_data):
        """Test reconstructing new data after fitting."""
        model = ArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Create new data with same number of features
        new_data = np.random.rand(5, 3)

        reconstructed = model.reconstruct(new_data)
        assert reconstructed.shape == new_data.shape

    def test_error_before_fit(self):
        """Test that appropriate errors are raised when methods are called before fitting."""
        model = ArchetypalAnalysis(n_archetypes=3)

        # Create some test data
        X = np.random.rand(10, 5)

        # Test transform
        with pytest.raises(ValueError, match="Model must be fitted before transform"):
            model.transform(X)

        # Test reconstruct without arguments
        with pytest.raises(ValueError, match="Model must be fitted before reconstruct"):
            model.reconstruct()


class TestImprovedArchetypalAnalysis:
    """Test suite for the ImprovedArchetypalAnalysis class."""

    def test_initialization(self):
        """Verify that the improved model initializes with correct parameters."""
        model = ImprovedArchetypalAnalysis(n_archetypes=3)
        assert model.n_archetypes == 3
        assert model.max_iter == 500
        assert model.archetypes is None
        assert model.weights is None

    @pytest.mark.slow
    def test_fit(self, small_sample_data):
        """Ensure that the improved model fits correctly to the provided data."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Verify that attributes are properly set after fitting
        assert model.archetypes is not None
        assert model.weights is not None

        # Confirm expected shapes of output matrices
        assert model.archetypes.shape == (2, 3)
        assert model.weights.shape == (20, 2)

        # Validate that weights satisfy the simplex constraint
        assert np.allclose(np.sum(model.weights, axis=1), 1.0)

    def test_transform(self, small_sample_data):
        """Test that transform method returns weights with proper constraints."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        weights = model.transform(small_sample_data)
        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_transform_new_data(self, small_sample_data):
        """Test transforming new data after fitting."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Create new data with same number of features
        new_data = np.random.rand(5, 3)

        weights = model.transform(new_data)
        assert weights.shape == (5, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    @pytest.mark.slow
    def test_fit_transform(self, small_sample_data):
        """Test that fit_transform correctly fits the model and returns weights."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        weights = model.fit_transform(small_sample_data)

        assert weights.shape == (20, 2)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

        # Verify model is fitted
        assert model.archetypes is not None
        assert model.weights is not None

    def test_reconstruct(self, small_sample_data):
        """Test that reconstruction produces output with the same shape as input data."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    def test_reconstruct_new_data(self, small_sample_data):
        """Test reconstructing new data after fitting."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data)

        # Create new data with same number of features
        new_data = np.random.rand(5, 3)

        reconstructed = model.reconstruct(new_data)
        assert reconstructed.shape == new_data.shape

    @pytest.mark.slow
    def test_fit_with_normalization(self, small_sample_data):
        """Test that the model fits correctly with normalization enabled."""
        model = ImprovedArchetypalAnalysis(n_archetypes=2, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        # Verify that normalization parameters are set
        assert model.X_mean is not None
        assert model.X_std is not None

        # Verify that attributes are properly set after fitting
        assert model.archetypes is not None
        assert model.weights is not None

        # Test reconstruction with normalized data
        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    def test_error_before_fit(self):
        """Test that appropriate errors are raised when methods are called before fitting."""
        model = ImprovedArchetypalAnalysis(n_archetypes=3)

        # Create some test data
        X = np.random.rand(10, 5)

        # Test transform
        with pytest.raises(ValueError, match="Model must be fitted before transform"):
            model.transform(X)

        # Test reconstruct without arguments
        with pytest.raises(ValueError, match="Model must be fitted before reconstruct"):
            model.reconstruct()


class TestBiarchetypalAnalysis:
    """Test suite for the BiarchetypalAnalysis class."""

    def test_initialization(self):
        """Verify that the biarchetypal model initializes with correct parameters."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1)
        assert model.n_archetypes == 3  # Total archetypes should be 2+1=3
        assert model.max_iter == 500
        assert model.archetypes is None
        assert model.weights is None

    @pytest.mark.slow
    def test_fit(self, small_sample_data):
        """Ensure that the biarchetypal model fits correctly with default parameters."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, max_iter=10)
        model.fit(small_sample_data)

        # Verify that attributes are properly set after fitting
        assert model.archetypes is not None
        assert model.weights is not None

        # Get positive and negative archetypes
        positive_archetypes, negative_archetypes = model.get_all_archetypes()
        positive_weights, negative_weights = model.get_all_weights()

        # Confirm expected shapes of output matrices
        assert positive_archetypes.shape == (2, 3)  # First set has 2 archetypes
        assert negative_archetypes.shape == (1, 3)  # Second set has 1 archetype
        assert positive_weights.shape == (20, 2)
        assert negative_weights.shape == (20, 1)

    def test_transform(self, small_sample_data):
        """Test that transform method returns weights with proper constraints."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, max_iter=10)
        model.fit(small_sample_data)

        weights_first, weights_second = model.transform(small_sample_data)

        # Check shapes
        assert weights_first.shape == (20, 2)
        assert weights_second.shape == (20, 1)

        # Check simplex constraints
        assert np.allclose(np.sum(weights_first, axis=1), 1.0)
        assert np.allclose(np.sum(weights_second, axis=1), 1.0)

    @pytest.mark.slow
    def test_fit_transform(self, small_sample_data):
        """Test that fit_transform correctly fits the model and returns weights."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, max_iter=10)
        weights_first, weights_second = model.fit_transform(small_sample_data)

        # Check shapes
        assert weights_first.shape == (20, 2)
        assert weights_second.shape == (20, 1)

        # Check simplex constraints
        assert np.allclose(np.sum(weights_first, axis=1), 1.0)
        assert np.allclose(np.sum(weights_second, axis=1), 1.0)

        # Verify model is fitted
        assert model.archetypes_first is not None
        assert model.archetypes_second is not None

    def test_reconstruct(self, small_sample_data):
        """Test that reconstruction produces output with the same shape as input data."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, max_iter=10)
        model.fit(small_sample_data)

        # Test reconstruction with default parameters (using training data)
        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

        # Test reconstruction with new data
        new_data = np.random.rand(5, 3)
        reconstructed_new = model.reconstruct(new_data)
        assert reconstructed_new.shape == new_data.shape

    def test_get_all_archetypes(self, small_sample_data):
        """Test that get_all_archetypes returns correctly shaped arrays."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, max_iter=10)
        model.fit(small_sample_data)

        archetypes_first, archetypes_second = model.get_all_archetypes()

        assert archetypes_first.shape == (2, 3)
        assert archetypes_second.shape == (1, 3)

    def test_get_all_weights(self, small_sample_data):
        """Test that get_all_weights returns correctly shaped arrays."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, max_iter=10)
        model.fit(small_sample_data)

        weights_first, weights_second = model.get_all_weights()

        assert weights_first.shape == (20, 2)
        assert weights_second.shape == (20, 1)

        # Check simplex constraints
        assert np.allclose(np.sum(weights_first, axis=1), 1.0)
        assert np.allclose(np.sum(weights_second, axis=1), 1.0)

    def test_error_before_fit(self):
        """Test that appropriate errors are raised when methods are called before fitting."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1)

        # Create some test data
        X = np.random.rand(10, 3)

        # Test get_all_archetypes
        with pytest.raises(ValueError, match="Model must be fitted before accessing archetypes"):
            model.get_all_archetypes()

        # Test get_all_weights
        with pytest.raises(ValueError, match="Model must be fitted before getting weights"):
            model.get_all_weights()

        # Test reconstruct without arguments
        with pytest.raises(ValueError, match="Model must be fitted before reconstruct"):
            model.reconstruct()

        # Test transform
        with pytest.raises(ValueError, match="Model must be fitted before transform"):
            model.transform(X)

    @pytest.mark.slow
    def test_fit_with_normalization(self, small_sample_data):
        """Test that the model fits correctly with normalization enabled."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        # Verify that normalization parameters are set
        assert model.X_mean is not None
        assert model.X_std is not None

        # Verify that attributes are properly set after fitting
        assert model.archetypes_first is not None
        assert model.archetypes_second is not None
        assert model.weights_first is not None
        assert model.weights_second is not None

        # Test reconstruction with normalized data
        reconstructed = model.reconstruct()
        assert reconstructed.shape == small_sample_data.shape

    @pytest.mark.slow
    def test_mixture_weight(self, small_sample_data):
        """Test that different mixture weights affect reconstruction."""
        # Create two models with different mixture weights
        model1 = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, mixture_weight=0.2, max_iter=10)
        model2 = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, mixture_weight=0.8, max_iter=10)

        # Fit both models with the same data
        model1.fit(small_sample_data)
        model2.fit(small_sample_data)

        # Get reconstructions
        recon1 = model1.reconstruct()
        recon2 = model2.reconstruct()

        # Reconstructions should be different due to different mixture weights
        assert not np.array_equal(recon1, recon2)

    @pytest.mark.slow
    def test_transform_with_normalization(self, small_sample_data):
        """Test transform with normalized data."""
        model = BiarchetypalAnalysis(n_archetypes_first=2, n_archetypes_second=1, max_iter=10)
        model.fit(small_sample_data, normalize=True)

        # Transform the original data
        weights_first, weights_second = model.transform(small_sample_data)

        # Check shapes and simplex constraints
        assert weights_first.shape == (20, 2)
        assert weights_second.shape == (20, 1)
        assert np.allclose(np.sum(weights_first, axis=1), 1.0)
        assert np.allclose(np.sum(weights_second, axis=1), 1.0)

        # Create new data and transform
        new_data = np.random.rand(5, 3)
        weights_first_new, weights_second_new = model.transform(new_data)

        # Check shapes and simplex constraints for new data
        assert weights_first_new.shape == (5, 2)
        assert weights_second_new.shape == (5, 1)
        assert np.allclose(np.sum(weights_first_new, axis=1), 1.0)
        assert np.allclose(np.sum(weights_second_new, axis=1), 1.0)


class TestCommonModelFunctionality:
    """共通のモデル機能をテストするためのパラメータ化されたテスト。"""

    def test_basic_initialization(self, model_class_and_params):
        """すべてのモデルクラスの基本的な初期化をテスト。"""
        model_class, params = model_class_and_params
        model = model_class(**params)

        # すべてのモデルに共通の属性をチェック
        assert model.max_iter == 500
        assert model.tol == 1e-6
        assert model.archetypes is None
        assert model.weights is None
