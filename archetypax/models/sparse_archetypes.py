"""Sparse Archetypal Analysis model using JAX."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from archetypax.logger import get_logger, get_message
from archetypax.models.archetypes import ImprovedArchetypalAnalysis


class SparseArchetypalAnalysis(ImprovedArchetypalAnalysis):
    """Archetypal Analysis with sparsity constraints on archetypes.

    This implementation extends the ImprovedArchetypalAnalysis by adding
    sparsity constraints to the archetypes, which can improve interpretability
    especially in high-dimensional data.
    """

    def __init__(
        self,
        n_archetypes: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_seed: int = 42,
        learning_rate: float = 0.001,
        lambda_reg: float = 0.01,
        lambda_sparsity: float = 0.1,
        sparsity_method: str = "l1",
        normalize: bool = False,
        projection_method: str = "cbap",
        projection_alpha: float = 0.1,
        archetype_init_method: str = "directional",
        **kwargs,
    ):
        """Initialize the Sparse Archetypal Analysis model.

        Args:
            n_archetypes: Number of archetypes to extract
            max_iter: Maximum number of iterations for optimization
            tol: Convergence tolerance
            random_seed: Random seed for reproducibility
            learning_rate: Learning rate for the optimizer
            lambda_reg: Regularization strength for weights
            lambda_sparsity: Regularization strength for archetype sparsity
            sparsity_method: Method for enforcing sparsity ("l1", "l0_approx", or "feature_selection")
            normalize: Whether to normalize data before fitting
            projection_method: Method for projecting archetypes ("cbap", "convex_hull", or "knn")
            projection_alpha: Strength of projection (0-1)
            archetype_init_method: Method for initializing archetypes
                ("directional", "qhull", "kmeans_pp")
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            learning_rate=learning_rate,
            lambda_reg=lambda_reg,
            normalize=normalize,
            projection_method=projection_method,
            projection_alpha=projection_alpha,
            archetype_init_method=archetype_init_method,
            **kwargs,
        )

        self.lambda_sparsity = lambda_sparsity
        self.sparsity_method = sparsity_method

        # Initialize class-specific logger with updated class name
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            get_message(
                "init",
                "model_init",
                model_name=self.__class__.__name__,
                n_archetypes=n_archetypes,
                sparsity_method=sparsity_method,
                lambda_sparsity=lambda_sparsity,
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, archetypes, weights, X):
        """JIT-compiled loss function with sparsity constraint on archetypes."""
        archetypes_f32 = archetypes.astype(jnp.float32)
        weights_f32 = weights.astype(jnp.float32)
        X_f32 = X.astype(jnp.float32)

        X_reconstructed = jnp.matmul(weights_f32, archetypes_f32)
        reconstruction_loss = jnp.mean(jnp.sum((X_f32 - X_reconstructed) ** 2, axis=1))

        # Calculate entropy for weights (higher values for uniform weights, lower for sparse weights)
        weight_entropy = -jnp.sum(weights_f32 * jnp.log(weights_f32 + 1e-10), axis=1)
        weight_entropy_reg = jnp.mean(weight_entropy)

        # Add incentive for archetypes to stay near convex hull boundary
        boundary_incentive = self._calculate_boundary_proximity(archetypes_f32, X_f32)

        # Calculate sparsity penalty based on the selected method
        if self.sparsity_method == "l1":
            # L1 regularization to promote sparsity in archetypes
            sparsity_penalty = jnp.mean(jnp.sum(jnp.abs(archetypes_f32), axis=1))
        elif self.sparsity_method == "l0_approx":
            # Approximation of L0 norm using continuous function
            # This provides a smoother approximation of counting non-zero elements
            epsilon = 1e-6
            sparsity_penalty = jnp.mean(
                jnp.sum(1 - jnp.exp(-(archetypes_f32**2) / epsilon), axis=1)
            )
        elif self.sparsity_method == "feature_selection":
            # Encourages each archetype to focus on a subset of features
            # by penalizing uniform distribution across features
            archetype_entropy = -jnp.sum(archetypes_f32 * jnp.log(archetypes_f32 + 1e-10), axis=1)
            sparsity_penalty = jnp.mean(archetype_entropy)
        else:
            # Default to L1 if method is not recognized
            sparsity_penalty = jnp.mean(jnp.sum(jnp.abs(archetypes_f32), axis=1))

        # Combined loss with reconstruction, regularizations, and boundary incentive
        total_loss = (
            reconstruction_loss
            + self.lambda_reg * weight_entropy_reg
            + self.lambda_sparsity * sparsity_penalty
            - 0.001 * boundary_incentive
        )

        return total_loss.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def update_archetypes(self, archetypes, weights, X) -> jnp.ndarray:
        """Update archetypes with additional sparsity promotion step."""
        # First perform the standard archetype update
        archetypes_updated = super().update_archetypes(archetypes, weights, X)

        # Apply additional sparsity promotion based on selected method
        if self.sparsity_method == "feature_selection":
            # For feature selection, apply soft thresholding to promote feature selectivity
            # This step keeps the largest values in each archetype and shrinks smaller values

            # Calculate thresholds for each archetype (adaptive thresholding)
            thresholds = jnp.percentile(archetypes_updated, 50, axis=1, keepdims=True)

            # Soft thresholding: shrink values below threshold
            shrinkage_factor = 0.7  # Controls how aggressively to shrink small values
            mask = archetypes_updated < thresholds
            archetypes_updated = jnp.where(
                mask, archetypes_updated * shrinkage_factor, archetypes_updated
            )

            # Re-normalize to maintain simplex constraints
            row_sums = jnp.sum(archetypes_updated, axis=1, keepdims=True)
            archetypes_updated = archetypes_updated / jnp.maximum(1e-10, row_sums)

        # Ensure archetypes are within the convex hull
        centroid = jnp.mean(X, axis=0)

        # Process each archetype to ensure it's inside the convex hull
        def _constrain_to_convex_hull(archetype):
            # Direction from centroid to archetype
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)

            # Handle near-zero norm case
            normalized_direction = jnp.where(
                direction_norm > 1e-10, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Project all points onto this direction
            projections = jnp.dot(X - centroid, normalized_direction)

            # Find max projection (extreme point in this direction)
            max_projection = jnp.max(projections)

            # Calculate archetype projection along this direction
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

            # Scale factor to bring the archetype inside the convex hull if it's outside
            # Apply a small margin (0.99) to ensure it's strictly inside
            scale_factor = jnp.where(
                archetype_projection > max_projection,
                0.99 * max_projection / (archetype_projection + 1e-10),
                1.0,
            )

            # Apply the scaling to the direction vector
            constrained_archetype = centroid + scale_factor * (archetype - centroid)
            return constrained_archetype

        # Apply constraint to each archetype
        constrained_archetypes = jax.vmap(_constrain_to_convex_hull)(archetypes_updated)

        return jnp.asarray(constrained_archetypes)

    def get_archetype_sparsity(self) -> np.ndarray:
        """Calculate sparsity of each archetype.

        Returns:
            Array containing the sparsity score for each archetype.
            Higher values indicate more sparse archetypes.
        """
        if not hasattr(self, "archetypes_") or self.archetypes_ is None:
            raise ValueError("Model has not been fitted yet.")

        archetypes = self.archetypes_
        n_archetypes, n_features = archetypes.shape
        sparsity_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Calculate the Gini coefficient as a measure of sparsity
            # (alternative to directly counting zeroes which isn't differentiable)
            sorted_values = np.sort(np.abs(archetypes[i]))
            cumsum = np.cumsum(sorted_values)
            gini = 1 - 2 * np.sum(cumsum) / (n_features * np.sum(sorted_values))
            sparsity_scores[i] = gini

        return sparsity_scores

    def fit(
        self,
        X: np.ndarray,
        normalize: bool = False,
        **kwargs,
    ) -> "SparseArchetypalAnalysis":
        """Fit the Sparse Archetypal Analysis model to the data.

        Args:
            X: Input data matrix of shape (n_samples, n_features)
            normalize: Whether to normalize data before fitting
            **kwargs: Additional keyword arguments

        Returns:
            Fitted model instance
        """
        self.logger.info(
            get_message(
                "training",
                "model_training_start",
                sparsity_method=self.sparsity_method,
                lambda_sparsity=self.lambda_sparsity,
            )
        )

        return super().fit(X, normalize, **kwargs)
