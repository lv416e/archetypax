"""Sparse Archetypal Analysis model utilizing JAX."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import ConvexHull

from archetypax.logger import get_logger, get_message
from archetypax.models.archetypes import ImprovedArchetypalAnalysis


class SparseArchetypalAnalysis(ImprovedArchetypalAnalysis):
    """Archetypal Analysis incorporating sparsity constraints on archetypes.

    This implementation enhances the ImprovedArchetypalAnalysis by introducing
    sparsity constraints to the archetypes, thereby improving interpretability,
    particularly in high-dimensional datasets.
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
        min_volume_factor: float = 0.001,
        **kwargs,
    ):
        """Initialize the Sparse Archetypal Analysis model.

        Args:
            n_archetypes: Number of archetypes to extract.
            max_iter: Maximum number of iterations for optimization.
            tol: Convergence tolerance.
            random_seed: Random seed for reproducibility.
            learning_rate: Learning rate for the optimizer.
            lambda_reg: Regularization strength for weights.
            lambda_sparsity: Regularization strength for archetype sparsity.
            sparsity_method: Method for enforcing sparsity ("l1", "l0_approx", or "feature_selection").
            normalize: Whether to normalize data prior to fitting.
            projection_method: Method for projecting archetypes ("cbap", "convex_hull", or "knn").
            projection_alpha: Strength of projection (0-1).
            archetype_init_method: Method for initializing archetypes
                ("directional", "qhull", "kmeans_pp").
            min_volume_factor: Minimum volume factor to prevent degeneracy (0-1).
            **kwargs: Additional keyword arguments.
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

        # Initialize a class-specific logger with the updated class name.
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            get_message(
                "init",
                "model_init",
                model_name=self.__class__.__name__,
                n_archetypes=n_archetypes,
                sparsity_method=sparsity_method,
                lambda_sparsity=lambda_sparsity,
                min_volume_factor=min_volume_factor,
            )
        )

        self.rng_key = jax.random.key(random_seed)
        self.lambda_sparsity = lambda_sparsity
        self.sparsity_method = sparsity_method
        self.min_volume_factor = min_volume_factor  # Parameter controlling the minimum volume of the convex hull.

        self.early_stopping_patience = kwargs.get("early_stopping_patience", 100)
        self.verbose_level = kwargs.get("verbose_level", 1)

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, archetypes, weights, X):
        """JIT-compiled loss function incorporating a sparsity constraint on archetypes."""
        archetypes_f32 = archetypes.astype(jnp.float32)
        weights_f32 = weights.astype(jnp.float32)
        X_f32 = X.astype(jnp.float32)

        X_reconstructed = jnp.matmul(weights_f32, archetypes_f32)
        reconstruction_loss = jnp.mean(jnp.sum((X_f32 - X_reconstructed) ** 2, axis=1))

        # Calculate entropy for weights (higher values indicate uniform weights, lower values indicate sparse weights).
        weight_entropy = -jnp.sum(weights_f32 * jnp.log(weights_f32 + 1e-10), axis=1)
        weight_entropy_reg = jnp.mean(weight_entropy)

        # Introduce an incentive for archetypes to remain near the boundary of the convex hull.
        boundary_incentive = self._calculate_boundary_proximity(archetypes_f32, X_f32)

        # Compute the sparsity penalty based on the selected method using a dictionary dispatch
        sparsity_methods = {
            "l1": lambda arc: jnp.mean(jnp.sum(jnp.abs(arc), axis=1)),
            "l0_approx": lambda arc: jnp.mean(jnp.sum(1 - jnp.exp(-(arc**2) / 1e-6), axis=1)),
            "feature_selection": lambda arc: jnp.mean(-jnp.sum(arc * jnp.log(arc + 1e-10), axis=1)),
        }

        # Get the sparsity method or default to L1
        sparsity_method_fn = sparsity_methods.get(self.sparsity_method, sparsity_methods["l1"])

        # Calculate sparsity penalty
        sparsity_penalty = sparsity_method_fn(archetypes_f32)

        # Calculate the archetype diversity penalty based on pairwise similarity.
        n_archetypes = archetypes_f32.shape[0]
        archetype_diversity_penalty = 0.0

        if n_archetypes > 1:
            # Compute the normalized cosine similarity matrix between archetypes.
            norms = jnp.sqrt(jnp.sum(archetypes_f32**2, axis=1, keepdims=True))
            normalized_archetypes = archetypes_f32 / jnp.maximum(norms, 1e-10)
            similarity_matrix = jnp.dot(normalized_archetypes, normalized_archetypes.T)

            # Exclude diagonal elements (self-similarity is always 1).
            mask = jnp.ones((n_archetypes, n_archetypes)) - jnp.eye(n_archetypes)
            masked_similarities = similarity_matrix * mask

            # Retrieve the maximum similarity (higher values indicate a problem).
            archetype_diversity_penalty = jax.device_get(jnp.mean(jnp.maximum(masked_similarities, 0)))

        # Add the archetype diversity penalty to the total loss (higher similarity = lower diversity penalty).
        diversity_weight = 0.1

        # Combined loss incorporating reconstruction, regularizations, boundary incentive, and diversity.
        total_loss = (
            reconstruction_loss
            + self.lambda_reg * weight_entropy_reg
            + self.lambda_sparsity * sparsity_penalty
            - 0.001 * boundary_incentive
            + diversity_weight * archetype_diversity_penalty
        )

        return total_loss.astype(jnp.float32)

    def _calculate_simplex_volume(self, archetypes):
        """Calculate the volume of the simplex formed by the archetypes.

        This is utilized to ensure that the archetypes do not collapse into a degenerate subspace.

        Args:
            archetypes: Array of shape (n_archetypes, n_features).

        Returns:
            Approximate volume of the simplex.
        """
        n_archetypes, n_features = archetypes.shape

        # If there are fewer archetypes than dimensions + 1, the volume is technically zero.
        # Instead, we will compute a proxy metric based on pairwise distances.
        if n_archetypes <= n_features:
            # Calculate pairwise distances between archetypes.
            pairwise_distances = np.zeros((n_archetypes, n_archetypes))
            for i in range(n_archetypes):
                for j in range(i + 1, n_archetypes):
                    dist = np.linalg.norm(archetypes[i] - archetypes[j])
                    pairwise_distances[i, j] = pairwise_distances[j, i] = dist

            # Use the product of pairwise distances as a proxy for volume.
            # Higher values indicate that archetypes are more spread out.
            volume_proxy = np.sum(pairwise_distances) / (n_archetypes * (n_archetypes - 1) / 2)
            return volume_proxy
        else:
            try:
                # When there are enough points, attempt to compute the actual convex hull volume.
                hull = ConvexHull(archetypes)
                return hull.volume
            except Exception:
                # Fallback to the pairwise distance approach if the convex hull computation fails.
                pairwise_distances = np.zeros((n_archetypes, n_archetypes))
                for i in range(n_archetypes):
                    for j in range(i + 1, n_archetypes):
                        dist = np.linalg.norm(archetypes[i] - archetypes[j])
                        pairwise_distances[i, j] = pairwise_distances[j, i] = dist

                volume_proxy = np.sum(pairwise_distances) / (n_archetypes * (n_archetypes - 1) / 2)
                return volume_proxy

    @partial(jax.jit, static_argnums=(0,))
    def update_archetypes(self, archetypes, weights, X) -> jnp.ndarray:
        """Update archetypes with an additional step to promote sparsity."""
        # First, perform the standard archetype update.
        archetypes_updated = super().update_archetypes(archetypes, weights, X)

        # Dictionary for sparsity methods
        sparsity_methods = {"feature_selection": lambda arc: self._apply_feature_selection(arc)}

        # Apply sparsity method if available in the dictionary
        sparsity_method_fn = sparsity_methods.get(self.sparsity_method)
        if sparsity_method_fn:
            archetypes_updated = sparsity_method_fn(archetypes_updated)

        # Calculate feature-wise variance of archetypes to identify potential degeneracy.
        # If variance is too low in some features across archetypes, it suggests potential degeneracy.
        archetype_variance = jnp.var(archetypes_updated, axis=0, keepdims=True)

        # Introduce small noise in the direction of low variance to prevent degeneracy.
        # Scale noise inversely with the variance to target low-variance dimensions.
        noise_scale = 0.01

        # Use jax.random instead of jnp.random
        _, noise_key = jax.random.split(self.rng_key)  # Fixed seed for deterministic noise
        noise = jax.random.uniform(noise_key, shape=archetypes_updated.shape) - 0.5  # Zero-centered noise

        # Scale noise inversely proportional to variance (more noise where variance is low).
        # Add a small epsilon to avoid division by zero.
        variance_scaling = noise_scale / (jnp.sqrt(archetype_variance) + 1e-8)
        scaled_noise = noise * variance_scaling

        # Apply noise selectively to avoid disrupting the sparsity pattern.
        # Only add noise where the archetype elements are already non-zero.
        archetypes_with_noise = archetypes_updated + scaled_noise * (archetypes_updated > 1e-5)

        # Re-normalize to maintain simplex constraints.
        row_sums = jnp.sum(archetypes_with_noise, axis=1, keepdims=True)
        archetypes_with_noise = archetypes_with_noise / jnp.maximum(1e-10, row_sums)

        # Ensure archetypes remain within the convex hull.
        centroid = jnp.mean(X, axis=0)

        # Process each archetype to ensure it resides within the convex hull.
        def _constrain_to_convex_hull(archetype):
            # Direction from centroid to archetype.
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)

            # Handle near-zero norm case.
            normalized_direction = jnp.where(
                direction_norm > 1e-10, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Project all points onto this direction.
            projections = jnp.dot(X - centroid, normalized_direction)

            # Identify the maximum projection (extreme point in this direction).
            max_projection = jnp.max(projections)

            # Calculate the archetype projection along this direction.
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

            # Scale factor to bring the archetype inside the convex hull if it is outside.
            # Apply a small margin (0.99) to ensure it is strictly inside.
            scale_factor = jnp.where(
                archetype_projection > max_projection,
                0.99 * max_projection / (archetype_projection + 1e-10),
                1.0,
            )

            # Apply the scaling to the direction vector.
            constrained_archetype = centroid + scale_factor * (archetype - centroid)
            return constrained_archetype

        # Apply the constraint to each archetype.
        constrained_archetypes = jax.vmap(_constrain_to_convex_hull)(archetypes_with_noise)

        return jnp.asarray(constrained_archetypes)

    def _apply_feature_selection(self, archetypes_updated):
        """Apply feature selection sparsity method."""
        # For feature selection, apply soft thresholding to enhance feature selectivity.
        # This step retains the largest values in each archetype while shrinking smaller values.

        # Calculate thresholds for each archetype (adaptive thresholding).
        thresholds = jnp.percentile(archetypes_updated, 50, axis=1, keepdims=True)

        # Soft thresholding: shrink values below the threshold.
        shrinkage_factor = 0.7  # Controls the aggressiveness of shrinking small values.
        mask = archetypes_updated < thresholds
        archetypes_updated = jnp.where(mask, archetypes_updated * shrinkage_factor, archetypes_updated)

        # Re-normalize to maintain simplex constraints.
        row_sums = jnp.sum(archetypes_updated, axis=1, keepdims=True)
        archetypes_updated = archetypes_updated / jnp.maximum(1e-10, row_sums)

        return archetypes_updated

    def diversify_archetypes(self, archetypes, X):
        """Non-differentiable post-processing step to ensure archetype diversity.

        This method guarantees that the archetypes form a non-degenerate simplex with sufficient volume.
        It is invoked outside the JAX-compiled update steps, as it employs non-differentiable operations.

        Args:
            archetypes: Current archetypes array.
            X: Data matrix.

        Returns:
            Diversified archetypes array.
        """
        archetypes_np = np.array(archetypes)
        X_np = np.array(X)
        n_archetypes = archetypes_np.shape[0]

        # Calculate initial volume or proxy.
        initial_volume = self._calculate_simplex_volume(archetypes_np)

        # If the volume is exceedingly small, attempt to increase it.
        if initial_volume < self.min_volume_factor:
            self.logger.info(
                f"Detected a potentially degenerate archetype configuration. "
                f"Volume proxy: {initial_volume:.6f}. Attempting to diversify."
            )

            # Calculate centroid.
            centroid = np.mean(X_np, axis=0)

            # For each archetype, push it away from other archetypes.
            for i in range(n_archetypes):
                # Calculate direction away from the average of other archetypes.
                other_archetypes = np.delete(archetypes_np, i, axis=0)
                other_centroid = np.mean(other_archetypes, axis=0)

                # Direction away from other archetypes.
                direction = archetypes_np[i] - other_centroid
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 1e-10:
                    normalized_direction = direction / direction_norm

                    # Determine how far we can push in this direction while remaining within the convex hull.
                    projections = np.dot(X_np - centroid, normalized_direction)
                    max_projection = np.max(projections)

                    current_projection = np.dot(archetypes_np[i] - centroid, normalized_direction)

                    # Push outward, but remain within the convex hull.
                    # Utilize a blend of the current position and the extreme point.
                    blend_factor = 0.5  # Move halfway toward the extreme point.
                    if current_projection < max_projection:
                        target_projection = current_projection + blend_factor * (max_projection - current_projection)
                        archetypes_np[i] = centroid + normalized_direction * target_projection

            # Re-normalize to maintain simplex constraints.
            row_sums = np.sum(archetypes_np, axis=1, keepdims=True)
            archetypes_np = archetypes_np / np.maximum(1e-10, row_sums)

            # Verify improvement.
            new_volume = self._calculate_simplex_volume(archetypes_np)
            self.logger.info(
                f"After diversification, volume proxy changed from {initial_volume:.6f} to {new_volume:.6f}"
            )

        return jnp.array(archetypes_np)

    def get_archetype_sparsity(self) -> np.ndarray:
        """Calculate the sparsity of each archetype.

        Returns:
            Array containing the sparsity score for each archetype.
            Higher values indicate more sparse archetypes.
        """
        if not hasattr(self, "archetypes") or self.archetypes is None:
            raise ValueError("The model has not yet been fitted.")

        archetypes = self.archetypes
        n_archetypes, n_features = archetypes.shape
        sparsity_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Calculate the Gini coefficient as a measure of sparsity.
            # (An alternative to directly counting zeroes, which isn't differentiable).
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
            X: Input data matrix of shape (n_samples, n_features).
            normalize: Whether to normalize data prior to fitting.
            **kwargs: Additional keyword arguments.

        Returns:
            Fitted model instance.
        """
        self.logger.info(
            get_message(
                "training",
                "model_training_start",
                sparsity_method=self.sparsity_method,
                lambda_sparsity=self.lambda_sparsity,
            )
        )

        X_np = X.values if hasattr(X, "values") else X

        # Utilize the parent class fit method.
        model = super().fit(X_np, normalize, **kwargs)

        # Apply post-processing to ensure archetype diversity.
        # This is executed outside the JAX-compiled optimization.
        if hasattr(model, "archetypes") and model.archetypes is not None:
            X_jax = jnp.array(X_np if not normalize else (X_np - model.X_mean) / model.X_std)
            model.archetypes = self.diversify_archetypes(model.archetypes, X_jax)

        return model

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Fit the Sparse Archetypal Analysis model to the data and return the transformed data.

        Args:
            X: Input data matrix of shape (n_samples, n_features).
            y: Target values (not used).
            normalize: Whether to normalize data prior to fitting.
            **kwargs: Additional keyword arguments.

        Returns:
            Transformed data matrix of shape (n_samples, n_archetypes).
        """
        X_np = X.values if hasattr(X, "values") else X
        model = self.fit(X_np, normalize, **kwargs)
        return model.transform(X_np)
