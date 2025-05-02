"""Sparse Archetypal Analysis using JAX.

This module extends ImprovedArchetypalAnalysis with sparsity-promoting regularization,
enabling more interpretable and focused archetype discovery.
By encouraging archetypes to utilize only essential features,
this approach addresses a key limitation of standard archetypal analysis:
the tendency to produce dense, difficult-to-interpret archetypes in high-dimensional spaces.

Core Features:
- Creates more interpretable archetypes focusing on truly relevant features
- Performs automatic feature selection within the archetypal framework
- Improves robustness to noise and irrelevant dimensions
- Prevents overfitting to spurious correlations
- Generates computationally efficient sparse representations

Multiple sparsity techniques are supported:
- "l1": L1 regularization (fastest, robust, tends to zero out features)
- "l0_approx": Approximated L0 regularization (more aggressive sparsity)
- "feature_selection": Entropy-based selection (focuses on key features)

Example usage:
    ```python
    from archetypax.models import SparseArchetypalAnalysis

    # Initialize model with sparsity parameters
    model = SparseArchetypalAnalysis(
        n_archetypes=5,
        lambda_sparsity=0.1,
        sparsity_method="l1",
        normalize=True
    )

    # Fit model and transform data
    weights = model.fit_transform(X)

    # Evaluate archetype sparsity
    sparsity_scores = model.get_archetype_sparsity()
    ```
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import ConvexHull

from archetypax.logger import get_logger, get_message
from archetypax.models.archetypes import ImprovedArchetypalAnalysis


class SparseArchetypalAnalysis(ImprovedArchetypalAnalysis):
    """Archetypal Analysis with sparsity constraints for enhanced interpretability.

    This implementation addresses a fundamental challenge in standard archetypal analysis:
    dense archetypes that utilize many features are often difficult to interpret,
    particularly in high-dimensional datasets where most features may be irrelevant to specific patterns.

    By incorporating sparsity constraints, this approach offers several key advantages:
    1. More interpretable archetypes that focus on truly relevant features
    2. Automatic feature selection within the archetypal framework
    3. Improved robustness to noise and irrelevant dimensions
    4. Better generalization by preventing overfitting to spurious correlations
    5. Computationally efficient representations, especially for high-dimensional data

    Multiple sparsity-promoting methods are supported,
    enabling adaptation to different data characteristics and interpretability requirements.
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
            n_archetypes: Number of archetypes to discover
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance for early stopping
            random_seed: Random seed for reproducibility
            learning_rate: Gradient descent step size
            lambda_reg: Weight regularization strength
            lambda_sparsity: Archetype sparsity strength
            sparsity_method: Technique for promoting archetype sparsity ("l1", "l0_approx", "feature_selection")
            normalize: Whether to normalize features
            projection_method: Method for projecting archetypes to convex hull
            projection_alpha: Strength of boundary projection
            archetype_init_method: Initialization strategy for archetypes
            min_volume_factor: Minimum volume requirement for archetype simplex
            **kwargs: Additional parameters
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

        if isinstance(kwargs.get("logger_level"), str) and kwargs.get("logger_level") is not None:
            logger_level = kwargs["logger_level"].upper()
        elif isinstance(kwargs.get("logger_level"), int) and kwargs.get("logger_level") is not None:
            logger_level = {
                0: "DEBUG",
                1: "INFO",
                2: "WARNING",
                3: "ERROR",
                4: "CRITICAL",
            }[kwargs["logger_level"]]
        elif "logger_level" not in kwargs and "verbose_level" in kwargs and kwargs["verbose_level"] is not None:
            logger_level = {
                4: "DEBUG",
                3: "INFO",
                2: "WARNING",
                1: "ERROR",
                0: "CRITICAL",
            }[kwargs["verbose_level"]]
        else:
            logger_level = "ERROR"

        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", level=logger_level)
        self.logger.info(
            get_message(
                "init",
                "model_init",
                model_name=self.__class__.__name__,
                n_archetypes=n_archetypes,
                sparsity_method=sparsity_method,
                lambda_sparsity=lambda_sparsity,
                min_volume_factor=min_volume_factor,
                learning_rate=learning_rate,
                lambda_reg=lambda_reg,
                normalize=normalize,
                projection_method=projection_method,
                projection_alpha=projection_alpha,
                archetype_init_method=archetype_init_method,
                max_iter=max_iter,
                tol=tol,
                random_seed=random_seed,
            )
        )

        self.rng_key = jax.random.key(random_seed)
        self.lambda_sparsity = lambda_sparsity
        self.sparsity_method = sparsity_method
        self.min_volume_factor = min_volume_factor
        self.early_stopping_patience = kwargs.get("early_stopping_patience", 100)

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, archetypes: jnp.ndarray, weights: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Calculate the composite loss function incorporating sparsity constraints.

        The balance between multiple terms is critical:
        - Reconstruction accuracy: Ensuring archetypes accurately represent the data
        - Archetype sparsity: Promoting focused archetypes that use fewer features
        - Weight interpretability: Encouraging sparse, distinctive weight patterns
        - Boundary alignment: Maintaining archetypes at meaningful data extremes
        - Archetype diversity: Preventing redundant or overlapping archetypes

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)
            weights: Weight matrix (n_samples, n_archetypes)
            X: Data matrix (n_samples, n_features)

        Returns:
            Combined loss incorporating reconstruction error and multiple regularization terms
        """
        archetypes_f32 = archetypes.astype(jnp.float32)
        weights_f32 = weights.astype(jnp.float32)
        X_f32 = X.astype(jnp.float32)

        X_reconstructed = jnp.matmul(weights_f32, archetypes_f32)
        reconstruction_loss = jnp.mean(jnp.sum((X_f32 - X_reconstructed) ** 2, axis=1))

        weight_entropy = -jnp.sum(weights_f32 * jnp.log(weights_f32 + 1e-10), axis=1)
        weight_entropy_reg = jnp.mean(weight_entropy)

        boundary_incentive = self._calculate_boundary_proximity(archetypes_f32, X_f32)

        # Dictionary of different sparsity methods with their implementations
        sparsity_methods = {
            "l1": lambda arc: jnp.mean(jnp.sum(jnp.abs(arc), axis=1)),
            "l0_approx": lambda arc: jnp.mean(jnp.sum(1 - jnp.exp(-(arc**2) / 1e-6), axis=1)),
            "feature_selection": lambda arc: jnp.mean(-jnp.sum(arc * jnp.log(arc + 1e-10), axis=1)),
        }
        sparsity_method_fn = sparsity_methods.get(self.sparsity_method, sparsity_methods["l1"])
        sparsity_penalty = sparsity_method_fn(archetypes_f32)

        n_archetypes = archetypes_f32.shape[0]
        archetype_diversity_penalty = 0.0

        # Calculate diversity penalty only when multiple archetypes exist
        if n_archetypes > 1:
            norms = jnp.sqrt(jnp.sum(archetypes_f32**2, axis=1, keepdims=True))
            normalized_archetypes = archetypes_f32 / jnp.maximum(norms, 1e-10)
            similarity_matrix = jnp.dot(normalized_archetypes, normalized_archetypes.T)

            # Mask to exclude self-similarities
            mask = jnp.ones((n_archetypes, n_archetypes)) - jnp.eye(n_archetypes)
            masked_similarities = similarity_matrix * mask

            archetype_diversity_penalty = jax.device_get(jnp.mean(jnp.maximum(masked_similarities, 0)))

        # Weights chosen empirically to balance competing objectives
        incentive_weight = 0.001
        diversity_weight = 0.1
        total_loss = (
            reconstruction_loss
            + self.lambda_reg * weight_entropy_reg
            + self.lambda_sparsity * sparsity_penalty
            - incentive_weight * boundary_incentive
            + diversity_weight * archetype_diversity_penalty
        )

        return jnp.asarray(total_loss).astype(jnp.float32)

    def _calculate_simplex_volume(self, archetypes: jnp.ndarray) -> float:
        """Calculate the volume of the simplex formed by archetypes.

        This geometric measure detects degenerate solutions where archetypes collapse to similar positions.
        Such collapses reduce model expressiveness, as multiple archetypes would represent the same pattern.

        Handles two challenging cases:
        1. High-dimensional spaces where direct volume calculation is unstable
        2. Situations with fewer archetypes than dimensions (true volume would be zero)

        In both cases, pairwise distances provide a reliable proxy for archetype diversity.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)

        Returns:
            Volume or proxy measure - higher values indicate better-distributed archetypes
        """
        n_archetypes, n_features = archetypes.shape

        # If fewer archetypes than dimensions + 1, use pairwise distance proxy
        if n_archetypes <= n_features:
            pairwise_distances = np.zeros((n_archetypes, n_archetypes))
            for i in range(n_archetypes):
                for j in range(i + 1, n_archetypes):
                    dist = np.linalg.norm(archetypes[i] - archetypes[j])
                    pairwise_distances[i, j] = pairwise_distances[j, i] = dist

            volume_proxy = np.sum(pairwise_distances) / (n_archetypes * (n_archetypes - 1) / 2)
            return float(volume_proxy)

        else:
            try:
                hull = ConvexHull(archetypes)
                return float(hull.volume)

            except Exception:
                # Fallback to pairwise distance proxy if ConvexHull fails
                pairwise_distances = np.zeros((n_archetypes, n_archetypes))
                for i in range(n_archetypes):
                    for j in range(i + 1, n_archetypes):
                        dist = np.linalg.norm(archetypes[i] - archetypes[j])
                        pairwise_distances[i, j] = pairwise_distances[j, i] = dist
                volume_proxy = np.sum(pairwise_distances) / (n_archetypes * (n_archetypes - 1) / 2)
                return float(volume_proxy)

    @partial(jax.jit, static_argnums=(0,))
    def update_archetypes(self, archetypes: jnp.ndarray, weights: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Update archetypes with sparsity promotion and degeneracy prevention.

        This enhanced update method extends the standard approach
        with critical improvements for sparse archetypal analysis:

        1. Sparsity promotion through the selected method (l1, l0, feature selection)
        2. Variance-aware noise injection to prevent dimensional collapse
        3. Constraint enforcement to maintain valid convex combinations
        4. Boundary projection to ensure archetypes remain at meaningful extremes

        Args:
            archetypes: Current archetype matrix (n_archetypes, n_features)
            weights: Weight matrix (n_samples, n_archetypes)
            X: Data matrix (n_samples, n_features)

        Returns:
            Updated archetypes incorporating sparsity and diversity constraints
        """
        # First, perform the standard archetype update.
        archetypes_updated = super().update_archetypes(archetypes, weights, X)

        sparsity_methods = {"feature_selection": lambda arc: self._apply_feature_selection(arc)}
        sparsity_method_fn = sparsity_methods.get(self.sparsity_method)
        if sparsity_method_fn:
            archetypes_updated = sparsity_method_fn(archetypes_updated)

        # Calculate feature-wise variance to identify potential degeneracy.
        # Low variance in features across archetypes suggests dimensional collapse.
        archetype_variance = jnp.var(archetypes_updated, axis=0, keepdims=True)

        # Introduce noise inversely scaled with variance to prevent degeneracy.
        # Using 0.01 as noise_scale balances stability and sufficient perturbation.
        noise_scale = 0.01

        # Use jax.random instead of jnp.random for better compatibility with JIT
        _, noise_key = jax.random.split(self.rng_key)
        noise = jax.random.uniform(noise_key, shape=archetypes_updated.shape) - 0.5  # Zero-centered noise

        # Scale noise inversely with variance - more noise in low-variance dimensions.
        variance_scaling = noise_scale / (jnp.sqrt(archetype_variance) + 1e-8)
        scaled_noise = noise * variance_scaling

        # Apply noise only to non-zero elements to preserve sparsity pattern.
        archetypes_with_noise = archetypes_updated + scaled_noise * (archetypes_updated > 1e-5)
        row_sums = jnp.sum(archetypes_with_noise, axis=1, keepdims=True)
        archetypes_with_noise = archetypes_with_noise / jnp.maximum(1e-10, row_sums)

        centroid = jnp.mean(X, axis=0)

        def _constrain_to_convex_hull(archetype: jnp.ndarray) -> jnp.ndarray:
            # Compute direction from centroid to archetype
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)

            # Handle numerical stability for near-zero norm case
            normalized_direction = jnp.where(
                direction_norm > 1e-10, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Project data points onto this direction to find extreme point
            projections = jnp.dot(X - centroid, normalized_direction)
            max_projection = jnp.max(projections)

            # Calculate the archetype's projection along this direction
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

            # Scale to ensure the archetype stays inside the convex hull with small margin (0.99)
            scale_factor = jnp.where(
                archetype_projection > max_projection,
                0.99 * max_projection / (archetype_projection + 1e-10),
                1.0,
            )

            constrained_archetype = centroid + scale_factor * (archetype - centroid)
            return constrained_archetype

        constrained_archetypes = jax.vmap(_constrain_to_convex_hull)(archetypes_with_noise)

        return jnp.asarray(constrained_archetypes)

    def _apply_feature_selection(self, archetypes_updated: jnp.ndarray) -> jnp.ndarray:
        """Apply feature selection-based sparsity to archetypes.

        This method enhances feature selectivity in each archetype by adaptively identifying
        and emphasizing significant features while suppressing less important ones.

        Particularly valuable when:
        1. Certain key features should dominate each archetype's interpretation
        2. The relative importance of features matters more than strict sparsity
        3. Some baseline activity across all features is expected or desirable

        Uses soft thresholding with adaptive percentile cutoffs for more nuanced approach.

        Args:
            archetypes_updated: Current archetype matrix to be sparsified

        Returns:
            Archetype matrix with enhanced feature selectivity
        """
        # Use median as threshold, shrinking values below this by 30%
        thresholds = jnp.percentile(archetypes_updated, 50, axis=1, keepdims=True)
        shrinkage_factor = 0.7
        mask = archetypes_updated < thresholds

        archetypes_updated = jnp.where(mask, archetypes_updated * shrinkage_factor, archetypes_updated)

        # Re-normalize to maintain convex combination constraints
        row_sums = jnp.sum(archetypes_updated, axis=1, keepdims=True)
        archetypes_updated = archetypes_updated / jnp.maximum(1e-10, row_sums)

        return archetypes_updated

    def diversify_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Prevent degenerate solutions by ensuring sufficient archetype diversity.

        Addresses a challenge where multiple archetypes can collapse to similar positions,
        especially when sparsity is enforced, reducing model expressiveness.

        This method:
        1. Detects potential degeneracy through simplex volume measurement
        2. Systematically pushes archetypes away from each other when needed
        3. Ensures archetypes remain valid (within the convex hull)
        4. Verifies improvement through before/after volume comparison

        Performed outside JAX-compiled steps due to non-differentiable operations.

        Args:
            archetypes: Current archetypes matrix to diversify
            X: Data matrix defining the convex hull boundary

        Returns:
            Diversified archetypes with improved distribution and volume
        """
        n_archetypes = archetypes.shape[0]
        initial_volume = self._calculate_simplex_volume(archetypes)

        # Only attempt diversification if volume is too small
        if initial_volume < self.min_volume_factor:
            self.logger.info(
                f"Detected a potentially degenerate archetype configuration. "
                f"Volume proxy: {initial_volume:.6f}. Attempting to diversify."
            )

            centroid = np.mean(X, axis=0)
            for i in range(n_archetypes):
                # Calculate direction away from the average of other archetypes
                other_archetypes = np.delete(archetypes, i, axis=0)
                other_centroid = np.mean(other_archetypes, axis=0)

                direction = archetypes[i] - other_centroid
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 1e-10:
                    normalized_direction = direction / direction_norm

                    # Find maximum possible projection while staying in convex hull
                    projections = np.dot(X - centroid, normalized_direction)
                    max_projection = np.max(projections)
                    current_projection = np.dot(archetypes[i] - centroid, normalized_direction)

                    # Move halfway toward the extreme point for balanced diversification
                    blend_factor = 0.5
                    if current_projection < max_projection:
                        target_projection = current_projection + blend_factor * (max_projection - current_projection)
                        archetypes[i] = centroid + normalized_direction * target_projection

            # Re-normalize to maintain convex combination constraints
            row_sums = np.sum(archetypes, axis=1, keepdims=True)
            archetypes = archetypes / np.maximum(1e-10, row_sums)

            new_volume = self._calculate_simplex_volume(archetypes)
            self.logger.info(
                f"After diversification, volume proxy changed from {initial_volume:.6f} to {new_volume:.6f}"
            )

        return archetypes

    def get_archetype_sparsity(self) -> np.ndarray:
        """Calculate the effective sparsity of each archetype.

        Uses the Gini coefficient rather than simply counting zeros, providing a
        more nuanced sparsity metric that works with both hard and soft thresholding.

        The Gini coefficient measures inequality among values,
        with higher values indicating greater sparsity (few large values, many small values).

        Returns:
            Array of sparsity scores (higher values = more focused archetypes)
        """
        if not hasattr(self, "archetypes") or self.archetypes is None:
            raise ValueError("The model has not yet been fitted.")

        archetypes = self.archetypes
        n_archetypes, n_features = archetypes.shape
        sparsity_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Calculate Gini coefficient as differentiable sparsity measure
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
        """Fit the model to discover sparse, interpretable archetypes.

        Orchestrates the complete sparse archetypal analysis process:
        1. Leverages the parent class for core optimization
        2. Applies the selected sparsity-promoting method during optimization
        3. Performs post-processing to ensure archetype diversity
        4. Validates sparsity and geometric properties of the solution

        Args:
            X: Data matrix (n_samples, n_features)
            normalize: Whether to normalize features before fitting
            **kwargs: Additional parameters for customizing the fitting process

        Returns:
            Self - fitted model instance with discovered sparse archetypes
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

        model = super().fit(X_np, normalize, **kwargs)
        if not hasattr(model, "archetypes") or model.archetypes is None:
            raise ValueError("Archetypes are not None")

        # Apply post-processing to ensure archetype diversity
        X_jax = jnp.asarray(X_np if not normalize else (X_np - model.X_mean) / model.X_std)
        archetypes = jnp.asarray(model.archetypes)
        archetypes = self.diversify_archetypes(archetypes=archetypes, X=X_jax)
        model.archetypes = np.asarray(archetypes)

        return model

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Fit the model and transform the input data in one operation.

        Combines model fitting and data transformation for:
        1. Computational efficiency
        2. Simplified workflow integration with scikit-learn compatible frameworks

        Args:
            X: Data matrix to fit and transform (n_samples, n_features)
            y: Ignored. Present for scikit-learn API compatibility
            normalize: Whether to normalize features before fitting
            **kwargs: Additional parameters passed to fit()

        Returns:
            Weight matrix representing samples as combinations of sparse archetypes
        """
        X_np = X.values if hasattr(X, "values") else X
        model = self.fit(X_np, normalize, **kwargs)
        return model.transform(X_np)
