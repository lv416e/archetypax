"""Sparse Archetypal Analysis: Interpretable pattern discovery with sparsity constraints.

This module extends archetypal analysis with sparsity-promoting regularization,
enabling more interpretable and focused archetype discovery. By encouraging
archetypes to utilize only essential features, this approach addresses a key
limitation of standard archetypal analysis: the tendency to produce dense,
difficult-to-interpret archetypes in high-dimensional spaces.
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
    particularly in high-dimensional datasets where most features may be irrelevant
    to specific patterns.

    By incorporating sparsity constraints, this approach offers several key advantages:

    1. More interpretable archetypes that focus on truly relevant features
    2. Automatic feature selection within the archetypal framework
    3. Improved robustness to noise and irrelevant dimensions
    4. Better generalization by preventing overfitting to spurious correlations
    5. Computationally efficient representations, especially for high-dimensional data

    Multiple sparsity-promoting methods are supported, enabling adaptation to different
    data characteristics and interpretability requirements.
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
            n_archetypes: Number of archetypes to discover - controls the model's
                         expressiveness and granularity of pattern discovery
            max_iter: Maximum optimization iterations - higher values enable better
                     convergence at computational cost
            tol: Convergence tolerance for early stopping - smaller values yield
                 more precise solutions but require more iterations
            random_seed: Random seed for reproducibility across runs
            learning_rate: Gradient descent step size - critical balance between
                          convergence speed and stability
            lambda_reg: Weight regularization strength - controls weight sparsity
                       for better interpretability
            lambda_sparsity: Archetype sparsity strength - higher values produce
                            more focused archetypes using fewer features
            sparsity_method: Technique for promoting archetype sparsity:
                - "l1": L1 regularization (fastest, robust, tends to zero out features)
                - "l0_approx": Approximated L0 regularization (more aggressive sparsity)
                - "feature_selection": Entropy-based selection (focuses on key features)
            normalize: Whether to normalize features - essential for data with
                      different scales
            projection_method: Method for projecting archetypes to convex hull:
                - "cbap": Convex boundary approximation (default, most stable)
                - "convex_hull": Exact convex hull vertices (more accurate)
                - "knn": K-nearest neighbors approximation (faster for large datasets)
            projection_alpha: Strength of boundary projection - higher values push
                             archetypes more aggressively toward extremes
            archetype_init_method: Initialization strategy for archetypes:
                - "directional": Directions from data centroid (robust default)
                - "qhull"/"convex_hull": Convex hull vertices (geometry-aware)
                - "kmeans"/"kmeans++": K-means++ initialization (density-aware)
            min_volume_factor: Minimum volume requirement for archetype simplex -
                              prevents degenerate solutions with collapsed archetypes
            **kwargs: Additional parameters including:
                - early_stopping_patience: Iterations with no improvement before stopping
                - verbose_level/logger_level: Controls logging detail
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
        self.min_volume_factor = min_volume_factor  # Parameter controlling the minimum volume of the convex hull.
        self.early_stopping_patience = kwargs.get("early_stopping_patience", 100)

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, archetypes: jnp.ndarray, weights: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Calculate the composite loss function incorporating sparsity constraints.

        This enhanced objective function extends the standard archetypal loss with
        multiple sparsity-promoting regularization terms, balancing several competing
        objectives:

        1. Reconstruction accuracy: Ensuring archetypes accurately represent the data
        2. Archetype sparsity: Promoting focused archetypes that use fewer features
        3. Weight interpretability: Encouraging sparse, distinctive weight patterns
        4. Boundary alignment: Maintaining archetypes at meaningful data extremes
        5. Archetype diversity: Preventing redundant or overlapping archetypes

        The balance between these terms is critical - too much emphasis on sparsity
        may sacrifice reconstruction quality, while too little won't yield the
        interpretability benefits.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)
            weights: Weight matrix (n_samples, n_archetypes)
            X: Data matrix (n_samples, n_features)

        Returns:
            Combined loss incorporating reconstruction error and multiple
            regularization terms
        """
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

        return jnp.asarray(total_loss).astype(jnp.float32)

    def _calculate_simplex_volume(self, archetypes: jnp.ndarray) -> float:
        """Calculate the volume of the simplex formed by archetypes.

        This geometric measure is critical for detecting and preventing degenerate
        solutions where archetypes collapse to similar positions. Such collapses
        significantly reduce model expressiveness and interpretability, as multiple
        archetypes would represent effectively the same pattern.

        The implementation handles two challenging cases:
        1. High-dimensional spaces where direct volume calculation is unstable
        2. Situations with fewer archetypes than dimensions, where the true volume
           would be zero mathematically

        In both cases, a robust proxy based on pairwise distances provides a reliable
        measure of archetype diversity.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)

        Returns:
            Volume or proxy measure of the archetype simplex - higher values
            indicate better-distributed archetypes
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
            return float(volume_proxy)
        else:
            try:
                # When there are enough points, attempt to compute the actual convex hull volume.
                hull = ConvexHull(archetypes)
                return float(hull.volume)
            except Exception:
                # Fallback to the pairwise distance approach if the convex hull computation fails.
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

        This enhanced update method extends the standard approach with critical
        improvements for sparse archetypal analysis:

        1. Sparsity promotion through the selected method (l1, l0, feature selection)
        2. Variance-aware noise injection to prevent dimensional collapse
        3. Constraint enforcement to maintain valid convex combinations
        4. Boundary projection to ensure archetypes remain at meaningful extremes

        These extensions are essential for learning truly interpretable archetypes
        while maintaining numerical stability and preventing convergence to
        degenerate solutions.

        Args:
            archetypes: Current archetype matrix (n_archetypes, n_features)
            weights: Weight matrix (n_samples, n_archetypes)
            X: Data matrix (n_samples, n_features)

        Returns:
            Updated archetypes incorporating sparsity and diversity constraints
        """
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
        def _constrain_to_convex_hull(archetype: jnp.ndarray) -> jnp.ndarray:
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

    def _apply_feature_selection(self, archetypes_updated: jnp.ndarray) -> jnp.ndarray:
        """Apply feature selection-based sparsity to archetypes.

        This sparsity method specifically targets interpretability by enhancing
        feature selectivity in each archetype. Rather than a uniform penalty on
        all features, it adaptively identifies and emphasizes the most significant
        features for each archetype while suppressing less important ones.

        This approach is particularly valuable when:
        1. Certain key features should dominate each archetype's interpretation
        2. The relative importance of features matters more than strict sparsity
        3. Some baseline activity across all features is expected or desirable

        The implementation uses soft thresholding with adaptive percentile cutoffs,
        providing a more nuanced approach than hard thresholding or L1 regularization.

        Args:
            archetypes_updated: Current archetype matrix to be sparsified

        Returns:
            Archetype matrix with enhanced feature selectivity
        """
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

    def diversify_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Prevent degenerate solutions by ensuring sufficient archetype diversity.

        This critical post-processing step addresses a fundamental challenge in
        archetypal analysis: optimization can sometimes converge to solutions where
        multiple archetypes collapse to similar positions, particularly when
        sparsity is enforced.

        Such degenerate solutions drastically reduce model expressiveness and
        interpretability. This method actively counteracts this tendency by:

        1. Detecting potential degeneracy through simplex volume measurement
        2. Systematically pushing archetypes away from each other when needed
        3. Ensuring archetypes remain valid (within the convex hull)
        4. Verifying improvement through before/after volume comparison

        This operation is performed outside the JAX-compiled update steps since it
        involves non-differentiable operations and contingent logic.

        Args:
            archetypes: Current archetypes matrix to diversify
            X: Data matrix defining the convex hull boundary

        Returns:
            Diversified archetypes with improved distribution and volume
        """
        n_archetypes = archetypes.shape[0]

        # Calculate initial volume or proxy.
        initial_volume = self._calculate_simplex_volume(archetypes)

        # If the volume is exceedingly small, attempt to increase it.
        if initial_volume < self.min_volume_factor:
            self.logger.info(
                f"Detected a potentially degenerate archetype configuration. "
                f"Volume proxy: {initial_volume:.6f}. Attempting to diversify."
            )

            # Calculate centroid.
            centroid = np.mean(X, axis=0)

            # For each archetype, push it away from other archetypes.
            for i in range(n_archetypes):
                # Calculate direction away from the average of other archetypes.
                other_archetypes = np.delete(archetypes, i, axis=0)
                other_centroid = np.mean(other_archetypes, axis=0)

                # Direction away from other archetypes.
                direction = archetypes[i] - other_centroid
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 1e-10:
                    normalized_direction = direction / direction_norm

                    # Determine how far we can push in this direction while remaining within the convex hull.
                    projections = np.dot(X - centroid, normalized_direction)
                    max_projection = np.max(projections)

                    current_projection = np.dot(archetypes[i] - centroid, normalized_direction)

                    # Push outward, but remain within the convex hull.
                    # Utilize a blend of the current position and the extreme point.
                    blend_factor = 0.5  # Move halfway toward the extreme point.
                    if current_projection < max_projection:
                        target_projection = current_projection + blend_factor * (max_projection - current_projection)
                        archetypes[i] = centroid + normalized_direction * target_projection

            # Re-normalize to maintain simplex constraints.
            row_sums = np.sum(archetypes, axis=1, keepdims=True)
            archetypes = archetypes / np.maximum(1e-10, row_sums)

            # Verify improvement.
            new_volume = self._calculate_simplex_volume(archetypes)
            self.logger.info(
                f"After diversification, volume proxy changed from {initial_volume:.6f} to {new_volume:.6f}"
            )

        return archetypes

    def get_archetype_sparsity(self) -> np.ndarray:
        """Calculate the effective sparsity of each archetype.

        This diagnostic method provides a quantitative measure of how successfully
        the sparsity constraints have been applied to each archetype. Rather than
        simply counting zeros (which may be unsuitable for soft-thresholded approaches),
        it uses the Gini coefficient as a more nuanced sparsity metric.

        The Gini coefficient measures the inequality among values, with higher values
        indicating greater sparsity (few large values, many small values). This provides:

        1. A standardized way to compare archetypes' feature utilization
        2. A continuous measure that works with both hard and soft sparsity
        3. A basis for identifying archetypes that may need further refinement
        4. A metric for evaluating different sparsity methods

        Returns:
            Array containing sparsity scores for each archetype (higher values
            indicate more focused archetypes using fewer features)
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
        """Fit the model to discover sparse, interpretable archetypes.

        This method orchestrates the complete sparse archetypal analysis process,
        building on the standard archetypal optimization while incorporating
        critical extensions for sparsity and stability:

        1. Leverages the parent class for core optimization
        2. Applies the selected sparsity-promoting method during optimization
        3. Performs post-processing to ensure archetype diversity
        4. Validates sparsity and geometric properties of the solution

        The result is a set of archetypes that balance reconstruction fidelity,
        interpretability, and geometric meaningfulness.

        Args:
            X: Data matrix (n_samples, n_features)
            normalize: Whether to normalize features before fitting - essential
                      for data with different scales
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

        # Utilize the parent class fit method.
        model = super().fit(X_np, normalize, **kwargs)

        # Apply post-processing to ensure archetype diversity.
        # This is executed outside the JAX-compiled optimization.
        if hasattr(model, "archetypes") and model.archetypes is not None:
            archetypes = jnp.asarray(model.archetypes)
            X_jax = jnp.asarray(X_np if not normalize else (X_np - model.X_mean) / model.X_std)

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
        """Fit the model and immediately transform the input data.

        This convenience method combines model fitting and data transformation
        in a single operation, which offers two key advantages:

        1. Computational efficiency by avoiding redundant calculations
        2. Simplified workflow for immediate archetype-based representation

        This method is particularly useful in analysis pipelines or when
        integrating with scikit-learn compatible frameworks that expect
        this pattern.

        Args:
            X: Data matrix to fit and transform (n_samples, n_features)
            y: Ignored. Present for scikit-learn API compatibility
            normalize: Whether to normalize features before fitting
            **kwargs: Additional parameters passed to fit()

        Returns:
            Weight matrix representing each sample as a combination of
            the discovered sparse archetypes (n_samples, n_archetypes)
        """
        X_np = X.values if hasattr(X, "values") else X
        model = self.fit(X_np, normalize, **kwargs)
        return model.transform(X_np)
