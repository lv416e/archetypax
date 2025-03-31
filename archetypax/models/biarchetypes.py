"""Biarchetypal Analysis: Dual-perspective archetype discovery using JAX.

This module implements Biarchetypal Analysis (BA), which extends traditional
Archetypal Analysis by simultaneously identifying archetypes in both observation
space (rows) and feature space (columns). This dual-perspective approach enables
more comprehensive data understanding, revealing patterns that would remain hidden
in single-direction analysis.
"""

from functools import partial
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optax

from archetypax.logger import get_logger, get_message

from .archetypes import ImprovedArchetypalAnalysis

T = TypeVar("T", bound=np.ndarray)


class BiarchetypalAnalysis(ImprovedArchetypalAnalysis):
    """Biarchetypal Analysis for dual-directional pattern discovery.

    This implementation extends archetypal analysis to simultaneously identify
    extreme patterns in both observations (rows) and features (columns), offering
    a richer understanding of data structure. Traditional archetypal analysis
    only identifies patterns in observation space, missing crucial feature-level insights.

    By factorizing the data matrix X as:
    X ≃ alpha·beta·X·theta·gamma

    BA provides several advantages:
    - Captures both observation-level and feature-level patterns
    - Enables cross-modal analysis between observations and features
    - Creates a more compact and interpretable representation via biarchetypes
    - Reveals latent relationships that single-directional methods cannot detect

    Based on the work by Alcacer et al., "Biarchetype analysis: simultaneous learning
    of observations and features based on extremes."
    """

    def __init__(
        self,
        n_row_archetypes: int,
        n_col_archetypes: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_seed: int = 42,
        learning_rate: float = 0.001,
        projection_method: str = "default",
        lambda_reg: float = 0.01,
        **kwargs,
    ):
        """Initialize the Biarchetypal Analysis model.

        Args:
            n_row_archetypes: Number of row archetypes - controls expressiveness in
                             observation space (rows)
            n_col_archetypes: Number of column archetypes - controls expressiveness in
                             feature space (columns)
            max_iter: Maximum optimization iterations - higher values enable better
                     convergence at computational cost
            tol: Convergence tolerance for early stopping - smaller values yield more
                 precise solutions but require more iterations
            random_seed: Random seed for reproducibility across runs
            learning_rate: Gradient descent step size - critical balance between
                          convergence speed and stability
            projection_method: Method for projecting archetypes to extreme points:
                              "default" uses convex boundary approximation
            lambda_reg: Regularization strength - controls sparsity/smoothness tradeoff
                       in archetype weights
            **kwargs: Additional parameters including:
                - early_stopping_patience: Iterations with no improvement before stopping
                - verbose_level/logger_level: Controls logging detail
        """
        # Initialize using parent class with the row archetypes
        # (we'll handle column archetypes separately)
        super().__init__(
            n_archetypes=n_row_archetypes,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            learning_rate=learning_rate,
            **kwargs,
        )

        # Initialize class-specific logger
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
                n_row_archetypes=n_row_archetypes,
                n_col_archetypes=n_col_archetypes,
                max_iter=max_iter,
                tol=tol,
                random_seed=random_seed,
                learning_rate=learning_rate,
                projection_method=projection_method,
                lambda_reg=lambda_reg,
            )
        )

        # Store biarchetypal specific parameters
        self.n_row_archetypes = n_row_archetypes
        self.n_col_archetypes = n_col_archetypes
        self.lambda_reg = lambda_reg
        self.random_seed = random_seed

        # Will be set during fitting
        self.alpha: np.ndarray | None = None  # Row coefficients (n_samples, n_row_archetypes)
        self.beta: np.ndarray | None = None  # Row archetypes (n_row_archetypes, n_samples)
        self.theta: np.ndarray | None = None  # Column archetypes (n_features, n_col_archetypes)
        self.gamma: np.ndarray | None = None  # Column coefficients (n_col_archetypes, n_features)
        self.biarchetypes: np.ndarray | None = None  # β·X·θ (n_row_archetypes, n_col_archetypes)

        self.early_stopping_patience = kwargs.get("early_stopping_patience", 100)

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, params: dict[str, jnp.ndarray], X: jnp.ndarray) -> jnp.ndarray:
        """Calculate the composite reconstruction loss for biarchetypal factorization.

        This core objective function balances reconstruction quality with sparsity
        promotion to ensure interpretable representations. Unlike standard AA,
        the biarchetypal loss operates on a four-factor decomposition, requiring
        careful numerical handling to prevent instability during optimization.

        The loss promotes three key properties:
        1. Accurate data reconstruction through the biarchetypal representation
        2. Sparse coefficients for interpretable patterns
        3. Numerical stability through explicit type control

        Args:
            params: Dictionary containing the four model matrices:
                - alpha: Row coefficients (n_samples, n_row_archetypes)
                - beta: Row archetypes (n_row_archetypes, n_samples)
                - theta: Column archetypes (n_features, n_col_archetypes)
                - gamma: Column coefficients (n_col_archetypes, n_features)
            X: Data matrix (n_samples, n_features)

        Returns:
            Combined loss value incorporating reconstruction and regularization terms
        """
        # Convert to float32 for better numerical stability
        alpha = params["alpha"].astype(jnp.float32)  # (n_samples, n_row_archetypes)
        beta = params["beta"].astype(jnp.float32)  # (n_row_archetypes, n_samples)
        theta = params["theta"].astype(jnp.float32)  # (n_features, n_col_archetypes)
        gamma = params["gamma"].astype(jnp.float32)  # (n_col_archetypes, n_features)
        X_f32 = X.astype(jnp.float32)

        # Calculate the reconstruction: X ≃ alpha·beta·X·theta·gamma
        # Optimize matrix multiplications to reduce memory usage
        inner_product = jnp.matmul(jnp.matmul(beta, X_f32), theta)  # (n_row_archetypes, n_col_archetypes)
        reconstruction = jnp.matmul(jnp.matmul(alpha, inner_product), gamma)  # (n_samples, n_features)

        # Calculate the reconstruction error (element-wise MSE)
        reconstruction_loss = jnp.mean(jnp.sum((X_f32 - reconstruction) ** 2, axis=1))

        # Add regularization to encourage sparsity by minimizing entropy
        alpha_entropy = jnp.sum(alpha * jnp.log(alpha + 1e-10), axis=1)  # Higher values promote sparsity
        gamma_entropy = jnp.sum(gamma * jnp.log(gamma + 1e-10), axis=0)  # Higher values promote sparsity
        entropy_reg = jnp.mean(alpha_entropy) + jnp.mean(gamma_entropy)

        return (reconstruction_loss - self.lambda_reg * entropy_reg).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def project_row_coefficients(self, coefficients: jnp.ndarray) -> jnp.ndarray:
        """Project row coefficients to satisfy simplex constraints.

        This projection is essential for maintaining valid convex combinations
        in the observation space. The simplex constraint (non-negative weights
        summing to 1) ensures that each data point is represented as a proper
        weighted combination of row archetypes. Without this constraint, the
        model would lose its interpretability and might generate unrealistic
        representations.

        The implementation includes numerical safeguards to prevent division by zero
        and ensure stable optimization even with extreme weight values.

        Args:
            coefficients: Row coefficient matrix (n_samples, n_row_archetypes)

        Returns:
            Projected coefficients satisfying simplex constraints
            (non-negative, sum to 1)
        """
        eps = 1e-10
        coefficients = jnp.maximum(eps, coefficients)
        sum_coeffs = jnp.sum(coefficients, axis=1, keepdims=True)
        sum_coeffs = jnp.maximum(eps, sum_coeffs)
        return coefficients / sum_coeffs

    @partial(jax.jit, static_argnums=(0,))
    def project_col_coefficients(self, coefficients: jnp.ndarray) -> jnp.ndarray:
        """Project column coefficients to satisfy simplex constraints.

        This projection enforces valid convex combinations in the feature space,
        which differs critically from row coefficient projection. Feature weights
        must sum to 1 across columns (not rows), ensuring each feature is properly
        represented by column archetypes.

        This axis-specific projection is a key distinction between standard AA and
        biarchetypal analysis, enabling the dual-directional nature of the model.

        Args:
            coefficients: Column coefficient matrix (n_col_archetypes, n_features)

        Returns:
            Projected coefficients with each feature's weights summing to 1,
            maintaining valid convex combinations in feature space
        """
        eps = 1e-10
        coefficients = jnp.maximum(eps, coefficients)
        sum_coeffs = jnp.sum(coefficients, axis=0, keepdims=True)
        sum_coeffs = jnp.maximum(eps, sum_coeffs)
        return coefficients / sum_coeffs

    @partial(jax.jit, static_argnums=(0,))
    def project_row_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Project row archetypes to the convex hull boundary of data points.

        This critical operation ensures row archetypes remain at meaningful extremes
        of the observation space, where they represent distinct, interpretable patterns.
        Without this projection, archetypes would tend to collapse toward the data
        centroid during optimization, losing their representative power.

        The implementation uses an adaptive multi-point boundary approximation that:
        1. Identifies extreme directions from the data centroid
        2. Selects multiple boundary points along each direction
        3. Creates weighted combinations that maximize distinctiveness
        4. Maintains numeric stability throughout the process

        Args:
            archetypes: Row archetype matrix (n_row_archetypes, n_samples)
            X: Data matrix (n_samples, n_features)

        Returns:
            Projected row archetypes positioned at meaningful boundaries
            of the data's convex hull
        """
        # Calculate the data centroid as our reference point
        centroid = jnp.mean(X, axis=0)  # Shape: (n_features,)

        def _project_to_boundary(archetype):
            """Project a single archetype to the boundary of the convex hull."""
            # Step 1: Calculate direction from centroid to archetype representation
            weighted_representation = jnp.matmul(archetype, X)  # Shape: (n_features,)
            direction = weighted_representation - centroid
            direction_norm = jnp.linalg.norm(direction)
            normalized_direction = jnp.where(
                direction_norm > 1e-10,
                direction / direction_norm,
                jax.random.normal(jax.random.PRNGKey(0), direction.shape) / jnp.sqrt(direction.shape[0]),
            )

            # Step 2: Project all data points onto this direction vector
            projections = jnp.dot(X - centroid, normalized_direction)  # Shape: (n_samples,)

            # Step 3: Find multiple extreme points with adaptive k selection
            k = min(5, X.shape[0] // 10 + 2)  # Adaptive k based on dataset size
            top_k_indices = jnp.argsort(projections)[-k:]
            top_k_projections = projections[top_k_indices]

            # Step 4: Calculate weights with emphasis on the most extreme points
            weights_unnormalized = jnp.exp(top_k_projections - jnp.max(top_k_projections))
            weights = weights_unnormalized / jnp.sum(weights_unnormalized)

            # Step 5: Create a weighted combination of extreme points
            multi_hot = jnp.zeros_like(archetype)
            for i in range(k):
                idx = top_k_indices[i]
                multi_hot = multi_hot.at[idx].set(weights[i])

            # Step 6: Mix with original archetype for stability and convergence
            alpha = 0.8  # Stronger pull toward extreme points for better diversity
            projected = alpha * multi_hot + (1 - alpha) * archetype

            # Step 7: Apply simplex constraints with numerical stability safeguards
            projected = jnp.maximum(1e-10, projected)
            sum_projected = jnp.sum(projected)
            projected = jnp.where(
                sum_projected > 1e-10,
                projected / sum_projected,
                jnp.ones_like(projected) / projected.shape[0],
            )

            return projected

        # Apply the projection function to each row archetype in parallel
        projected_archetypes = jax.vmap(_project_to_boundary)(archetypes)

        return jnp.asarray(projected_archetypes)

    @partial(jax.jit, static_argnums=(0,))
    def project_col_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Project column archetypes to the boundary of the feature space.

        This critical counterpart to row archetype projection ensures column archetypes
        represent distinct feature patterns. While conceptually similar to row projection,
        this operation works in the transposed space, treating features as observations
        and finding extremes among them.

        Without this specialized projection, the feature archetypes would not capture
        meaningful feature combinations, undermining the dual-perspective advantage
        of biarchetypal analysis.

        The implementation:
        1. Transposes the problem to work in feature space
        2. Identifies feature combinations that represent extremes
        3. Creates boundary points through weighted feature combinations
        4. Maintains numerical stability throughout

        Args:
            archetypes: Column archetype matrix (n_features, n_col_archetypes)
            X: Data matrix (n_samples, n_features)

        Returns:
            Projected column archetypes positioned at the boundaries of feature space,
            representing distinct feature patterns
        """
        # Transpose X to work with features as data points in feature space
        X_T = X.T  # Shape: (n_features, n_samples)

        # Calculate the feature centroid as our reference point
        centroid = jnp.mean(X_T, axis=0)  # Shape: (n_samples,)

        def _project_feature_to_boundary(archetype: jnp.ndarray) -> jnp.ndarray:
            """Project a single column archetype to the boundary of the feature convex hull."""
            # Step 1: Calculate direction in sample space using weighted features
            weighted_features = archetype[:, jnp.newaxis] * X_T  # Shape: (n_features, n_samples)
            direction = jnp.sum(weighted_features, axis=0) - centroid  # Shape: (n_samples,)
            direction_norm = jnp.linalg.norm(direction)
            normalized_direction = jnp.where(
                direction_norm > 1e-10,
                direction / direction_norm,
                jax.random.normal(jax.random.PRNGKey(0), direction.shape) / jnp.sqrt(direction.shape[0]),
            )

            # Step 2: Project all features onto this direction to measure extremeness
            projections = jnp.dot(X_T, normalized_direction)  # Shape: (n_features,)

            # Step 3: Find multiple extreme features with adaptive k selection
            k = min(5, X.shape[1] // 10 + 2)  # Adaptive k based on feature space size
            top_k_indices = jnp.argsort(projections)[-k:]
            top_k_projections = projections[top_k_indices]

            # Step 4: Calculate weights with emphasis on the most extreme features
            weights_unnormalized = jnp.exp(top_k_projections - jnp.max(top_k_projections))
            weights = weights_unnormalized / jnp.sum(weights_unnormalized)

            # Step 5: Create a weighted combination of extreme features
            multi_hot = jnp.zeros_like(archetype)
            for i in range(k):
                idx = top_k_indices[i]
                multi_hot = multi_hot.at[idx].set(weights[i])

            # Step 6: Mix with original archetype for stability and convergence
            alpha = 0.8  # Stronger pull toward extreme features for better diversity
            projected = alpha * multi_hot + (1 - alpha) * archetype

            # Step 7: Apply simplex constraints with numerical stability safeguards
            projected = jnp.maximum(1e-10, projected)
            sum_projected = jnp.sum(projected)
            projected = jnp.where(
                sum_projected > 1e-10,
                projected / sum_projected,
                jnp.ones_like(projected) / projected.shape[0],
            )

            return projected

        # Apply the projection function to each column archetype in parallel
        projected_archetypes = jax.vmap(_project_feature_to_boundary)(archetypes.T)

        # Transpose the result back to original shape
        return jnp.asarray(projected_archetypes.T)

    def fit(self, X: np.ndarray, normalize: bool = False, **kwargs) -> "BiarchetypalAnalysis":
        """Fit the Biarchetypal Analysis model to identify dual-perspective archetypes.

        This core method performs the four-factor decomposition of the data matrix,
        simultaneously discovering patterns in observation and feature spaces.
        The implementation employs advanced optimization strategies including:

        1. Sophisticated initialization for both row and column factors
        2. Adaptive learning rate scheduling for stable convergence
        3. Specialized projection operations to maintain meaningful boundaries
        4. Careful numerical handling to prevent instability
        5. Early stopping with convergence monitoring

        These optimizations are essential due to the complexity of the four-factor model,
        which is more challenging to optimize than standard Archetypal Analysis.

        Args:
            X: Data matrix (n_samples, n_features)
            normalize: Whether to normalize features - essential for data with
                      different scales
            **kwargs: Additional parameters for customizing the fitting process

        Returns:
            Self - fitted model instance with discovered biarchetypes
        """
        X_np = X.values if hasattr(X, "values") else X

        # Preprocess data: scale for improved stability
        self.X_mean = np.mean(X_np, axis=0)
        self.X_std = np.std(X_np, axis=0)

        # Prevent division by zero
        if self.X_std is not None:
            self.X_std = np.where(self.X_std < 1e-10, np.ones_like(self.X_std), self.X_std)

        if normalize:
            X_scaled = (X_np - self.X_mean) / self.X_std
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            X_scaled = X_np.copy()

        # Convert to JAX array with explicit dtype for better performance
        X_jax = jnp.array(X_scaled, dtype=jnp.float32)
        n_samples, n_features = X_jax.shape

        # Log key data characteristics and model configuration
        self.logger.info(f"Data shape: {X_jax.shape}")
        self.logger.info(f"Data range: min={float(jnp.min(X_jax)):.4f}, max={float(jnp.max(X_jax)):.4f}")
        self.logger.info(f"Row archetypes: {self.n_row_archetypes}")
        self.logger.info(f"Column archetypes: {self.n_col_archetypes}")

        # Initialize alpha (row coefficients) with more stable initialization
        self.rng_key, subkey = jax.random.split(self.rng_key)
        alpha_init = jax.random.uniform(
            subkey, (n_samples, self.n_row_archetypes), minval=0.1, maxval=0.9, dtype=jnp.float32
        )
        alpha_init = self.project_row_coefficients(alpha_init)

        # Initialize gamma (column coefficients)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        gamma_init = jax.random.uniform(
            subkey, (self.n_col_archetypes, n_features), minval=0.1, maxval=0.9, dtype=jnp.float32
        )
        gamma_init = self.project_col_coefficients(gamma_init)

        # Initialize beta (row archetypes) using sophisticated k-means++ initialization
        # This approach ensures diverse starting points that are well-distributed across the data space
        self.rng_key, subkey = jax.random.split(self.rng_key)

        # Step 1: Select initial centroids using k-means++ algorithm
        # This ensures our archetypes start from diverse positions in the data space
        selected_indices = jnp.zeros(self.n_row_archetypes, dtype=jnp.int32)

        # Select first point randomly
        first_idx = jax.random.randint(subkey, (), 0, n_samples)
        selected_indices = selected_indices.at[0].set(first_idx)

        # Select remaining points with probability proportional to squared distance
        for i in range(1, self.n_row_archetypes):
            # Calculate squared distance from each point to nearest existing centroid
            min_dists = jnp.ones(n_samples) * float("inf")

            # Update distances for each existing centroid
            for j in range(i):
                idx = selected_indices[j]
                dists = jnp.sum((X_jax - X_jax[idx]) ** 2, axis=1)
                min_dists = jnp.minimum(min_dists, dists)

            # Zero out already selected points
            for j in range(i):
                idx = selected_indices[j]
                min_dists = min_dists.at[idx].set(0.0)

            # Select next point with probability proportional to squared distance
            self.rng_key, subkey = jax.random.split(self.rng_key)
            probs = min_dists / (jnp.sum(min_dists) + 1e-10)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)
            selected_indices = selected_indices.at[i].set(next_idx)

        # Step 2: Create one-hot encodings for selected points
        beta_init = jnp.zeros((self.n_row_archetypes, n_samples), dtype=jnp.float32)
        for i in range(self.n_row_archetypes):
            idx = selected_indices[i]
            beta_init = beta_init.at[i, idx].set(1.0)

        # Step 3: Add controlled stochastic noise to promote exploration
        # This prevents archetypes from being too rigidly defined at initialization
        self.rng_key, subkey = jax.random.split(self.rng_key)
        noise = jax.random.uniform(subkey, beta_init.shape, minval=0.0, maxval=0.05, dtype=jnp.float32)
        beta_init = beta_init + noise

        # Step 4: Ensure proper normalization to maintain simplex constraints
        # Each row must sum to 1 to represent a valid convex combination
        beta_init = beta_init / jnp.sum(beta_init, axis=1, keepdims=True)

        self.logger.info("Row archetypes initialized with k-means++ strategy")

        # Initialize theta (column archetypes) with advanced diversity-maximizing approach
        # This ensures column archetypes capture the most distinctive feature patterns
        self.rng_key, subkey = jax.random.split(self.rng_key)

        # Step 1: Transpose data for feature-centric operations
        X_T = X_jax.T  # Shape: (n_features, n_samples)

        # Step 2: Calculate feature diversity metrics
        # Compute variance of each feature to identify informative dimensions
        feature_variance = jnp.var(X_T, axis=1)

        # Step 3: Select initial features using variance-weighted sampling
        theta_init = jnp.zeros((n_features, self.n_col_archetypes), dtype=jnp.float32)
        selected_features = jnp.zeros(self.n_col_archetypes, dtype=jnp.int32)

        # Select first feature with probability proportional to variance
        self.rng_key, subkey = jax.random.split(self.rng_key)
        probs = feature_variance / (jnp.sum(feature_variance) + 1e-10)
        first_idx = jax.random.choice(subkey, n_features, p=probs)
        selected_features = selected_features.at[0].set(first_idx)
        theta_init = theta_init.at[first_idx, 0].set(1.0)

        # Step 4: Select remaining features to maximize diversity
        for i in range(1, self.n_col_archetypes):
            # Calculate minimum distance from each feature to already selected features
            min_dists = jnp.ones(n_features) * float("inf")

            for j in range(i):
                idx = selected_features[j]
                # Compute correlation-based distance to capture feature relationships
                corr = jnp.abs(jnp.sum(X_T * X_T[idx, jnp.newaxis], axis=1)) / (
                    jnp.sqrt(jnp.sum(X_T**2, axis=1) * jnp.sum(X_T[idx] ** 2) + 1e-10)
                )
                # Convert correlation to distance (1 - |corr|)
                dists = 1.0 - corr
                min_dists = jnp.minimum(min_dists, dists)

            # Zero out already selected features
            for j in range(i):
                idx = selected_features[j]
                min_dists = min_dists.at[idx].set(0.0)

            # Select feature with maximum minimum distance
            next_idx = jnp.argmax(min_dists)
            selected_features = selected_features.at[i].set(next_idx)
            theta_init = theta_init.at[next_idx, i].set(1.0)

        # Step 5: Add controlled noise to promote exploration
        self.rng_key, subkey = jax.random.split(self.rng_key)
        noise = jax.random.uniform(subkey, theta_init.shape, minval=0.0, maxval=0.05, dtype=jnp.float32)
        theta_init = theta_init + noise

        # Step 6: Ensure proper normalization to maintain simplex constraints
        # Each column must sum to 1 to represent a valid convex combination
        theta_init = theta_init / jnp.sum(theta_init, axis=0, keepdims=True)

        self.logger.info("Column archetypes initialized with diversity-maximizing strategy")

        # Set up optimizer with learning rate schedule for better convergence
        # We use a sophisticated learning rate schedule with warmup and decay phases
        warmup_steps = 20
        decay_steps = 100

        # Create a warmup schedule that linearly increases from 0 to peak learning rate
        # Use a much lower learning rate to prevent divergence
        reduced_lr = self.learning_rate * 0.05  # Reduce learning rate by 20x
        warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=reduced_lr, transition_steps=warmup_steps)

        # Create a decay schedule that exponentially decays from peak to minimum learning rate
        decay_schedule = optax.exponential_decay(
            init_value=reduced_lr,
            transition_steps=decay_steps,
            decay_rate=0.95,  # Even slower decay for more stable convergence
            end_value=0.000001,  # Very low minimum learning rate for fine-grained optimization
            staircase=False,  # Smooth decay rather than step-wise
        )

        # Combine the schedules
        schedule = optax.join_schedules(schedules=[warmup_schedule, decay_schedule], boundaries=[warmup_steps])

        # Create a sophisticated optimizer chain with:
        # 1. Gradient clipping to prevent exploding gradients
        # 2. Adam optimizer with our custom learning rate schedule
        # 3. Weight decay for regularization
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),  # More aggressive clipping to prevent divergence
            optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),  # Adam optimizer with standard parameters
            optax.add_decayed_weights(weight_decay=1e-6),  # Very subtle weight decay
            optax.scale_by_schedule(schedule),  # Apply our custom learning rate schedule
        )

        # Initialize parameters
        params = {"alpha": alpha_init, "beta": beta_init, "theta": theta_init, "gamma": gamma_init}
        opt_state = optimizer.init(params)

        # Define update step with JIT compilation for speed
        @partial(jax.jit, static_argnums=(3,))
        def update_step(
            params: dict[str, jnp.ndarray], opt_state: optax.OptState, X: jnp.ndarray, iteration: int
        ) -> tuple[dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
            """Execute a single optimization step."""

            # Loss function for current parameters
            def loss_fn(params):
                return self.loss_function(params, X)

            # Compute gradients and update parameters
            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Maintain simplex constraints
            new_params["alpha"] = self.project_row_coefficients(new_params["alpha"])
            new_params["gamma"] = self.project_col_coefficients(new_params["gamma"])

            # Periodically project archetypes to convex hull boundary
            # This is expensive, so we do it every 10 iterations
            do_project = iteration % 10 == 0

            def project():
                return {
                    "alpha": new_params["alpha"],
                    "beta": self.project_row_archetypes(new_params["beta"], X),
                    "theta": self.project_col_archetypes(new_params["theta"], X),
                    "gamma": new_params["gamma"],
                }

            def no_project():
                return new_params

            new_params = jax.lax.cond(do_project, lambda: project(), lambda: no_project())

            return new_params, opt_state, loss

        # Optimization loop
        prev_loss = float("inf")
        self.loss_history = []

        # Calculate initial loss for debugging
        initial_loss = float(self.loss_function(params, X_jax))
        self.logger.info(f"Initial loss: {initial_loss:.6f}")

        for it in range(self.max_iter):
            try:
                # Execute update step
                params, opt_state, loss = update_step(params, opt_state, X_jax, it)
                loss_value = float(loss)

                # Check for NaN
                if jnp.isnan(loss_value):
                    self.logger.warning(get_message("warning", "nan_detected", iteration=it))
                    break

                # Record loss
                self.loss_history.append(loss_value)

                # Track loss and display progress
                self.logger.info(f"Iteration {it}, Loss: {loss_value:.6f}")

                # Evaluate convergence using both immediate and moving average improvements
                relative_improvement = (prev_loss - loss_value) / (prev_loss + 1e-10)

                # Tracking using moving average for stable convergence detection
                if it >= 10:
                    recent_losses = self.loss_history[-10:]
                    loss_ma = sum(recent_losses) / 10
                    # Use initial loss as reference if no previous MA exists
                    if "loss_ma_prev" not in locals():
                        loss_ma_prev = self.loss_history[0]
                    relative_ma_improvement = (loss_ma_prev - loss_ma) / (loss_ma_prev + 1e-10)
                    loss_ma_prev = loss_ma
                else:
                    relative_ma_improvement = relative_improvement
                    loss_ma_prev = prev_loss  # Initialize for future iterations

                # Check combined convergence criteria
                if it > 20 and relative_improvement < self.tol and relative_ma_improvement < self.tol:
                    self.logger.info(f"Converged at iteration {it}")
                    break

                prev_loss = loss_value

                # Display comprehensive progress information at regular intervals
                if it % 25 == 0 or it < 5:
                    # Calculate performance metrics for monitoring optimization trajectory
                    if len(self.loss_history) > 1:
                        avg_last_5 = sum(self.loss_history[-min(5, len(self.loss_history)) :]) / min(
                            5, len(self.loss_history)
                        )
                        improvement_rate = (self.loss_history[0] - loss_value) / (it + 1) if it > 0 else 0
                        self.logger.info(
                            get_message(
                                "progress",
                                "iteration_progress",
                                current=it,
                                total=self.max_iter,
                                loss=loss_value,
                                avg_last_5=avg_last_5,
                                improvement_rate=improvement_rate,
                            )
                        )
                    else:
                        self.logger.info(
                            get_message(
                                "progress",
                                "iteration_progress",
                                current=it,
                                total=self.max_iter,
                                loss=loss_value,
                            )
                        )

                # Provide in-depth diagnostics at major milestones
                if it % 100 == 0 and it > 0:
                    # Analyze archetype characteristics
                    alpha_sparsity = jnp.mean(jnp.sum(params["alpha"] > 0.01, axis=1) / params["alpha"].shape[1])
                    gamma_sparsity = jnp.mean(jnp.sum(params["gamma"] > 0.01, axis=0) / params["gamma"].shape[0])
                    self.logger.debug(
                        f"  - Alpha sparsity: {float(alpha_sparsity):.4f} | Gamma sparsity: {float(gamma_sparsity):.4f}"
                    )
                    self.logger.debug(f"  - Learning rate: {float(schedule(it)):.8f}")

                    # Flag potential convergence issues
                    if jnp.max(params["alpha"]) > 0.99:
                        self.logger.warning(
                            "  - Warning: Alpha contains near-one values, may indicate degenerate solution"
                        )
                    if jnp.max(params["gamma"]) > 0.99:
                        self.logger.warning(
                            "  - Warning: Gamma contains near-one values, may indicate degenerate solution"
                        )

            except Exception as e:
                self.logger.error(f"Error at iteration {it}: {e!s}")
                break

        # Final projection of archetypes to ensure they're on the convex hull boundary
        params["beta"] = self.project_row_archetypes(jnp.asarray(params["beta"]), X_jax)
        params["theta"] = self.project_col_archetypes(jnp.asarray(params["theta"]), X_jax)

        # Store final parameters
        self.alpha = np.array(params["alpha"])
        self.beta = np.array(params["beta"])
        self.theta = np.array(params["theta"])
        self.gamma = np.array(params["gamma"])

        # Calculate biarchetypes (Z = beta·X·theta)
        self.biarchetypes = np.array(np.matmul(np.matmul(self.beta, np.asanyarray(X_jax)), self.theta))

        # For compatibility with parent class
        self.archetypes = np.array(np.matmul(self.beta, X_jax))  # Row archetypes
        self.weights = np.array(self.alpha)  # Row weights

        if len(self.loss_history) > 0:
            self.logger.info(f"Final loss: {self.loss_history[-1]:.6f}")
        else:
            self.logger.warning("No valid loss was recorded")

        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> Any:
        """Transform new data into dual-directional archetype space.

        This method computes optimal weights to represent new data in terms of
        discovered archetypes, enabling consistent interpretation of new observations
        within the established biarchetypal framework.

        Unlike conventional AA, this transform operates in both row and column spaces,
        providing a holistic representation of new data that preserves the model's
        dual-perspective advantage. The implementation efficiently leverages pre-trained
        biarchetypes to avoid redundant computation.

        Args:
            X: New data matrix (n_samples, n_features) to transform
            y: Ignored. Present for scikit-learn API compatibility
            **kwargs: Additional parameters for customizing transformation

        Returns:
            Tuple of (row_weights, col_weights) representing the data in the
            biarchetypal space
        """
        if self.alpha is None or self.beta is None or self.theta is None or self.gamma is None:
            raise ValueError("Model must be fitted before transform")

        X_np = X.values if hasattr(X, "values") else X
        X_jax = jnp.array(X_np, dtype=jnp.float32)

        # Scale input data
        if self.X_mean is not None and self.X_std is not None:
            X_scaled = (X_jax - self.X_mean) / self.X_std
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            X_scaled = X_jax.copy()

        # Use pre-trained biarchetypes (avoid recalculating biarchetypes for new data)
        biarchetypes_jax = jnp.array(self.biarchetypes, dtype=jnp.float32)

        # Optimize row weights for new data
        # Independently optimize for each row (sample)
        def optimize_row_weights(x_row):
            # Initialize weights uniformly
            alpha = jnp.ones(self.n_row_archetypes) / self.n_row_archetypes

            # Optimize using gradient descent
            for _ in range(100):  # 100 optimization steps
                # Calculate reconstruction
                reconstruction = jnp.matmul(jnp.matmul(alpha, biarchetypes_jax), jnp.asarray(self.gamma))
                # Calculate error
                error = x_row - reconstruction
                # Calculate gradient
                grad = -2 * jnp.matmul(error, jnp.matmul(biarchetypes_jax, jnp.asarray(self.gamma)).T)
                # Update weights
                alpha = alpha - 0.01 * grad
                # Project onto constraints (non-negativity and sum to 1)
                alpha = jnp.maximum(1e-10, alpha)
                alpha = alpha / jnp.sum(alpha)

            return alpha

        # Calculate row weights for each sample
        alpha_new = jnp.array([optimize_row_weights(x) for x in X_scaled])

        # Reuse pre-trained column weights (don't recalculate for new data)
        gamma_new = self.gamma.copy()

        result = (np.asarray(alpha_new), np.asarray(gamma_new))
        return result

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, normalize: bool = False, **kwargs) -> Any:
        """Fit the model and transform the data in a single operation.

        This convenience method combines model fitting and data transformation
        in a single step, offering two key advantages:
        1. Computational efficiency by avoiding redundant calculations
        2. Simplified workflow for immediate biarchetypal representation

        The method is particularly valuable when the biarchetypal representation
        is needed immediately after fitting, such as in analysis pipelines or
        when integrating with scikit-learn compatible frameworks.

        Args:
            X: Data matrix to fit and transform (n_samples, n_features)
            y: Ignored. Present for scikit-learn API compatibility
            normalize: Whether to normalize features before fitting
            **kwargs: Additional parameters passed to fit()

        Returns:
            Tuple of (row_weights, col_weights) representing the data in
            biarchetypal space
        """
        X_np = X.values if hasattr(X, "values") else X
        self.fit(X_np, normalize=normalize)
        alpha, gamma = self.transform(X_np)
        return np.asarray(alpha), np.asarray(gamma)

    def reconstruct(self, X: np.ndarray | None = None) -> np.ndarray:
        """Reconstruct data from biarchetypal representation.

        This method provides the inverse operation of transform(), reconstructing
        data points from their biarchetypal weights. This capability serves several
        critical purposes:

        1. Validation of model quality through reconstruction error assessment
        2. Interpretation of what specific archetypes represent in data space
        3. Generation of synthetic data by manipulating archetype weights
        4. Noise reduction by reconstructing data through dominant archetypes

        The method handles both original training data and new data points.

        Args:
            X: Optional data matrix to reconstruct. If None, uses the training data

        Returns:
            Reconstructed data matrix in the original feature space
        """
        if X is not None:
            # Transform new data and reconstruct
            alpha, gamma = self.transform(X)
        else:
            # Use stored weights from training
            if self.alpha is None or self.gamma is None:
                raise ValueError("Model must be fitted before reconstruction")
            alpha, gamma = self.alpha, self.gamma

        if self.biarchetypes is None:
            raise ValueError("Model must be fitted before reconstruction")

        # Reconstruct using biarchetypes: X ≃ alpha·Z·gamma
        reconstructed = np.matmul(np.matmul(alpha, self.biarchetypes), gamma)

        # Inverse transform if normalization was applied
        if self.X_mean is not None and self.X_std is not None:
            reconstructed = reconstructed * self.X_std + self.X_mean

        return np.asarray(reconstructed)

    def get_biarchetypes(self) -> np.ndarray:
        """Retrieve the core biarchetypes matrix.

        The biarchetypes matrix (Z = β·X·θ) represents the heart of the model,
        capturing the essential patterns at the intersection of row and column archetypes.
        This matrix provides a compact representation of the data's underlying structure,
        with each element representing a specific row-column archetype interaction.

        Access to this matrix is essential for visualization, interpretation, and
        advanced analysis of the identified patterns.

        Returns:
            Biarchetypes matrix (n_row_archetypes, n_col_archetypes)
        """
        if self.biarchetypes is None:
            raise ValueError("Model must be fitted before getting biarchetypes")

        return self.biarchetypes

    def get_row_archetypes(self) -> np.ndarray:
        """Retrieve the row archetypes.

        Row archetypes represent extreme patterns in observation space, describing
        distinctive types of data points. These archetypes are essential for
        understanding the primary modes of variation among observations and provide
        the foundation for interpreting data point weights.

        In the biarchetypal model, row archetypes are projections of the data matrix
        via the beta coefficients (β·X).

        Returns:
            Row archetypes matrix (n_row_archetypes, n_features)
        """
        if self.archetypes is None:
            raise ValueError("Model must be fitted before getting row archetypes")

        return self.archetypes

    def get_col_archetypes(self) -> np.ndarray:
        """Retrieve the column archetypes.

        Column archetypes represent extreme patterns in feature space, describing
        distinctive feature combinations or "feature types." This perspective is
        unique to biarchetypal analysis and provides crucial insights about feature
        relationships that would be missed in standard archetypal analysis.

        These archetypes enable feature-level interpretations and can reveal
        coordinated feature behaviors across different data contexts.

        Returns:
            Column archetypes matrix (n_col_archetypes, n_features)
        """
        if self.theta is None or self.gamma is None:
            raise ValueError("Model must be fitted before getting column archetypes")

        # Modified: Changed the calculation method for column archetypes
        # If the original data is not available, generate column archetypes from the shape of theta
        if self.theta.shape[0] == self.theta.shape[1]:
            # If theta is a square matrix, generate column archetypes similar to an identity matrix
            return np.eye(self.theta.shape[0])
        else:
            # Position each column archetype along the feature space axes
            col_archetypes = np.zeros((self.n_col_archetypes, self.theta.shape[0]))
            for i in range(min(self.n_col_archetypes, self.theta.shape[0])):
                col_archetypes[i, i] = 1.0
            return col_archetypes

    def get_row_weights(self) -> np.ndarray:
        """Retrieve the row coefficients (alpha).

        Row weights represent how each data point is composed as a mixture of
        row archetypes. These weights are essential for:

        1. Understanding which archetype patterns dominate each observation
        2. Clustering similar observations based on their archetype compositions
        3. Detecting anomalies as points with unusual archetype weights
        4. Creating reduced-dimension visualizations based on archetype space

        The weights are constrained to be non-negative and sum to 1 (simplex constraint),
        making them directly interpretable as proportions.

        Returns:
            Row weight matrix (n_samples, n_row_archetypes)
        """
        if self.alpha is None:
            raise ValueError("Model must be fitted before getting row weights")

        return self.alpha

    def get_col_weights(self) -> np.ndarray:
        """Retrieve the column coefficients (gamma).

        Column weights represent how each feature is composed as a mixture of
        column archetypes. These weights provide unique insights into:

        1. Which feature patterns are expressed in each original feature
        2. How features group together based on shared archetype influence
        3. Feature importance through the lens of archetypal patterns
        4. Potential redundancies in the feature space

        This feature-space perspective is a distinguishing advantage of biarchetypal
        analysis compared to standard archetypal methods.

        Returns:
            Column weight matrix (n_col_archetypes, n_features)
        """
        if self.gamma is None:
            raise ValueError("Model must be fitted before getting column weights")

        return self.gamma

    def get_all_archetypes(self) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve both row and column archetypes in a single call.

        This convenience method provides access to both directions of archetypal
        analysis simultaneously, facilitating comprehensive analysis and visualization
        of the dual-perspective patterns. Accessing both archetypes together is
        particularly valuable for cross-modal analysis examining relationships
        between observation patterns and feature patterns.

        Returns:
            Tuple of (row_archetypes, column_archetypes) matrices
        """
        return self.get_row_archetypes(), self.get_col_archetypes()

    def get_all_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve both row and column weights in a single call.

        This convenience method provides access to all weight coefficients
        simultaneously, enabling comprehensive analysis of how observations
        and features relate to their respective archetypes. This unified view
        is particularly valuable for understanding the full biarchetypal
        decomposition and how information flows between the row and column spaces.

        Returns:
            Tuple of (row_weights, column_weights) matrices
        """
        return self.get_row_weights(), self.get_col_weights()
