"""Biarchetypal Analysis using JAX.

This module extends ImprovedArchetypalAnalysis to perform dual-directional pattern discovery,
identifying archetypes in both observation space (rows) and feature space (columns) simultaneously.
While traditional archetypal analysis only finds patterns in one direction,
biarchetypal analysis provides a richer understanding by decomposing data as: X ≃ alpha·beta·X·theta·gamma

Core Features:
- Discovers extreme patterns in both observations and features
- Reveals cross-modal relationships between row and column archetypes
- Creates more interpretable representation via biarchetypes
- Handles complex data with interdependent row and column structures

The four-factor decomposition offers deeper insights than traditional methods by capturing
how observation patterns interact with feature patterns throughout the data.

Example usage:
    ```python
    from archetypax.models import BiarchetypalAnalysis

    # Initialize model with separate row and column archetype counts
    model = BiarchetypalAnalysis(
        n_row_archetypes=4,
        n_col_archetypes=3,
        projection_method="default",
        normalize=True
    )

    # Fit model and get dual-directional representations
    row_weights, col_weights = model.fit_transform(X)

    # Extract bi-archetypes matrix (core patterns)
    biarchetypes = model.get_biarchetypes()
    ```

Based on Alcacer et al., "Biarchetype analysis: simultaneous learning of observations
and features based on extremes."
"""

from functools import partial
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optax

from archetypax.logger import get_logger, get_message
from archetypax.models.archetypes import ImprovedArchetypalAnalysis

T = TypeVar("T", bound=np.ndarray)


class BiarchetypalAnalysis(ImprovedArchetypalAnalysis):
    """Biarchetypal Analysis for dual-directional pattern discovery.

    This implementation extends archetypal analysis to simultaneously identify
    extreme patterns in both observations (rows) and features (columns),
    offering a richer understanding of data structure.
    Traditional archetypal analysis only identifies patterns
    in observation space, missing crucial feature-level insights.

    By factorizing the data matrix X as:
    X ≃ alpha·beta·X·theta·gamma

    BA provides several advantages:
    - Captures both observation-level and feature-level patterns
    - Enables cross-modal analysis between observations and features
    - Creates a more compact and interpretable representation via biarchetypes
    - Reveals latent relationships that single-directional methods cannot detect

    This implementation is based on the work by Alcacer et al.,
    "Biarchetype analysis: simultaneous learning of observations and features based on extremes."
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
            n_row_archetypes:
                Number of row archetypes - controls expressiveness in observation space (rows)
            n_col_archetypes:
                Number of column archetypes - controls expressiveness in feature space (columns)
            max_iter:
                Maximum optimization iterations
                - higher values enable better convergence at computational cost
            tol:
                Convergence tolerance for early stopping
                - smaller values yield more precise solutions but require more iterations
            random_seed:
                Random seed for reproducibility across runs
            learning_rate:
                Gradient descent step size
                - critical balance between convergence speed and stability
            projection_method:
                Method for projecting archetypes to extreme points:
                - "default" uses convex boundary approximation
                - "convex_hull" uses convex hull approximation
                - "knn" uses k-nearest neighbors approximation
            lambda_reg:
                Regularization strength
                - controls sparsity/smoothness tradeoff n archetype weights
            **kwargs: Additional parameters including:
                - early_stopping_patience: Iterations with no improvement before stopping
                - verbose_level/logger_level: Controls logging detail
        """
        # Initialize using parent class with the row archetypes
        super().__init__(
            n_archetypes=n_row_archetypes,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            learning_rate=learning_rate,
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

        self.eps = jnp.finfo(jnp.float32).eps
        self.lambda_reg = lambda_reg
        self.random_seed = random_seed

        self.n_row_archetypes = n_row_archetypes
        self.n_col_archetypes = n_col_archetypes
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
        promotion to ensure interpretable representations.
        Unlike standard AA, the biarchetypal loss operates on a four-factor decomposition,
        requiring careful numerical handling to prevent instability during optimization.

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
        alpha_entropy = jnp.sum(alpha * jnp.log(alpha + self.eps), axis=1)
        gamma_entropy = jnp.sum(gamma * jnp.log(gamma + self.eps), axis=0)
        entropy_reg = jnp.mean(alpha_entropy) + jnp.mean(gamma_entropy)

        return (reconstruction_loss - self.lambda_reg * entropy_reg).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def project_row_coefficients(self, coefficients: jnp.ndarray) -> jnp.ndarray:
        """Project row coefficients to satisfy simplex constraints.

        This projection is essential for maintaining valid convex combinations in the observation space.
        The simplex constraint (non-negative weights summing to 1) ensures that
        each data point is represented as a proper weighted combination of row archetypes.
        Without this constraint, the model would lose its interpretability
        and might generate unrealistic representations.

        The implementation includes numerical safeguards to prevent division by zero
        and ensure stable optimization even with extreme weight values.

        Args:
            coefficients: Row coefficient matrix (n_samples, n_row_archetypes)

        Returns:
            Projected coefficients satisfying simplex constraints
            (non-negative, sum to 1)
        """
        coefficients = jnp.maximum(self.eps, coefficients)
        sum_coeffs = jnp.sum(coefficients, axis=1, keepdims=True)
        sum_coeffs = jnp.maximum(self.eps, sum_coeffs)
        return coefficients / sum_coeffs

    @partial(jax.jit, static_argnums=(0,))
    def project_col_coefficients(self, coefficients: jnp.ndarray) -> jnp.ndarray:
        """Project column coefficients to satisfy simplex constraints.

        This projection enforces valid convex combinations in the feature space,
        which differs critically from row coefficient projection.
        Feature weights must sum to 1 across columns (not rows),
        ensuring each feature is properly represented by column archetypes.

        This axis-specific projection is a key distinction between standard AA and biarchetypal analysis,
        enabling the dual-directional nature of the model.

        Args:
            coefficients: Column coefficient matrix (n_col_archetypes, n_features)

        Returns:
            Projected coefficients with each feature's weights summing to 1,
            maintaining valid convex combinations in feature space
        """
        coefficients = jnp.maximum(self.eps, coefficients)
        sum_coeffs = jnp.sum(coefficients, axis=0, keepdims=True)
        sum_coeffs = jnp.maximum(self.eps, sum_coeffs)
        return coefficients / sum_coeffs

    @partial(jax.jit, static_argnums=(0,))
    def project_row_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Project row archetypes to the convex hull boundary of data points.

        This critical operation ensures row archetypes remain at meaningful extremes
        of the observation space, where they represent distinct, interpretable patterns.
        Without this projection, archetypes would tend to collapse toward the data centroid during optimization,
        losing their representative power.

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
        centroid = jnp.mean(X, axis=0)

        def _project_to_boundary(archetype):
            """Project a single archetype to the boundary of the convex hull."""
            # Step 1: Calculate direction from centroid to archetype representation
            weighted_representation = jnp.matmul(archetype, X)
            direction = weighted_representation - centroid
            direction_norm = jnp.linalg.norm(direction)
            normalized_direction = jnp.where(
                direction_norm > self.eps,
                direction / direction_norm,
                jax.random.normal(jax.random.PRNGKey(0), direction.shape) / jnp.sqrt(direction.shape[0]),
            )

            # Step 2: Project all data points onto this direction vector
            projections = jnp.dot(X - centroid, normalized_direction)

            # Step 3: Find multiple extreme points with adaptive k selection
            k = min(5, X.shape[0] // 10 + 2)
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
            alpha = 0.8
            projected = alpha * multi_hot + (1 - alpha) * archetype

            # Step 7: Apply simplex constraints with numerical stability safeguards
            projected = jnp.maximum(self.eps, projected)
            sum_projected = jnp.sum(projected)
            projected = jnp.where(
                sum_projected > self.eps,
                projected / sum_projected,
                jnp.ones_like(projected) / projected.shape[0],
            )

            return projected

        projected_archetypes = jax.vmap(_project_to_boundary)(archetypes)

        return jnp.asarray(projected_archetypes)

    @partial(jax.jit, static_argnums=(0,))
    def project_col_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Project column archetypes to the boundary of the feature space.

        This critical counterpart to row archetype projection ensures column archetypes represent distinct feature patterns.
        While conceptually similar to row projection, this operation works in the transposed space,
        treating features as observations and finding extremes among them.

        Without this specialized projection, the feature archetypes would not capture meaningful feature combinations,
        undermining the dual-perspective advantage of biarchetypal analysis.

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
        X_T = X.T

        centroid = jnp.mean(X_T, axis=0)

        def _project_feature_to_boundary(archetype: jnp.ndarray) -> jnp.ndarray:
            """Project a single column archetype to the boundary of the feature convex hull."""
            # Step 1: Calculate direction in sample space using weighted features
            weighted_features = archetype[:, jnp.newaxis] * X_T
            direction = jnp.sum(weighted_features, axis=0) - centroid
            direction_norm = jnp.linalg.norm(direction)
            normalized_direction = jnp.where(
                direction_norm > self.eps,
                direction / direction_norm,
                jax.random.normal(jax.random.PRNGKey(0), direction.shape) / jnp.sqrt(direction.shape[0]),
            )

            # Step 2: Project all features onto this direction to measure extremeness
            projections = jnp.dot(X_T, normalized_direction)

            # Step 3: Find multiple extreme features with adaptive k selection
            k = min(5, X.shape[1] // 10 + 2)
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
            alpha = 0.8
            projected = alpha * multi_hot + (1 - alpha) * archetype

            # Step 7: Apply simplex constraints with numerical stability safeguards
            projected = jnp.maximum(self.eps, projected)
            sum_projected = jnp.sum(projected)
            projected = jnp.where(
                sum_projected > self.eps,
                projected / sum_projected,
                jnp.ones_like(projected) / projected.shape[0],
            )

            return projected

        projected_archetypes = jax.vmap(_project_feature_to_boundary)(archetypes.T)

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
            normalize: Whether to normalize features - essential for data with different scales
            **kwargs: Additional parameters for customizing the fitting process

        Returns:
            Self - fitted model instance with discovered biarchetypes
        """
        X_np = X.values if hasattr(X, "values") else X
        self.X_mean = np.mean(X_np, axis=0)
        self.X_std = np.std(X_np, axis=0)

        if self.X_std is not None:
            self.X_std = np.where(self.X_std < self.eps, np.ones_like(self.X_std), self.X_std)

        if normalize:
            X_scaled = (X_np - self.X_mean) / self.X_std
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            X_scaled = X_np.copy()

        X_jax = jnp.array(X_scaled, dtype=jnp.float32)
        n_samples, n_features = X_jax.shape

        self.logger.info(f"Data shape: {X_jax.shape}")
        self.logger.info(f"Data range: min={float(jnp.min(X_jax)):.4f}, max={float(jnp.max(X_jax)):.4f}")
        self.logger.info(f"Row archetypes: {self.n_row_archetypes}")
        self.logger.info(f"Column archetypes: {self.n_col_archetypes}")

        self.rng_key, subkey = jax.random.split(self.rng_key)
        alpha_init = jax.random.uniform(
            subkey, (n_samples, self.n_row_archetypes), minval=0.1, maxval=0.9, dtype=jnp.float32
        )
        alpha_init = self.project_row_coefficients(alpha_init)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        gamma_init = jax.random.uniform(
            subkey, (self.n_col_archetypes, n_features), minval=0.1, maxval=0.9, dtype=jnp.float32
        )
        gamma_init = self.project_col_coefficients(gamma_init)

        # Initialize beta (row archetypes) using sophisticated k-means++ initialization
        # This approach ensures diverse starting points that are well-distributed across the data space
        self.rng_key, subkey = jax.random.split(self.rng_key)

        first_idx = jax.random.randint(subkey, (), 0, n_samples)
        selected_indices = jnp.zeros(self.n_row_archetypes, dtype=jnp.int32)
        selected_indices = selected_indices.at[0].set(first_idx)

        for i in range(1, self.n_row_archetypes):
            min_dists = jnp.ones(n_samples) * float("inf")

            for j in range(i):
                idx = selected_indices[j]
                dists = jnp.sum((X_jax - X_jax[idx]) ** 2, axis=1)
                min_dists = jnp.minimum(min_dists, dists)

            for j in range(i):
                idx = selected_indices[j]
                min_dists = min_dists.at[idx].set(0.0)

            self.rng_key, subkey = jax.random.split(self.rng_key)
            probs = min_dists / (jnp.sum(min_dists) + self.eps)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)
            selected_indices = selected_indices.at[i].set(next_idx)

        beta_init = jnp.zeros((self.n_row_archetypes, n_samples), dtype=jnp.float32)
        for i in range(self.n_row_archetypes):
            idx = selected_indices[i]
            beta_init = beta_init.at[i, idx].set(1.0)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        noise = jax.random.uniform(subkey, beta_init.shape, minval=0.0, maxval=0.05, dtype=jnp.float32)
        beta_init = beta_init + noise
        beta_init = beta_init / jnp.sum(beta_init, axis=1, keepdims=True)

        self.logger.info("Row archetypes initialized with k-means++ strategy")

        # Initialize theta (column archetypes) with advanced diversity-maximizing approach
        # This ensures column archetypes capture the most distinctive feature patterns
        self.rng_key, subkey = jax.random.split(self.rng_key)

        X_T = X_jax.T
        theta_init = jnp.zeros((n_features, self.n_col_archetypes), dtype=jnp.float32)
        feature_variance = jnp.var(X_T, axis=1)
        selected_features = jnp.zeros(self.n_col_archetypes, dtype=jnp.int32)

        probs = feature_variance / (jnp.sum(feature_variance) + self.eps)
        first_idx = jax.random.choice(subkey, n_features, p=probs)
        selected_features = selected_features.at[0].set(first_idx)
        theta_init = theta_init.at[first_idx, 0].set(1.0)

        for i in range(1, self.n_col_archetypes):
            min_dists = jnp.ones(n_features) * float("inf")

            for j in range(i):
                idx = selected_features[j]
                corr = jnp.abs(jnp.sum(X_T * X_T[idx, jnp.newaxis], axis=1)) / (
                    jnp.sqrt(jnp.sum(X_T**2, axis=1) * jnp.sum(X_T[idx] ** 2) + self.eps)
                )
                dists = 1.0 - corr
                min_dists = jnp.minimum(min_dists, dists)

            for j in range(i):
                idx = selected_features[j]
                min_dists = min_dists.at[idx].set(0.0)

            next_idx = jnp.argmax(min_dists)
            selected_features = selected_features.at[i].set(next_idx)
            theta_init = theta_init.at[next_idx, i].set(1.0)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        noise = jax.random.uniform(subkey, theta_init.shape, minval=0.0, maxval=0.05, dtype=jnp.float32)
        theta_init = theta_init + noise
        theta_init = theta_init / jnp.sum(theta_init, axis=0, keepdims=True)

        self.logger.info("Column archetypes initialized with diversity-maximizing strategy")

        # Set up optimizer with learning rate schedule for better convergence
        # We use a sophisticated learning rate schedule with warmup and decay phases
        warmup_steps = 20
        decay_steps = 100
        reduced_lr = self.learning_rate * 0.05
        warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=reduced_lr, transition_steps=warmup_steps)
        decay_schedule = optax.exponential_decay(
            init_value=reduced_lr,
            transition_steps=decay_steps,
            decay_rate=0.95,
            end_value=0.000001,
            staircase=False,
        )
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

        params = {"alpha": alpha_init, "beta": beta_init, "theta": theta_init, "gamma": gamma_init}
        opt_state = optimizer.init(params)

        @partial(jax.jit, static_argnums=(3,))
        def update_step(
            params: dict[str, jnp.ndarray], opt_state: optax.OptState, X: jnp.ndarray, iteration: int
        ) -> tuple[dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
            """Execute a single optimization step."""

            def loss_fn(params):
                return self.loss_function(params, X)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)

            new_params = optax.apply_updates(params, updates)
            new_params["alpha"] = self.project_row_coefficients(new_params["alpha"])
            new_params["gamma"] = self.project_col_coefficients(new_params["gamma"])

            def project():
                return {
                    "alpha": new_params["alpha"],
                    "beta": self.project_row_archetypes(new_params["beta"], X),
                    "theta": self.project_col_archetypes(new_params["theta"], X),
                    "gamma": new_params["gamma"],
                }

            def no_project():
                return new_params

            do_project = jnp.mod(iteration, 10) == 0
            new_params = jax.lax.cond(do_project, lambda: project(), lambda: no_project())

            return new_params, opt_state, loss

        prev_loss = float("inf")
        initial_loss = float(self.loss_function(params, X_jax))
        self.loss_history = []
        self.logger.info(f"Initial loss: {initial_loss:.6f}")

        for it in range(self.max_iter):
            try:
                params, opt_state, loss = update_step(params, opt_state, X_jax, it)
                loss_value = float(loss)
                relative_improvement = (prev_loss - loss_value) / (prev_loss + 1e-10)

                if jnp.isnan(loss_value):
                    self.logger.warning(get_message("warning", "nan_detected", iteration=it))
                    break

                self.loss_history.append(loss_value)
                self.logger.info(f"Iteration {it}, Loss: {loss_value:.6f}")

                if it >= 10:
                    recent_losses = self.loss_history[-10:]
                    loss_ma = sum(recent_losses) / 10
                    if "loss_ma_prev" not in locals():
                        loss_ma_prev = self.loss_history[0]
                    relative_ma_improvement = (loss_ma_prev - loss_ma) / (loss_ma_prev + 1e-10)
                    loss_ma_prev = loss_ma
                else:
                    relative_ma_improvement = relative_improvement
                    loss_ma_prev = prev_loss

                if it > 20 and relative_improvement < self.tol and relative_ma_improvement < self.tol:
                    self.logger.info(f"Converged at iteration {it}")
                    break

                prev_loss = loss_value

                if it % 25 == 0 or it < 5:
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

                if it % 100 == 0 and it > 0:
                    alpha_sparsity = jnp.mean(jnp.sum(params["alpha"] > 0.01, axis=1) / params["alpha"].shape[1])
                    gamma_sparsity = jnp.mean(jnp.sum(params["gamma"] > 0.01, axis=0) / params["gamma"].shape[0])
                    self.logger.debug(
                        f"  - Alpha sparsity: {float(alpha_sparsity):.4f} | Gamma sparsity: {float(gamma_sparsity):.4f}"
                    )
                    self.logger.debug(f"  - Learning rate: {float(schedule(it)):.8f}")

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

        params["beta"] = self.project_row_archetypes(jnp.asarray(params["beta"]), X_jax)
        params["theta"] = self.project_col_archetypes(jnp.asarray(params["theta"]), X_jax)

        self.alpha = np.array(params["alpha"])
        self.beta = np.array(params["beta"])
        self.theta = np.array(params["theta"])
        self.gamma = np.array(params["gamma"])

        self.biarchetypes = np.array(np.matmul(np.matmul(self.beta, np.asanyarray(X_jax)), self.theta))
        self.archetypes = np.array(np.matmul(self.beta, X_jax))
        self.weights = np.array(self.alpha)

        if len(self.loss_history) > 0:
            self.logger.info(f"Final loss: {self.loss_history[-1]:.6f}")
        else:
            self.logger.warning("No valid loss was recorded")

        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> Any:
        """Transform new data into dual-directional archetype space.

        This method computes optimal weights to represent new data in terms of discovered archetypes,
        enabling consistent interpretation of new observations within the established biarchetypal framework.

        Unlike conventional AA, this transform operates in both row and column spaces,
        providing a holistic representation of new data that preserves the model's dual-perspective advantage.
        The implementation efficiently leverages pre-trained biarchetypes to avoid redundant computation.

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

        if self.X_mean is not None and self.X_std is not None:
            X_scaled = (X_jax - self.X_mean) / self.X_std
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            X_scaled = X_jax.copy()

        biarchetypes_jax = jnp.array(self.biarchetypes, dtype=jnp.float32)

        def optimize_row_weights(x_row):
            alpha = jnp.ones(self.n_row_archetypes) / self.n_row_archetypes

            for _ in range(100):
                reconstruction = jnp.matmul(jnp.matmul(alpha, biarchetypes_jax), jnp.asarray(self.gamma))
                error = x_row - reconstruction
                grad = -2 * jnp.matmul(error, jnp.matmul(biarchetypes_jax, jnp.asarray(self.gamma)).T)
                alpha = alpha - 0.01 * grad
                alpha = jnp.maximum(1e-10, alpha)
                alpha = alpha / jnp.sum(alpha)

            return alpha

        alpha_new = jnp.array([optimize_row_weights(x) for x in X_scaled])
        gamma_new = self.gamma.copy()

        return (np.asarray(alpha_new), np.asarray(gamma_new))

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, normalize: bool = False, **kwargs) -> Any:
        """Fit the model and transform the data in a single operation.

        This convenience method combines model fitting and data transformation in a single step,
        offering two key advantages:
        1. Computational efficiency by avoiding redundant calculations
        2. Simplified workflow for immediate biarchetypal representation

        The method is particularly valuable when the biarchetypal representation is needed immediately after fitting,
        such as in analysis pipelines or when integrating with scikit-learn compatible frameworks.

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

        This method provides the inverse operation of transform(),
        reconstructing data points from their biarchetypal weights.
        This capability serves several critical purposes:
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
        if self.biarchetypes is None or self.alpha is None or self.gamma is None:
            raise ValueError("Model must be fitted before reconstruction")

        if X is None:
            alpha, gamma = self.alpha, self.gamma
        else:
            alpha, gamma = self.transform(X)

        reconstructed = np.matmul(np.matmul(alpha, self.biarchetypes), gamma)
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

        Row archetypes represent extreme patterns in observation space,
        describing distinctive types of data points.
        These archetypes are essential for understanding the primary modes of variation among observations
        and provide the foundation for interpreting data point weights.

        In the biarchetypal model, row archetypes are projections of the data matrix via the beta coefficients (β·X).

        Returns:
            Row archetypes matrix (n_row_archetypes, n_features)
        """
        if self.archetypes is None:
            raise ValueError("Model must be fitted before getting row archetypes")

        return self.archetypes

    def get_col_archetypes(self) -> np.ndarray:
        """Retrieve the column archetypes.

        Column archetypes represent extreme patterns in feature space,
        describing distinctive feature combinations or "feature types."

        This perspective is unique to biarchetypal analysis and provides crucial insights about feature
        relationships that would be missed in standard archetypal analysis.

        These archetypes enable feature-level interpretations
        and can reveal coordinated feature behaviors across different data contexts.

        Returns:
            Column archetypes matrix (n_col_archetypes, n_features)
        """
        if self.theta is None or self.gamma is None:
            raise ValueError("Model must be fitted before getting column archetypes")

        if self.theta.shape[0] == self.theta.shape[1]:
            return np.eye(self.theta.shape[0])

        else:
            col_archetypes = np.zeros((self.n_col_archetypes, self.theta.shape[0]))

            for i in range(min(self.n_col_archetypes, self.theta.shape[0])):
                col_archetypes[i, i] = 1.0

            return col_archetypes

    def get_row_weights(self) -> np.ndarray:
        """Retrieve the row coefficients (alpha).

        Row weights represent how each data point is composed as a mixture of row archetypes.
        These weights are essential for:
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

        This convenience method provides access to both directions of archetypal analysis simultaneously,
        facilitating comprehensive analysis and visualization of the dual-perspective patterns.
        Accessing both archetypes together is particularly valuable for cross-modal analysis examining relationships
        between observation patterns and feature patterns.

        Returns:
            Tuple of (row_archetypes, column_archetypes) matrices
        """
        return self.get_row_archetypes(), self.get_col_archetypes()

    def get_all_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve both row and column weights in a single call.

        This convenience method provides access to all weight coefficients simultaneously,
        enabling comprehensive analysis of how observations and features relate to their respective archetypes.
        This unified view is particularly valuable for understanding the full biarchetypal
        decomposition and how information flows between the row and column spaces.

        Returns:
            Tuple of (row_weights, column_weights) matrices
        """
        return self.get_row_weights(), self.get_col_weights()
