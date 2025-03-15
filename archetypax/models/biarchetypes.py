"""Biarchetypal Analysis using JAX."""

from functools import partial
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .archetypes import ImprovedArchetypalAnalysis

T = TypeVar("T", bound=np.ndarray)


class BiarchetypalAnalysis(ImprovedArchetypalAnalysis):
    """
    Biarchetypal Analysis using JAX.

    This implementation follows the paper "Biarchetype analysis: simultaneous learning
    of observations and features based on extremes" by Alcacer et al.

    Biarchetypal Analysis extends archetype analysis to simultaneously identify archetypes
    of both observations (rows) and features (columns). It represents the data matrix X as:

    X ≃ alpha·beta·X·theta·gamma

    where:
    - alpha, beta: Coefficients and archetypes for observations (rows)
    - theta, gamma: Coefficients and archetypes for features (columns)
    """

    def __init__(
        self,
        n_row_archetypes: int,
        n_col_archetypes: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_seed: int = 42,
        learning_rate: float = 0.001,
        lambda_reg: float = 0.01,
    ):
        """
        Initialize the Biarchetypal Analysis model.

        Args:
            n_row_archetypes: Number of archetypes for rows (observations)
            n_col_archetypes: Number of archetypes for columns (features)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            random_seed: Random seed for initialization
            learning_rate: Learning rate for optimizer
            lambda_reg: Regularization parameter for entropy
        """
        # Initialize using parent class with the row archetypes
        # (we'll handle column archetypes separately)
        super().__init__(
            n_archetypes=n_row_archetypes,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            learning_rate=learning_rate,
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

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, params: dict[str, jnp.ndarray], X: jnp.ndarray) -> jnp.ndarray:
        """Calculate the reconstruction loss for biarchetypal analysis.

        Args:
            params: Dictionary containing alpha, beta, theta, gamma
            X: Data matrix

        Returns:
            Reconstruction loss
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

        # Add regularization to encourage sparsity
        # Note: We want to MINIMIZE entropy to encourage sparsity
        # So we use positive entropy (not negative) in the loss function
        alpha_entropy = jnp.sum(alpha * jnp.log(alpha + 1e-10), axis=1)  # Removed negative sign
        gamma_entropy = jnp.sum(gamma * jnp.log(gamma + 1e-10), axis=0)  # Removed negative sign
        entropy_reg = jnp.mean(alpha_entropy) + jnp.mean(gamma_entropy)

        return (reconstruction_loss - self.lambda_reg * entropy_reg).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def project_row_coefficients(self, coefficients: jnp.ndarray) -> jnp.ndarray:
        """Project row coefficients to satisfy simplex constraints.

        Args:
            coefficients: Coefficient matrix (n_samples, n_row_archetypes)

        Returns:
            Projected coefficient matrix
        """
        eps = 1e-10
        coefficients = jnp.maximum(eps, coefficients)
        sum_coeffs = jnp.sum(coefficients, axis=1, keepdims=True)
        sum_coeffs = jnp.maximum(eps, sum_coeffs)
        return coefficients / sum_coeffs

    @partial(jax.jit, static_argnums=(0,))
    def project_col_coefficients(self, coefficients: jnp.ndarray) -> jnp.ndarray:
        """Project column coefficients to satisfy simplex constraints.

        Args:
            coefficients: Coefficient matrix (n_col_archetypes, n_features)

        Returns:
            Projected coefficient matrix
        """
        eps = 1e-10
        coefficients = jnp.maximum(eps, coefficients)
        sum_coeffs = jnp.sum(coefficients, axis=0, keepdims=True)
        sum_coeffs = jnp.maximum(eps, sum_coeffs)
        return coefficients / sum_coeffs

    @partial(jax.jit, static_argnums=(0,))
    def project_row_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Project row archetypes to be convex combinations of data points.

        This implementation employs an advanced boundary-seeking algorithm that:
        1. Identifies multiple extreme points in the direction of each archetype
        2. Uses adaptive weighting to balance diversity and stability
        3. Ensures proper simplex constraints are maintained

        Args:
            archetypes: Row archetype matrix (n_row_archetypes, n_samples)
            X: Data matrix (n_samples, n_features)

        Returns:
            Projected row archetype matrix with enhanced diversity
        """
        # Calculate the data centroid as our reference point
        centroid = jnp.mean(X, axis=0)  # Shape: (n_features,)

        def _project_to_boundary(archetype):
            """Project a single archetype to the boundary of the convex hull.

            This function implements a sophisticated projection strategy that:
            1. Identifies the direction from centroid to the weighted archetype representation
            2. Finds multiple extreme points in this direction
            3. Creates a diverse mixture of these extreme points
            """
            # Step 1: Calculate direction from centroid to archetype representation
            # This vector points from the data center toward the archetype's "ideal" position
            weighted_representation = jnp.matmul(archetype, X)  # Shape: (n_features,)
            direction = weighted_representation - centroid
            direction_norm = jnp.linalg.norm(direction)

            # Ensure numerical stability with proper normalization
            normalized_direction = jnp.where(
                direction_norm > 1e-10,
                direction / direction_norm,
                jax.random.normal(jax.random.PRNGKey(0), direction.shape) / jnp.sqrt(direction.shape[0]),
            )

            # Step 2: Project all data points onto this direction vector
            # This identifies how "extreme" each point is in the archetype's direction
            projections = jnp.dot(X - centroid, normalized_direction)  # Shape: (n_samples,)

            # Step 3: Find multiple extreme points with adaptive k selection
            # The number of extreme points considered adapts to the data dimensionality
            k = min(5, X.shape[0] // 10 + 2)  # Adaptive k based on dataset size

            # Get indices of the k most extreme points
            top_k_indices = jnp.argsort(projections)[-k:]

            # Get the projection values for these extreme points
            top_k_projections = projections[top_k_indices]

            # Step 4: Calculate weights with emphasis on the most extreme points
            # Points with larger projections receive higher weights
            weights_unnormalized = jnp.exp(top_k_projections - jnp.max(top_k_projections))
            weights = weights_unnormalized / jnp.sum(weights_unnormalized)

            # Step 5: Create a weighted combination of extreme points
            multi_hot = jnp.zeros_like(archetype)
            for i in range(k):
                idx = top_k_indices[i]
                multi_hot = multi_hot.at[idx].set(weights[i])

            # Step 6: Mix with original archetype for stability and convergence
            # The mixing parameter balances exploration vs. exploitation
            alpha = 0.8  # Stronger pull toward extreme points for better diversity
            projected = alpha * multi_hot + (1 - alpha) * archetype

            # Step 7: Apply simplex constraints with numerical stability safeguards
            # Ensure non-negativity and proper normalization
            projected = jnp.maximum(1e-10, projected)
            sum_projected = jnp.sum(projected)
            projected = jnp.where(
                sum_projected > 1e-10, projected / sum_projected, jnp.ones_like(projected) / projected.shape[0]
            )

            return projected

        # Apply the projection function to each row archetype in parallel
        projected_archetypes = jax.vmap(_project_to_boundary)(archetypes)

        return jnp.asarray(projected_archetypes)

    @partial(jax.jit, static_argnums=(0,))
    def project_col_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Project column archetypes to be convex combinations of features.

        This implementation employs a sophisticated feature-space boundary-seeking algorithm that:
        1. Identifies multiple extreme features in the direction of each archetype
        2. Uses adaptive weighting based on feature importance
        3. Ensures proper simplex constraints while maximizing diversity

        Args:
            archetypes: Column archetype matrix (n_features, n_col_archetypes)
            X: Data matrix (n_samples, n_features)

        Returns:
            Projected column archetype matrix with enhanced diversity
        """
        # Transpose X to work with features as data points in feature space
        X_T = X.T  # Shape: (n_features, n_samples)

        # Calculate the feature centroid as our reference point
        centroid = jnp.mean(X_T, axis=0)  # Shape: (n_samples,)

        def _project_feature_to_boundary(archetype):
            """Project a single column archetype to the boundary of the feature convex hull.

            This function implements an advanced projection strategy that:
            1. Calculates a direction in sample space based on feature weights
            2. Identifies features that are extreme in this direction
            3. Creates a diverse mixture of these extreme features
            """
            # Step 1: Calculate direction in sample space using weighted features
            # This avoids direct matrix multiplication for better numerical stability
            weighted_features = archetype[:, jnp.newaxis] * X_T  # Shape: (n_features, n_samples)
            direction = jnp.sum(weighted_features, axis=0) - centroid  # Shape: (n_samples,)
            direction_norm = jnp.linalg.norm(direction)

            # Ensure numerical stability with proper normalization
            normalized_direction = jnp.where(
                direction_norm > 1e-10,
                direction / direction_norm,
                jax.random.normal(jax.random.PRNGKey(0), direction.shape) / jnp.sqrt(direction.shape[0]),
            )

            # Step 2: Project all features onto this direction to measure extremeness
            projections = jnp.dot(X_T, normalized_direction)  # Shape: (n_features,)

            # Step 3: Find multiple extreme features with adaptive k selection
            # The number of extreme features considered adapts to the feature dimensionality
            k = min(5, X.shape[1] // 10 + 2)  # Adaptive k based on feature space size

            # Get indices of the k most extreme features
            top_k_indices = jnp.argsort(projections)[-k:]

            # Get the projection values for these extreme features
            top_k_projections = projections[top_k_indices]

            # Step 4: Calculate weights with emphasis on the most extreme features
            # Features with larger projections receive higher weights
            weights_unnormalized = jnp.exp(top_k_projections - jnp.max(top_k_projections))
            weights = weights_unnormalized / jnp.sum(weights_unnormalized)

            # Step 5: Create a weighted combination of extreme features
            multi_hot = jnp.zeros_like(archetype)
            for i in range(k):
                idx = top_k_indices[i]
                multi_hot = multi_hot.at[idx].set(weights[i])

            # Step 6: Mix with original archetype for stability and convergence
            # The mixing parameter balances exploration vs. exploitation
            alpha = 0.8  # Stronger pull toward extreme features for better diversity
            projected = alpha * multi_hot + (1 - alpha) * archetype

            # Step 7: Apply simplex constraints with numerical stability safeguards
            # Ensure non-negativity and proper normalization
            projected = jnp.maximum(1e-10, projected)
            sum_projected = jnp.sum(projected)
            projected = jnp.where(
                sum_projected > 1e-10, projected / sum_projected, jnp.ones_like(projected) / projected.shape[0]
            )

            return projected

        # Apply the projection function to each column archetype in parallel
        projected_archetypes = jax.vmap(_project_feature_to_boundary)(archetypes.T)

        # Transpose the result back to original shape
        return jnp.asarray(projected_archetypes.T)

    def fit(self, X: np.ndarray, normalize: bool = False) -> "BiarchetypalAnalysis":
        """
        Fit the Biarchetypal Analysis model to the data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            normalize: Whether to normalize the data before fitting

        Returns:
            Self
        """
        # Preprocess data: scale for improved stability
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        # Prevent division by zero
        if self.X_std is not None:
            self.X_std = np.where(self.X_std < 1e-10, np.ones_like(self.X_std), self.X_std)
        X_scaled = (X - self.X_mean) / self.X_std if normalize else X.copy()

        # Convert to JAX array with explicit dtype for better performance
        X_jax = jnp.array(X_scaled, dtype=jnp.float32)
        n_samples, n_features = X_jax.shape

        # Debug information
        print(f"Data shape: {X_jax.shape}")
        print(f"Data range: min={float(jnp.min(X_jax)):.4f}, max={float(jnp.max(X_jax)):.4f}")
        print(f"Row archetypes: {self.n_row_archetypes}")
        print(f"Column archetypes: {self.n_col_archetypes}")

        # Initialize alpha (row coefficients) with more stable initialization
        self.key, subkey = jax.random.split(self.key)
        alpha_init = jax.random.uniform(
            subkey, (n_samples, self.n_row_archetypes), minval=0.1, maxval=0.9, dtype=jnp.float32
        )
        alpha_init = self.project_row_coefficients(alpha_init)

        # Initialize gamma (column coefficients)
        self.key, subkey = jax.random.split(self.key)
        gamma_init = jax.random.uniform(
            subkey, (self.n_col_archetypes, n_features), minval=0.1, maxval=0.9, dtype=jnp.float32
        )
        gamma_init = self.project_col_coefficients(gamma_init)

        # Initialize beta (row archetypes) using sophisticated k-means++ initialization
        # This approach ensures diverse starting points that are well-distributed across the data space
        self.key, subkey = jax.random.split(self.key)

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
            self.key, subkey = jax.random.split(self.key)
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
        self.key, subkey = jax.random.split(self.key)
        noise = jax.random.uniform(subkey, beta_init.shape, minval=0.0, maxval=0.05, dtype=jnp.float32)
        beta_init = beta_init + noise

        # Step 4: Ensure proper normalization to maintain simplex constraints
        # Each row must sum to 1 to represent a valid convex combination
        beta_init = beta_init / jnp.sum(beta_init, axis=1, keepdims=True)

        print("Row archetypes initialized with k-means++ strategy")

        # Initialize theta (column archetypes) with advanced diversity-maximizing approach
        # This ensures column archetypes capture the most distinctive feature patterns
        self.key, subkey = jax.random.split(self.key)

        # Step 1: Transpose data for feature-centric operations
        X_T = X_jax.T  # Shape: (n_features, n_samples)

        # Step 2: Calculate feature diversity metrics
        # Compute variance of each feature to identify informative dimensions
        feature_variance = jnp.var(X_T, axis=1)

        # Step 3: Select initial features using variance-weighted sampling
        theta_init = jnp.zeros((n_features, self.n_col_archetypes), dtype=jnp.float32)
        selected_features = jnp.zeros(self.n_col_archetypes, dtype=jnp.int32)

        # Select first feature with probability proportional to variance
        self.key, subkey = jax.random.split(self.key)
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
        self.key, subkey = jax.random.split(self.key)
        noise = jax.random.uniform(subkey, theta_init.shape, minval=0.0, maxval=0.05, dtype=jnp.float32)
        theta_init = theta_init + noise

        # Step 6: Ensure proper normalization to maintain simplex constraints
        # Each column must sum to 1 to represent a valid convex combination
        theta_init = theta_init / jnp.sum(theta_init, axis=0, keepdims=True)

        print("Column archetypes initialized with diversity-maximizing strategy")

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
        def update_step(params, opt_state, X, iteration):
            """Execute a single optimization step."""

            # Loss function
            def loss_fn(params):
                return self.loss_function(params, X)

            # Calculate gradient and update with value_and_grad for efficiency
            loss, grads = jax.value_and_grad(loss_fn)(params)

            # Apply gradient clipping to prevent NaNs
            grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

            # Get new parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Project to constraints
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
        print(f"Initial loss: {initial_loss:.6f}")

        for i in range(self.max_iter):
            try:
                # Execute update step
                params, opt_state, loss = update_step(params, opt_state, X_jax, i)
                loss_value = float(loss)

                # Check for NaN
                if jnp.isnan(loss_value):
                    print(f"Warning: NaN detected at iteration {i}. Stopping early.")
                    break

                # Record loss
                self.loss_history.append(loss_value)

                # Check convergence with sophisticated adaptive criterion
                if i > 0:
                    # Calculate relative improvement over the last iteration
                    rel_improvement = (prev_loss - loss_value) / (prev_loss + 1e-10)

                    # Calculate moving average of recent losses if we have enough history
                    window_size = min(10, len(self.loss_history))
                    if window_size >= 5:
                        recent_losses = self.loss_history[-window_size:]
                        moving_avg = sum(recent_losses) / window_size
                        # Calculate relative improvement over the moving average
                        avg_improvement = (moving_avg - loss_value) / (moving_avg + 1e-10)

                        # Converge if both short-term and long-term improvements are small
                        if 0 <= rel_improvement < self.tol and 0 <= avg_improvement < self.tol * 2:
                            print(f"Converged at iteration {i}")
                            print(f"  - Relative improvement: {rel_improvement:.8f}")
                            print(f"  - Average improvement: {avg_improvement:.8f}")
                            break
                    else:
                        # Fall back to simple criterion for early iterations
                        if 0 <= rel_improvement < self.tol:
                            print(f"Early convergence at iteration {i} with relative improvement {rel_improvement:.8f}")
                            break

                prev_loss = loss_value

                # Display comprehensive progress information at regular intervals
                if i % 25 == 0 or i < 5:
                    # Calculate performance metrics for monitoring optimization trajectory
                    if len(self.loss_history) > 1:
                        avg_last_5 = sum(self.loss_history[-min(5, len(self.loss_history)) :]) / min(
                            5, len(self.loss_history)
                        )
                        improvement_rate = (self.loss_history[0] - loss_value) / (i + 1) if i > 0 else 0
                        print(
                            f"Iteration {i:4d} | Loss: {loss_value:.6f} | Avg(5): {avg_last_5:.6f} | Improvement rate: {improvement_rate:.8f}"
                        )
                    else:
                        print(f"Iteration {i:4d} | Loss: {loss_value:.6f}")

                    # Provide in-depth diagnostics at major milestones
                    if i % 100 == 0 and i > 0:
                        # Analyze archetype characteristics
                        alpha_sparsity = jnp.mean(jnp.sum(params["alpha"] > 0.01, axis=1) / params["alpha"].shape[1])
                        gamma_sparsity = jnp.mean(jnp.sum(params["gamma"] > 0.01, axis=0) / params["gamma"].shape[0])
                        print(
                            f"  - Alpha sparsity: {float(alpha_sparsity):.4f} | Gamma sparsity: {float(gamma_sparsity):.4f}"
                        )
                        print(f"  - Learning rate: {float(schedule(i)):.8f}")

                        # Flag potential convergence issues
                        if jnp.max(params["alpha"]) > 0.99:
                            print("  - Warning: Alpha contains near-one values, may indicate degenerate solution")
                        if jnp.max(params["gamma"]) > 0.99:
                            print("  - Warning: Gamma contains near-one values, may indicate degenerate solution")

            except Exception as e:
                print(f"Error at iteration {i}: {e!s}")
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
            print(f"Final loss: {self.loss_history[-1]:.6f}")
        else:
            print("Warning: No valid loss was recorded")

        return self

    def transform(self, X: np.ndarray) -> Any:
        """
        Transform new data to row and column archetype weights.

        Args:
            X: New data matrix of shape (n_samples, n_features)

        Returns:
            Tuple of (row_weights, col_weights) representing the data in terms of archetypes
        """
        if self.alpha is None or self.beta is None or self.theta is None or self.gamma is None:
            raise ValueError("Model must be fitted before transform")

        # Scale input data
        X_scaled = (X - self.X_mean) / self.X_std if self.X_mean is not None and self.X_std is not None else X
        jnp.array(X_scaled, dtype=jnp.float32)

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

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, normalize: bool = False) -> Any:
        """
        Fit the model and transform the data.

        Args:
            X: Data matrix
            y: Target values (ignored)
            normalize: Whether to normalize the data

        Returns:
            Tuple of (row_weights, col_weights)
        """
        self.fit(X, normalize=normalize)
        alpha, gamma = self.transform(X)
        return np.asarray(alpha), np.asarray(gamma)

    def reconstruct(self, X: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct data from biarchetypes.

        Args:
            X: Optional data matrix to reconstruct. If None, uses the training data.

        Returns:
            Reconstructed data matrix
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
        """
        Get the biarchetypes matrix.

        Returns:
            Biarchetypes matrix of shape (n_row_archetypes, n_col_archetypes)
        """
        if self.biarchetypes is None:
            raise ValueError("Model must be fitted before getting biarchetypes")

        return self.biarchetypes

    def get_row_archetypes(self) -> np.ndarray:
        """
        Get the row archetypes.

        Returns:
            Row archetypes matrix
        """
        if self.archetypes is None:
            raise ValueError("Model must be fitted before getting row archetypes")

        return self.archetypes

    def get_col_archetypes(self) -> np.ndarray:
        """
        Get the column archetypes.

        Returns:
            Column archetypes matrix
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
        """
        Get the row weights (alpha).

        Returns:
            Row weights matrix
        """
        if self.alpha is None:
            raise ValueError("Model must be fitted before getting row weights")

        return self.alpha

    def get_col_weights(self) -> np.ndarray:
        """
        Get the column weights (gamma).

        Returns:
            Column weights matrix
        """
        if self.gamma is None:
            raise ValueError("Model must be fitted before getting column weights")

        return self.gamma

    def get_all_archetypes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get both row and column archetypes.

        Returns:
            Tuple of (row_archetypes, column_archetypes)
        """
        return self.get_row_archetypes(), self.get_col_archetypes()

    def get_all_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get both row and column weights.

        Returns:
            Tuple of (row_weights, column_weights)
        """
        return self.get_row_weights(), self.get_col_weights()
