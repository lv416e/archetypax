"""Improved Archetypal Analysis model using JAX."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from archetypax.logger import get_logger, get_message

from .base import ArchetypalAnalysis


class ImprovedArchetypalAnalysis(ArchetypalAnalysis):
    """Improved Archetypal Analysis model using JAX."""

    def __init__(
        self,
        n_archetypes: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_seed: int = 42,
        learning_rate: float = 0.001,
        lambda_reg: float = 0.01,
        normalize: bool = False,
        projection_method: str = "cbap",
        projection_alpha: float = 0.1,
        archetype_init_method: str = "directional",
        **kwargs,
    ):
        """Initialize the Improved Archetypal Analysis model.

        Args:
            n_archetypes: Number of archetypes to extract
            max_iter: Maximum number of iterations for optimization
            tol: Convergence tolerance
            random_seed: Random seed for reproducibility
            learning_rate: Learning rate for the optimizer
            lambda_reg: Regularization strength
            normalize: Whether to normalize data before fitting
            projection_method: Method for projecting archetypes ("cbap", "convex_hull", or "knn")
            projection_alpha: Strength of projection (0-1)
            archetype_init_method: Method for initializing archetypes
                ("directional", "qhull", "kmeans_pp")
            **kwargs: Additional keyword arguments

        Note:
            The improved model uses advanced initialization strategies, robust optimization,
            and incorporates multiple projection techniques to produce better archetypes.
        """
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            learning_rate=learning_rate,
            **kwargs,
        )

        # Initialize class-specific logger
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            get_message(
                "init",
                "model_init",
                model_name=self.__class__.__name__,
                n_archetypes=n_archetypes,
            )
        )

        # Ensure JAX RNG key is properly initialized from random seed
        self.rng_key = jax.random.key(random_seed)

        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.normalize = normalize
        self.projection_method = (
            "default" if projection_method == "cbap" or projection_method == "default" else projection_method
        )
        self.projection_alpha = projection_alpha
        self.archetype_init_method = archetype_init_method
        self.early_stopping_patience = min(kwargs.get("early_stopping_patience", 100), 100)
        # Initialize tracking
        self.archetype_history: list[np.ndarray] = []
        self.loss_history: list[float] = []
        self.optimizer: optax.GradientTransformation = optax.adam(learning_rate=self.learning_rate)
        # Specific settings for archetype updates
        self.archetype_grad_scale = 1.0  # Gradient scale for archetypes (reduced from 1.5 to 1.0)
        self.noise_scale = 0.02  # Magnitude of initial noise (reduced from 0.03)
        self.exploration_noise_scale = 0.05  # Magnitude of exploration noise (reduced from 0.07)

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Transform data by calculating the weights matrix.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            y: Ignored. Present for scikit-learn API compatibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Weight matrix of shape (n_samples, n_archetypes).
        """
        # Preprocess data: apply same normalization as during fit
        if self.X_mean is not None and self.X_std is not None:
            X_scaled = (X - self.X_mean) / self.X_std if self.normalize else X.copy()
        else:
            X_scaled = X.copy()

        # Extract parameters from kwargs
        method = kwargs.get("method", "lbfgs")
        max_iter = kwargs.get("max_iter", 50)
        tol = kwargs.get("tol", 1e-5)

        # Log transformation method
        self.logger.info(get_message("data", "transformation", method=method))

        # Choose the appropriate optimization method
        if method == "lbfgs":
            return self.transform_with_lbfgs(X_scaled, max_iter=max_iter, tol=tol)
        elif method == "sgd":
            return self._transform_with_improved_sgd(X_scaled, max_iter=max_iter, tol=tol)
        elif method == "adam":
            return self._transform_with_adam(X_scaled, max_iter=max_iter, tol=tol)
        else:
            return self._transform_with_adam(X_scaled, max_iter=max_iter, tol=tol)

    def transform_with_lbfgs(self, X: np.ndarray, max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
        """Transform new data using improved L-BFGS optimization.

        Args:
            X: Data matrix
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            Weight matrix
        """
        if self.archetypes is None:
            raise ValueError("Model must be fitted before transform")

        X_jax = jnp.array(X)

        # Scale archetypes
        archetypes_scaled = (
            (self.archetypes - self.X_mean) / self.X_std
            if self.X_mean is not None and self.X_std is not None
            else self.archetypes
        )
        archetypes_jax = jnp.array(archetypes_scaled)

        @jax.jit
        def objective(w, x):
            pred = jnp.dot(w, archetypes_jax)
            return jnp.sum((x - pred) ** 2)

        @jax.jit
        def project_to_simplex(w):
            w = jnp.maximum(1e-10, w)
            sum_w = jnp.sum(w)
            return jnp.where(sum_w > 1e-10, w / sum_w, jnp.ones_like(w) / self.n_archetypes)

        @jax.jit
        def optimize_single_sample(x):
            w_init = jnp.ones(self.n_archetypes) / self.n_archetypes
            optimizer = optax.adam(learning_rate=0.05)
            opt_state = optimizer.init(w_init)

            def cond_fun(state):
                _, _, _, i, converged = state
                return jnp.logical_and(jnp.logical_not(converged), i < max_iter)

            def body_fun(state):
                w, opt_state, prev_loss, i, _ = state

                loss_val, grad = jax.value_and_grad(lambda w: objective(w, x))(w)
                grad = jnp.clip(grad, -1.0, 1.0)
                updates, new_opt_state = optimizer.update(grad, opt_state)
                new_w = optax.apply_updates(w, updates)
                new_w = project_to_simplex(new_w)

                # Check convergence
                converged = jnp.abs(prev_loss - loss_val) < tol

                return (new_w, new_opt_state, loss_val, i + 1, converged)

            # Run optimization with early stopping
            init_state = (w_init, opt_state, jnp.inf, 0, False)
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

            return final_state[0]  # Return optimized weights

        # Efficient batch processing
        batch_size = min(2000, X_jax.shape[0])
        n_samples = X_jax.shape[0]
        weights = []

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X_jax[i:end]
            batch_weights = jax.vmap(optimize_single_sample)(X_batch)
            weights.append(np.array(batch_weights))

        weights_array = np.vstack(weights) if len(weights) > 1 else weights[0]
        return weights_array

    def _transform_with_adam(self, X: np.ndarray, max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
        """Transform using Adam optimizer with early stopping.

        Args:
            X: Data matrix
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            Weight matrix
        """
        X_jax = jnp.array(X)

        archetypes_scaled = (
            (self.archetypes - self.X_mean) / self.X_std
            if self.X_mean is not None and self.X_std is not None
            else self.archetypes
        )
        archetypes_jax = jnp.array(archetypes_scaled)

        @jax.jit
        def objective(w, x):
            pred = jnp.dot(w, archetypes_jax)
            return jnp.sum((x - pred) ** 2)

        @jax.jit
        def project_to_simplex(w):
            w = jnp.maximum(1e-10, w)
            sum_w = jnp.sum(w)
            return jnp.where(sum_w > 1e-10, w / sum_w, jnp.ones_like(w) / self.n_archetypes)

        @jax.jit
        def optimize_single_sample(x):
            w_init = jnp.ones(self.n_archetypes) / self.n_archetypes
            optimizer = optax.adam(learning_rate=0.03)
            opt_state = optimizer.init(w_init)

            def cond_fun(state):
                _, _, _, i, converged = state
                return jnp.logical_and(jnp.logical_not(converged), i < max_iter)

            def body_fun(state):
                w, opt_state, prev_loss, i, _ = state

                loss_val, grad = jax.value_and_grad(lambda w: objective(w, x))(w)
                grad = jnp.clip(grad, -1.0, 1.0)
                updates, new_opt_state = optimizer.update(grad, opt_state)
                new_w = optax.apply_updates(w, updates)
                new_w = project_to_simplex(new_w)

                # Check convergence
                converged = jnp.abs(prev_loss - loss_val) < tol

                return (new_w, new_opt_state, loss_val, i + 1, converged)

            # Initialize state with iteration counter and convergence flag
            init_state = (w_init, opt_state, jnp.inf, 0, False)
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

            return final_state[0]  # Return optimized weights

        batch_size = min(1000, X_jax.shape[0])
        n_samples = X_jax.shape[0]
        weights = []

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X_jax[i:end]

            batch_weights = jax.vmap(optimize_single_sample)(X_batch)
            weights.append(np.array(batch_weights))

        weights_array = np.vstack(weights) if len(weights) > 1 else weights[0]
        return weights_array

    def _transform_with_improved_sgd(self, X: np.ndarray, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
        """Transform using improved SGD with adaptive learning rate and convergence criteria.

        Args:
            X: Data matrix
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            Weight matrix
        """
        X_jax = jnp.array(X)

        archetypes_scaled = (
            (self.archetypes - self.X_mean) / self.X_std
            if self.X_mean is not None and self.X_std is not None
            else self.archetypes
        )
        archetypes_jax = jnp.array(archetypes_scaled)

        @jax.jit
        def optimize_weights_with_convergence(x_sample):
            w_init = jnp.ones(self.n_archetypes) / self.n_archetypes

            def cond_fun(state):
                _, _, i, converged = state
                return jnp.logical_and(jnp.logical_not(converged), i < max_iter)

            def body_fun(state):
                w, prev_loss, i, _ = state

                # Calculate prediction, error and loss
                pred = jnp.dot(w, archetypes_jax)
                error = x_sample - pred
                loss = jnp.sum(error**2)

                # Check convergence
                converged = jnp.abs(prev_loss - loss) < tol

                # Adaptive learning rate
                lr = 0.01 / (1.0 + 0.005 * i)

                # Gradient computation and update
                grad = -2 * jnp.dot(error, archetypes_jax.T)
                grad = jnp.clip(grad, -1.0, 1.0)

                # Update weights (only if not converged)
                w_new = jnp.where(converged, w, w - lr * grad)

                # Project to constraints
                w_new = jnp.maximum(1e-10, w_new)
                sum_w = jnp.sum(w_new)
                w_new = jnp.where(sum_w > 1e-10, w_new / sum_w, jnp.ones_like(w_new) / self.n_archetypes)

                return (w_new, loss, i + 1, converged)

            # Initialize state with iteration counter and convergence flag
            init_state = (w_init, jnp.inf, 0, False)
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

            return final_state[0]  # Return final weights

        # Batch processing for memory efficiency
        batch_size = min(1000, X_jax.shape[0])
        n_samples = X_jax.shape[0]
        weights = []

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X_jax[i:end]

            batch_weights = jax.vmap(optimize_weights_with_convergence)(X_batch)
            weights.append(np.array(batch_weights))

        weights_array = np.vstack(weights) if len(weights) > 1 else weights[0]
        return weights_array

    def directional_init(self, X_jax, n_samples, n_features):
        """Generate directions using points that are evenly distributed on a sphere."""
        centroid = jnp.mean(X_jax, axis=0)

        # Special handling for low dimensions (2D and 3D)
        if n_features == 2:  # In 2D, arrange points evenly on the circumference of a circle
            angles = jnp.linspace(0, 2 * jnp.pi, self.n_archetypes, endpoint=False)
            directions = jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])
        elif n_features == 3:  # In 3D, use the Fibonacci sphere lattice method
            golden_ratio = (1 + 5**0.5) / 2
            i = jnp.arange(self.n_archetypes)
            theta = 2 * jnp.pi * i / golden_ratio
            phi = jnp.arccos(1 - 2 * (i + 0.5) / self.n_archetypes)

            x = jnp.sin(phi) * jnp.cos(theta)
            y = jnp.sin(phi) * jnp.sin(theta)
            z = jnp.cos(phi)
            directions = jnp.column_stack([x, y, z])
        else:  # For higher dimensions, employ a repulsion method
            # Generate initial directions randomly using the instance's key
            self.rng_key, direction_key = jax.random.split(self.rng_key)
            directions = jax.random.normal(direction_key, (self.n_archetypes, n_features))

            # Normalize the direction vectors
            norms = jnp.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / (norms + 1e-10)

            # Execute the repulsion simulation
            repulsion_strength = 0.1  # Strength of the repulsion force
            n_iterations = 50  # Number of iterations for the repulsion simulation

            def repulsion_step(directions, _):
                # Calculate the dot product between all pairs of directions (a measure of similarity)
                similarities = jnp.dot(directions, directions.T)

                # Set the diagonal elements (self-similarity) to zero
                similarities = similarities - jnp.eye(self.n_archetypes) * similarities

                # Calculate repulsion forces for each direction
                repulsion_forces = jnp.zeros_like(directions)

                # Compute repulsion forces for each pair of directions
                def compute_pair_repulsion(i, forces):
                    # Repulsion from all directions towards the i-th direction
                    repulsions = similarities[i, :, jnp.newaxis] * directions

                    # Exclude repulsion from itself
                    mask = jnp.ones(self.n_archetypes, dtype=bool)
                    mask = mask.at[i].set(False)
                    mask = mask[:, jnp.newaxis]

                    # Calculate the total repulsion (stronger for higher similarity)
                    total_repulsion = jnp.sum(repulsions * mask, axis=0)

                    # Update the repulsion force for the i-th direction
                    return forces.at[i].set(forces[i] - repulsion_strength * total_repulsion)

                # Apply repulsion forces to all directions
                forces = jax.lax.fori_loop(0, self.n_archetypes, compute_pair_repulsion, repulsion_forces)

                # Update the direction vectors
                new_directions = directions + forces

                # Normalize the updated directions
                norms = jnp.linalg.norm(new_directions, axis=1, keepdims=True)
                new_directions = new_directions / (norms + 1e-10)

                return new_directions, None

            # Execute the repulsion simulation
            directions, _ = jax.lax.scan(repulsion_step, directions, jnp.arange(n_iterations))

        # Identify the most extreme points in each direction
        archetypes = jnp.zeros((self.n_archetypes, n_features))

        def find_extreme_point(i, archetypes):
            # Project data points onto the direction
            projections = jnp.dot(X_jax - centroid, directions[i])
            # Find the farthest point
            max_idx = jnp.argmax(projections)
            return archetypes.at[i].set(X_jax[max_idx])

        archetypes = jax.lax.fori_loop(0, self.n_archetypes, find_extreme_point, archetypes)

        indices = jnp.zeros(self.n_archetypes, dtype=jnp.int32)
        return archetypes, indices

    def qhull_init(self, X_jax, n_samples, n_features):
        """Initialize archetypes using convex hull vertices via QHull algorithm."""
        # Convert to numpy for scipy compatibility
        X_np = np.array(X_jax)

        try:
            # Compute the convex hull using scipy's implementation of QHull
            from scipy.spatial import ConvexHull

            self.logger.info("Computing convex hull...")
            hull = ConvexHull(X_np)
        except Exception as e:
            self.logger.warning(
                get_message(
                    "warning", "initialization_failed", strategy="QHull", error_msg=str(e), fallback="k-means++"
                )
            )
            return self.kmeans_pp_init(X_jax, n_samples, n_features)

        # Get the vertices of the convex hull
        vertices = hull.vertices

        # Different strategies depending on the number of vertices vs. required archetypes
        if len(vertices) > self.n_archetypes:
            # Strategy: Select a diverse subset of vertices
            # Start with the first vertex
            selected_indices = [0]
            vertex_points = X_np[vertices]

            # Greedy selection: always pick the farthest point from existing selection
            while len(selected_indices) < self.n_archetypes:
                distances = []
                # For each unselected vertex, find the minimum distance to any selected vertex
                for i in range(len(vertex_points)):
                    if i in selected_indices:
                        continue
                    min_dist = float("inf")
                    for j in selected_indices:
                        dist = np.sum((vertex_points[i] - vertex_points[j]) ** 2)
                        min_dist = min(min_dist, dist)
                    distances.append((i, min_dist))

                # Select the farthest point
                if distances:
                    next_idx = max(distances, key=lambda x: x[1])[0]
                    selected_indices.append(next_idx)

            # Get the final selected vertices
            selected_vertices = [vertices[i] for i in selected_indices]

        # If we have fewer vertices than required archetypes, add some random points
        elif len(vertices) < self.n_archetypes:
            # Strategy: Use all vertices and add random points from the data
            selected_vertices = list(vertices)

            # How many more archetypes do we need?
            remaining = self.n_archetypes - len(vertices)

            # Sample additional points randomly using the instance's key
            self.rng_key, choice_key = jax.random.split(self.rng_key)
            additional_indices = jax.random.choice(choice_key, n_samples, shape=(remaining,), replace=False, p=None)
            selected_vertices.extend(additional_indices)

        # Use the selected vertices as initial archetypes
        archetypes = jnp.array(X_np[selected_vertices])

        return archetypes, jnp.array(selected_vertices)

    def kmeans_pp_init(self, X_jax, n_samples, n_features):
        """More efficient k-means++ style initialization using JAX."""
        # Use the instance's RNG key
        self.rng_key, first_center_key = jax.random.split(self.rng_key)

        # Randomly select the first center
        first_idx = jax.random.randint(first_center_key, (), 0, n_samples)

        # Store selected indices and centers
        chosen_indices = jnp.zeros(self.n_archetypes, dtype=jnp.int32)
        chosen_indices = chosen_indices.at[0].set(first_idx)

        # Store selected archetypes
        archetypes = jnp.zeros((self.n_archetypes, n_features))
        archetypes = archetypes.at[0].set(X_jax[first_idx])

        # Select remaining archetypes
        for i in range(1, self.n_archetypes):
            # Calculate squared distance from each point to the nearest existing center
            dists = jnp.sum((X_jax[:, jnp.newaxis, :] - archetypes[jnp.newaxis, :i, :]) ** 2, axis=2)
            min_dists = jnp.min(dists, axis=1)

            # Set distance to 0 for already selected points
            mask = jnp.ones(n_samples, dtype=bool)
            for j in range(i):
                mask = mask & (jnp.arange(n_samples) != chosen_indices[j])
            min_dists = min_dists * mask

            # Select next center with probability proportional to squared distance
            sum_dists = jnp.sum(min_dists) + 1e-10
            probs = min_dists / sum_dists

            self.rng_key, choice_key = jax.random.split(self.rng_key)
            next_idx = jax.random.choice(choice_key, n_samples, p=probs)

            # Update selected indices and centers
            chosen_indices = chosen_indices.at[i].set(next_idx)
            archetypes = archetypes.at[i].set(X_jax[next_idx])

        return archetypes, chosen_indices

    def fit(
        self,
        X: np.ndarray,
        normalize: bool = False,
        **kwargs,
    ) -> "ImprovedArchetypalAnalysis":
        """Fit the model with improved k-means++ initialization.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            normalize: Whether to normalize the data before fitting.
            **kwargs: Additional keyword arguments for the fit method.

        Returns:
            self: The fitted model.
        """

        @partial(jax.jit, static_argnums=(3))
        def update_step(params, opt_state, X, iteration):
            """Execute a single optimization step with mixed precision."""

            def loss_fn(params):
                return self.loss_function(params["archetypes"], params["weights"], X_f32)

            params_f32 = jax.tree.map(lambda p: p.astype(jnp.float32), params)
            X_f32 = X.astype(jnp.float32)

            loss, grads = jax.value_and_grad(loss_fn)(params_f32)
            grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params_f32, updates)

            # Always project weights to simplex
            new_params["weights"] = self.project_weights(new_params["weights"])

            # Alternating optimization: periodically use direct archetype update instead of gradient
            # This helps break out of local minima and improves convergence characteristics
            use_direct_update = jnp.mod(iteration, 15) == 0  # Every 15 iterations

            def apply_direct_update():
                """Apply direct algebraic update to archetypes."""
                archetypes_dir = self.update_archetypes(new_params["archetypes"], new_params["weights"], X_f32)
                # Blend with gradient-based update to maintain stability
                blend = 0.2  # 20% direct update, 80% gradient update
                return blend * archetypes_dir + (1 - blend) * new_params["archetypes"]

            # Apply direct update conditionally
            new_params["archetypes"] = jax.lax.cond(
                use_direct_update, lambda: apply_direct_update(), lambda: new_params["archetypes"]
            )

            # Store pre-projection archetypes
            pre_projection_archetypes = new_params["archetypes"]
            pre_projection_loss = self.loss_function(pre_projection_archetypes, new_params["weights"], X_f32)

            # Intermittent projection: only project archetypes every N iterations
            # This allows optimization to make progress between projections
            do_projection = jnp.mod(iteration, 10) == 0

            # Conditional projection function
            def project_archetypes():
                # Select appropriate projection method
                if self.projection_method == "cbap" or self.projection_method == "default":
                    projected = self.project_archetypes(new_params["archetypes"], X_f32)
                elif self.projection_method == "convex_hull":
                    projected = self.project_archetypes_convex_hull(new_params["archetypes"], X_f32)
                else:
                    projected = self.project_archetypes_knn(new_params["archetypes"], X_f32)

                # Calculate post-projection loss
                post_projection_loss = self.loss_function(projected, new_params["weights"], X_f32)

                # Use adaptive blending based on loss differential
                # If projection increases loss, use less of the projected result
                loss_ratio = post_projection_loss / (pre_projection_loss + 1e-10)
                blend_factor = jnp.where(
                    loss_ratio > 1.01,  # Loss increased by more than 1%
                    0.01,  # Use only 7% of projected result if loss increases (reduced from 10%)
                    0.5,  # Otherwise use 60% of projected result (reduced from 50%)
                )

                # Severe loss increase detection and mitigation
                blend_factor = jnp.where(
                    loss_ratio > 1.1,  # Loss increased by more than 10%
                    0.005,  # Barely use the projection (0.5%)
                    blend_factor,  # Otherwise use standard blend factor
                )

                # Blend original and projected archetypes
                return blend_factor * projected + (1 - blend_factor) * pre_projection_archetypes

            # Only apply projection on designated iterations
            new_params["archetypes"] = jax.lax.cond(
                do_projection, lambda: project_archetypes(), lambda: pre_projection_archetypes
            )

            # Ensure consistent data types
            new_params = jax.tree.map(lambda p: p.astype(jnp.float32), new_params)

            return new_params, opt_state, loss

        # Preprocess data: scale for improved stability
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        # Prevent division by zero with explicit type casting
        if self.X_std is not None:
            self.X_std = np.where(self.X_std < 1e-10, np.ones_like(self.X_std), self.X_std)

        X_scaled = (X - self.X_mean) / self.X_std if self.normalize else X.copy()

        # Convert to JAX array
        X_jax = jnp.array(X_scaled, dtype=jnp.float32)
        n_samples, n_features = X_jax.shape

        # Debug information
        self.logger.info(
            get_message("data", "data_shape", shape=X_jax.shape, min=float(jnp.min(X_jax)), max=float(jnp.max(X_jax)))
        )

        # Use the instance's RNG key for weight initialization
        self.rng_key, weights_key = jax.random.split(self.rng_key)

        # Initialize weights (more stable initialization)
        weights_init = jax.random.uniform(
            weights_key,
            (n_samples, self.n_archetypes),
            minval=0.1,
            maxval=0.9,
            dtype=jnp.float32,
        )
        weights_init = self.project_weights(weights_init)

        # archetype initialization
        if self.archetype_init_method == "directional" or self.archetype_init_method == "direction":
            archetypes_init, _ = self.directional_init(X_jax, n_samples, n_features)
        elif self.archetype_init_method == "qhull" or self.archetype_init_method == "convex_hull":
            archetypes_init, _ = self.qhull_init(X_jax, n_samples, n_features)
        else:
            archetypes_init, _ = self.kmeans_pp_init(X_jax, n_samples, n_features)

        archetypes_init = archetypes_init.astype(jnp.float32)

        # The rest is the same as the original fit method
        # Set up optimizer (Adam with reduced learning rate)
        optimizer: optax.GradientTransformation = optax.adam(learning_rate=self.learning_rate)

        # Initialize parameters
        params = {"archetypes": archetypes_init, "weights": weights_init}
        opt_state = optimizer.init(params)

        # Optimization loop
        prev_loss = float("inf")
        self.loss_history = []

        # Track best parameters for early stopping and parameter recovery
        best_loss = float("inf")
        best_params = {k: v.copy() for k, v in params.items()}
        no_improvement_count = 0
        # Save the previous archetypes to track changes
        prev_archetypes = archetypes_init.copy()

        # Calculate initial loss for debugging
        initial_loss = float(self.loss_function(archetypes_init, weights_init, X_jax))
        self.logger.info(f"Initial loss: {initial_loss:.6f}")

        for i in range(self.max_iter):
            # Execute update step
            try:
                params, opt_state, loss = update_step(params, opt_state, X_jax, i)
                loss_value = float(loss)

                # Calculate the changes in archetypes
                current_archetypes = params["archetypes"]
                archetype_changes = np.array(current_archetypes) - np.array(prev_archetypes)

                # Calculate the norm of changes for each archetype
                change_norms = np.linalg.norm(archetype_changes, axis=1)
                avg_change = np.mean(change_norms)
                max_change = np.max(change_norms)

                # Display changes (every 50 iterations or if there are significant changes)
                if i % 50 == 0 or max_change > 0.1:
                    self.logger.info(
                        f"Iteration {i}, Archetype changes: Average={avg_change:.6f}, Maximum={max_change:.6f}"
                    )
                    # Display indices of archetypes with significant changes
                    if max_change > 0.01:  # Set threshold
                        large_changes = np.where(change_norms > 0.01)[0]
                        if len(large_changes) > 0:
                            self.logger.info(
                                f"  Archetypes with significant changes: {large_changes}, Changes: {change_norms[large_changes]}"
                            )

                # Save the current archetypes as the previous archetypes for the next iteration
                prev_archetypes = current_archetypes.copy()

                # Check for NaN
                if jnp.isnan(loss_value):
                    self.logger.warning(get_message("warning", "nan_detected", iteration=i))
                    break

                # Record loss
                self.loss_history.append(loss_value)

                # Track best parameters
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_params = {k: v.copy() for k, v in params.items()}
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Implement early stopping with parameter restoration
                if no_improvement_count >= self.early_stopping_patience:
                    self.logger.info(
                        get_message("progress", "early_stopping", iteration=i, patience=self.early_stopping_patience)
                    )
                    # Restore best parameters
                    params = best_params
                    break

                # Periodically check if loss is increasing and restore best parameters if necessary
                if i > 0 and i % 20 == 0 and loss_value > prev_loss * 1.05:
                    self.logger.warning(get_message("warning", "loss_increase", previous=prev_loss, current=loss_value))
                    params = {k: v.copy() for k, v in best_params.items()}
                    # Re-initialize optimizer state
                    opt_state = optimizer.init(params)

                # Check convergence
                if i > 0 and abs(prev_loss - loss_value) < self.tol:
                    self.logger.info(get_message("progress", "converged", iteration=i, tolerance=self.tol))
                    break

                prev_loss = loss_value

                # Show progress
                if i % 50 == 0:
                    self.logger.info(
                        get_message("progress", "iteration_progress", current=i, total=self.max_iter, loss=loss_value)
                    )

            except Exception as e:
                self.logger.error(get_message("error", "computation_error", error_msg=str(e)))
                # Restore best parameters in case of error
                params = best_params
                break

        # Use best parameters for final model
        if best_loss < loss_value:
            self.logger.info(get_message("result", "final_loss", loss=best_loss, iterations=len(self.loss_history)))
            params = best_params

        # Display the total change in archetypes
        total_change = np.linalg.norm(np.array(params["archetypes"]) - np.array(archetypes_init), axis=1)
        self.logger.info("Total change in archetypes:")
        for i, change in enumerate(total_change):
            self.logger.info(f"  Archetype {i + 1}: {change:.6f}")

        # Inverse scale transformation
        archetypes_scaled = np.array(params["archetypes"])
        self.archetypes = archetypes_scaled * self.X_std + self.X_mean if normalize else archetypes_scaled
        self.weights = np.array(params["weights"])

        if len(self.loss_history) > 0:
            self.logger.info(
                get_message("result", "final_loss", loss=self.loss_history[-1], iterations=len(self.loss_history))
            )
        else:
            self.logger.warning("No valid loss was recorded")

        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit the model and return the transformed data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            y: Ignored. Present for API consistency by convention.
            normalize: Whether to normalize the data before fitting.
            **kwargs: Additional keyword arguments for the fit method.

        Returns:
            Weight matrix representing each sample as a combination of archetypes
        """
        model = self.fit(X, **kwargs)
        return np.asarray(model.transform(X, **kwargs))

    @partial(jax.jit, static_argnums=(0,))
    def project_archetypes(self, archetypes, X) -> jnp.ndarray:
        """JIT-compiled archetype projection that pushes archetypes towards the convex hull boundary.

        Instead of using k-NN which tends to pull archetypes inside the convex hull,
        this implementation pushes archetypes towards the boundary of the convex hull
        by finding extreme points in the direction of each archetype.

        Technical details:
        - Boundary Projection Approach: Projects data points along the direction from the
          data centroid to each archetype, then identifies the most extreme point in that direction.
          This effectively "pushes" archetypes toward the convex hull boundary rather than
          pulling them inward.
        - Stability Enhancement: Blends the original archetype with the extreme point using
          a weighted average (20% extreme point, 80% original archetype) to prevent abrupt
          changes and ensure optimization stability.

        Args:
            archetypes: Current archetype matrix
            X: Original data matrix

        Returns:
            Projected archetype matrix positioned closer to the convex hull boundary
        """
        # Find the centroid of the data
        centroid = jnp.mean(X, axis=0)

        def _project_to_boundary(archetype):
            # Direction from centroid to archetype
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)

            # Avoid division by zero
            normalized_direction = jnp.where(
                direction_norm > 1e-10, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Project all points onto this direction
            projections = jnp.dot(X - centroid, normalized_direction)

            # Find the most extreme point in this direction
            max_idx = jnp.argmax(projections)
            extreme_point = X[max_idx]

            # Calculate the projection of the extreme point onto the direction
            extreme_projection = jnp.dot(extreme_point - centroid, normalized_direction)
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

            # Ensure the archetype doesn't go beyond the extreme point
            # If archetype is already outside, pull it back inside
            is_outside = archetype_projection > extreme_projection

            # More conservative projection - smaller alpha value
            # Use a much smaller coefficient to prevent overshooting
            adaptive_alpha = jnp.minimum(0.2, self.projection_alpha * 2.0)

            # Different blending strategy depending on whether archetype is inside or outside
            projected = jnp.where(
                is_outside,
                # If outside, interpolate back toward boundary (move 60% back toward extreme point)
                0.6 * extreme_point + 0.4 * archetype,
                # If inside, move gently toward boundary
                adaptive_alpha * extreme_point + (1 - adaptive_alpha) * archetype,
            )

            return projected

        # Apply the projection to each archetype
        projected_archetypes = jax.vmap(_project_to_boundary)(archetypes)

        return jnp.asarray(projected_archetypes)

    # Alternative implementation that can be used for comparison or experimentation
    @partial(jax.jit, static_argnums=(0,))
    def project_archetypes_convex_hull(self, archetypes, X) -> jnp.ndarray:
        """Alternative archetype projection that uses convex combinations of extreme points.

        This method identifies potential extreme points and creates archetypes as
        sparse convex combinations of these points, ensuring they lie on the boundary.

        Technical details:
        - Multi-directional Exploration: Generates multiple random directions around the
          main archetype direction, allowing for more diverse extreme point discovery.
        - Sparse Convex Combinations: Creates archetypes as weighted combinations of
          extreme points found in different directions, with emphasis on the main direction.
        - Boundary Positioning: By using convex combinations of extreme points, archetypes
          are positioned on or near the convex hull boundary rather than in its interior.

        This approach offers potentially better exploration of the convex hull boundary
        at the cost of slightly higher computational complexity.

        Args:
            archetypes: Current archetype matrix
            X: Original data matrix

        Returns:
            Projected archetype matrix positioned on the convex hull boundary
        """
        # Find the centroid of the data
        centroid = jnp.mean(X, axis=0)

        # For each archetype, find a set of extreme points
        def _find_extreme_points(archetype, rng_key):
            # Generate multiple random directions around the archetype direction
            # Use the provided RNG key for stochastic operations
            n_directions = 5

            # Direction from centroid to archetype
            main_direction = archetype - centroid
            main_direction_norm = jnp.linalg.norm(main_direction)

            # Split the key for different random operations
            rng_key, direction_key = jax.random.split(rng_key)

            # Avoid division by zero
            main_direction = jnp.where(
                main_direction_norm > 1e-10,
                main_direction / main_direction_norm,
                jax.random.normal(direction_key, shape=main_direction.shape),
            )

            # Generate random perturbations of the main direction
            rng_key, perturbation_key = jax.random.split(rng_key)
            perturbations = jax.random.normal(perturbation_key, shape=(n_directions, main_direction.shape[0]))

            # Normalize the perturbations
            perturbation_norms = jnp.linalg.norm(perturbations, axis=1, keepdims=True)
            normalized_perturbations = perturbations / (perturbation_norms + 1e-10)

            # Create directions as combinations of main direction and perturbations
            directions = jnp.vstack([main_direction, normalized_perturbations * 0.3 + main_direction * 0.7])

            # Normalize again
            direction_norms = jnp.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / (direction_norms + 1e-10)

            # Find extreme points in each direction
            extreme_indices = jnp.zeros(directions.shape[0], dtype=jnp.int32)

            def _find_extreme(i, indices):
                # Project all points onto this direction
                projections = jnp.dot(X - centroid, directions[i])
                # Find the most extreme point
                max_idx = jnp.argmax(projections)
                # Update the indices
                return indices.at[i].set(max_idx)

            extreme_indices = jax.lax.fori_loop(0, directions.shape[0], _find_extreme, extreme_indices)

            # Get the extreme points
            extreme_points = X[extreme_indices]

            # Create a sparse convex combination of these extreme points
            # with higher weight on the main direction's extreme point
            weights = jnp.ones(extreme_points.shape[0]) / extreme_points.shape[0]
            weights = weights.at[0].set(weights[0] * 2)  # Double weight for main direction
            weights = weights / jnp.sum(weights)  # Normalize to sum to 1

            projected = jnp.sum(weights[:, jnp.newaxis] * extreme_points, axis=0)

            return projected, rng_key

        # Derive RNG keys from the instance key for each archetype
        self.rng_key, subkey = jax.random.split(self.rng_key)
        rng_keys = jax.random.split(subkey, archetypes.shape[0])

        # Process each archetype with its own RNG key
        projected_archetypes = jnp.zeros_like(archetypes)

        # Apply projection to each archetype
        for i in range(archetypes.shape[0]):
            projected, _ = _find_extreme_points(archetypes[i], rng_keys[i])
            projected_archetypes = projected_archetypes.at[i].set(projected)

        return jnp.asarray(projected_archetypes)

    # Keep the original k-NN method for comparison
    @partial(jax.jit, static_argnums=(0,))
    def project_archetypes_knn(self, archetypes, X) -> jnp.ndarray:
        """Original k-NN based archetype projection (kept for comparison).

        This method tends to pull archetypes inside the convex hull due to its
        averaging nature, which is suboptimal for archetypal analysis where
        archetypes should ideally lie on the convex hull boundary.

        Args:
            archetypes: Current archetype matrix
            X: Original data matrix

        Returns:
            Projected archetype matrix (typically positioned inside the convex hull)
        """

        def _process_single_archetype(i):
            archetype_dists = dists[:, i]
            top_k_indices = jnp.argsort(archetype_dists)[:k]
            top_k_dists = archetype_dists[top_k_indices]
            weights = 1.0 / (top_k_dists + 1e-10)
            weights = weights / jnp.sum(weights)
            projected = jnp.sum(weights[:, jnp.newaxis] * X[top_k_indices], axis=0)
            return projected

        dists = jnp.sum((X[:, jnp.newaxis, :] - archetypes[jnp.newaxis, :, :]) ** 2, axis=2)
        k = 10

        projected_archetypes = jax.vmap(_process_single_archetype)(jnp.arange(archetypes.shape[0]))

        return jnp.asarray(projected_archetypes)

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, archetypes, weights, X):
        """JIT-compiled loss function with mixed precision."""
        archetypes_f32 = archetypes.astype(jnp.float32)
        weights_f32 = weights.astype(jnp.float32)
        X_f32 = X.astype(jnp.float32)

        X_reconstructed = jnp.matmul(weights_f32, archetypes_f32)
        reconstruction_loss = jnp.mean(jnp.sum((X_f32 - X_reconstructed) ** 2, axis=1))

        # Calculate entropy (higher values for uniform weights, lower for sparse weights)
        entropy = -jnp.sum(weights_f32 * jnp.log(weights_f32 + 1e-10), axis=1)

        # Corrected sign: Penalize high entropy (uniform weights) to encourage sparsity
        # Previous implementation unintentionally encouraged uniform weights
        entropy_reg = jnp.mean(entropy)

        # Add incentive for archetypes to stay near convex hull boundary
        boundary_incentive = self._calculate_boundary_proximity(archetypes_f32, X_f32)

        # Combined loss with reconstruction, entropy regularization, and boundary incentive
        # Negative sign for boundary_incentive as higher values should reduce the loss
        total_loss = reconstruction_loss + self.lambda_reg * entropy_reg - 0.001 * boundary_incentive

        return total_loss.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_boundary_proximity(self, archetypes, X):
        """Calculate a metric that rewards archetypes for being near convex hull boundary.

        A high value indicates archetypes are closer to the convex hull boundary,
        which is desirable for archetypal analysis. This serves as a regularization term
        that encourages archetypes to move toward extremal positions.
        """
        # Find the centroid of the data
        centroid = jnp.mean(X, axis=0)

        def _boundary_score_for_archetype(archetype):
            # Direction from centroid to archetype
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)

            # Skip computation for archetypes too close to centroid
            normalized_direction = jnp.where(
                direction_norm > 1e-10, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Project all points onto this direction
            projections = jnp.dot(X - centroid, normalized_direction)

            # Calculate distance from archetype to most extreme point in that direction
            max_projection = jnp.max(projections)
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

            # Normalized proximity to boundary (1.0 = on boundary, 0.0 = at centroid)
            # Small epsilon to prevent division by zero
            normalized_proximity = archetype_projection / (max_projection + 1e-10)

            # Penalize archetypes outside the convex hull
            # This creates a peak at exactly the boundary (normalized_proximity = 1.0)
            # and penalizes positions both inside (< 1.0) and outside (> 1.0)
            boundary_score = 1.0 - jnp.abs(normalized_proximity - 1.0)

            # Severe penalty for being outside the boundary
            is_outside = normalized_proximity > 1.0
            outside_penalty = jnp.where(is_outside, jnp.exp(normalized_proximity - 1.0) - 1.0, 0.0)

            # Final combined score with exponential penalty for outside positions
            return jnp.power(boundary_score, 2) - outside_penalty

        # Calculate score for each archetype and return mean
        scores = jax.vmap(_boundary_score_for_archetype)(archetypes)
        return jnp.mean(scores)

    @partial(jax.jit, static_argnums=(0,))
    def project_weights(self, weights):
        """JIT-compiled weight projection function."""
        eps = 1e-10
        weights = jnp.maximum(eps, weights)
        sum_weights = jnp.sum(weights, axis=1, keepdims=True)
        sum_weights = jnp.maximum(eps, sum_weights)
        return weights / sum_weights

    @partial(jax.jit, static_argnums=(0,))
    def update_archetypes(self, archetypes, weights, X) -> jnp.ndarray:
        """Alternative archetype update strategy based on weighted reconstruction.

        This approach directly optimizes archetypes by computing the pseudo-inverse of weights,
        which often provides a more targeted and mathematically sound update than gradient
        descent for this specific subproblem.
        """
        # Calculate weight matrix pseudoinverse with improved numerical stability
        # Add a small regularization term to the weights to stabilize the computation
        W = weights
        WtW = jnp.dot(W.T, W) + 1e-6 * jnp.eye(W.shape[1])
        WtX = jnp.dot(W.T, X)

        # Solve for archetypes using the normal equations
        # This is equivalent to minimizing ||X - W*A||^2 with respect to A
        archetypes_updated = jnp.linalg.solve(WtW, WtX)

        # Ensure the updated archetypes remain within the convex hull of X
        # by applying a weighted convex combination with data points

        # Compute distances to find the closest data point for each archetype
        # Add a small regularization term to ensure numerical stability
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


class ArchetypeTracker(ImprovedArchetypalAnalysis):
    """Extension of ImprovedArchetypalAnalysis that tracks archetype movement over iterations."""

    def __init__(self, *args, **kwargs):
        """Initialize the ArchetypeTracker.

        Args:
            *args: Positional arguments for ImprovedArchetypalAnalysis
            **kwargs: Keyword arguments for ImprovedArchetypalAnalysis
        """
        super().__init__(*args, **kwargs)
        # Initialize class-specific logger
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            get_message("init", "model_init", model_name=self.__class__.__name__, n_archetypes=self.n_archetypes)
        )

        self.early_stopping_patience = min(kwargs.get("early_stopping_patience", 100), 100)
        self.archetype_history = []
        self.loss_history = []
        self.optimizer: optax.GradientTransformation = optax.adam(learning_rate=self.learning_rate)
        # Specific settings for archetype updates
        self.archetype_grad_scale = 1.0  # Gradient scale for archetypes (reduced from 1.5 to 1.0)
        self.noise_scale = 0.02  # Magnitude of initial noise (reduced from 0.03)
        self.exploration_noise_scale = 0.05  # Magnitude of exploration noise (reduced from 0.07)
        # Track position metrics
        self.boundary_proximity_history = []  # History of boundary proximity scores
        self.is_outside_history = []  # History of whether archetypes are outside the convex hull

    def fit(self, X: np.ndarray, normalize: bool = False, **kwargs) -> "ArchetypeTracker":
        """Train the model while documenting the positions of archetypes at each iteration.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            normalize: Whether to normalize the data before fitting.
            **kwargs: Additional keyword arguments for the fit method.

        Returns:
            Self
        """
        # Data preprocessing
        if normalize:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            X = (X - self.X_mean) / self.X_std

        # Convert to JAX array
        X_jax = jnp.array(X)
        n_samples, n_features = X_jax.shape

        # Store current iteration for use in adaptive projection
        self.current_iteration = 0

        # Initialize archetypes based on selected method
        if self.archetype_init_method == "directional" or self.archetype_init_method == "direction":
            archetypes, _ = self.directional_init(X_jax, n_samples, n_features)
        elif self.archetype_init_method == "qhull" or self.archetype_init_method == "convex_hull":
            archetypes, _ = self.qhull_init(X_jax, n_samples, n_features)
        else:
            archetypes, _ = self.kmeans_pp_init(X_jax, n_samples, n_features)

        # Store initial archetypes
        self.archetype_history = [np.array(archetypes)]

        # Track initial boundary proximity
        initial_proximity = self._calculate_boundary_proximity(archetypes, X_jax)
        self.boundary_proximity_history.append(float(initial_proximity))

        # Track whether archetypes are outside the convex hull
        self.is_outside_history.append(self._check_archetypes_outside(archetypes, X_jax))

        # Set up optimization
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = self.optimizer.init(archetypes)

        # Optimization loop
        prev_loss = float("inf")
        best_loss = float("inf")
        best_archetypes = archetypes.copy()
        no_improvement_count = 0

        # Save previous archetypes for change tracking
        prev_archetypes = archetypes.copy()

        # Calculate initial loss
        weights = self._calculate_weights(X_jax, archetypes)
        initial_loss = float(self.loss_function(archetypes, weights, X_jax))
        self.loss_history.append(initial_loss)
        self.logger.info(f"Initial loss: {initial_loss:.6f}")

        # Calculate loss and gradients with archetype as the only variable
        def loss_fn(arch):
            return self.loss_function(arch, weights, X_jax)

        for i in range(self.max_iter):
            # Update current iteration
            self.current_iteration = i

            # Calculate weights
            weights = self._calculate_weights(X_jax, archetypes)
            loss_value, grads = jax.value_and_grad(loss_fn)(archetypes)

            # Adjust and clip gradients - use tighter clipping initially, gradually relaxing
            gradient_clip = jnp.minimum(0.2 + i * 0.001, 1.0)  # Start with tight clipping, gradually increase
            grads = grads * self.archetype_grad_scale
            grads = jnp.clip(grads, -gradient_clip, gradient_clip)  # Progressive clipping strategy

            # Execute optimization step
            updates, opt_state = self.optimizer.update(grads, opt_state)
            archetypes = optax.apply_updates(archetypes, updates)

            # Calculate the changes in archetypes
            current_archetypes = archetypes
            archetype_changes = np.array(current_archetypes) - np.array(prev_archetypes)

            # Calculate the norm of changes for each archetype
            change_norms = np.linalg.norm(archetype_changes, axis=1)
            avg_change = np.mean(change_norms)
            max_change = np.max(change_norms)

            # Display changes (every 50 iterations or if there are significant changes)
            if i % 50 == 0 or max_change > 0.1:
                self.logger.info(
                    f"Iteration {i}, Archetype changes: Average={avg_change:.6f}, Maximum={max_change:.6f}"
                )
                # Display indices of archetypes with significant changes
                if max_change > 0.01:  # Set threshold
                    large_changes = np.where(change_norms > 0.01)[0]
                    if len(large_changes) > 0:
                        self.logger.info(
                            f"  Archetypes with significant changes: {large_changes}, Changes: {change_norms[large_changes]}"
                        )

            # Save the current archetypes as the previous archetypes for the next iteration
            prev_archetypes = current_archetypes.copy()

            # Apply direct algebraic update periodically - match parent class interval
            if i % 15 == 0:
                archetypes_dir = self.update_archetypes(archetypes, weights, X_jax)
                # Blend with gradient-based update - match parent class blend factor
                # Use a more conservative blend factor for better stability
                blend = 0.2  # 20% direct update, 80% gradient update (matching parent class)
                archetypes = blend * archetypes_dir + (1 - blend) * archetypes

            # Apply projection to convex hull boundary periodically - match parent class interval
            if i % 10 == 0:  # Matching parent class interval of 10
                pre_projection_archetypes = archetypes.copy()
                if self.projection_method == "cbap" or self.projection_method == "default":
                    projected = self.project_archetypes(archetypes, X_jax)
                elif self.projection_method == "convex_hull":
                    projected = self.project_archetypes_convex_hull(archetypes, X_jax)
                else:
                    projected = self.project_archetypes_knn(archetypes, X_jax)

                # Calculate loss before and after projection
                pre_loss = float(self.loss_function(pre_projection_archetypes, weights, X_jax))
                post_loss = float(self.loss_function(projected, weights, X_jax))

                # Adaptive blending based on loss change - using parent class logic
                loss_ratio = post_loss / (pre_loss + 1e-10)
                # Use extremely conservative blending during the initial iterations
                early_phase_factor = jnp.maximum(0.5, 1.0 - i / 50)  # Reduces from 1.0 to 0.5 over first 50 iterations

                if loss_ratio > 1.1:  # Loss increased by more than 10%
                    blend_factor = 0.005 * (1.0 - early_phase_factor)  # Near zero in early iterations
                elif loss_ratio > 1.01:  # Loss increased by more than 1%
                    blend_factor = 0.01 * (1.0 - early_phase_factor)  # Near zero in early iterations
                else:
                    # Start very conservatively and gradually increase
                    max_blend = 0.5  # Same as parent class
                    blend_factor = max_blend * (1.0 - early_phase_factor)

                # Apply blended projection
                archetypes = blend_factor * projected + (1 - blend_factor) * pre_projection_archetypes

            # Ensure archetypes remain within convex hull
            archetypes = self._constrain_to_convex_hull_batch(archetypes, X_jax)

            # Calculate loss with updated archetypes and weights
            weights = self._calculate_weights(X_jax, archetypes)
            loss_value = float(self.loss_function(archetypes, weights, X_jax))

            # Store history
            self.archetype_history.append(np.array(archetypes))
            self.loss_history.append(loss_value)

            # Track boundary proximity
            boundary_proximity = self._calculate_boundary_proximity(archetypes, X_jax)
            self.boundary_proximity_history.append(float(boundary_proximity))

            # Track whether archetypes are outside the convex hull
            self.is_outside_history.append(self._check_archetypes_outside(archetypes, X_jax))

            # Update best parameters
            if loss_value < best_loss:
                best_loss = loss_value
                best_archetypes = archetypes.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # More aggressive early stopping for tracker to prevent excessive movement
            if no_improvement_count >= self.early_stopping_patience:
                self.logger.info(f"No improvement for {no_improvement_count} iterations. Early stopping.")
                break

            # More sensitive loss increase detection for tracker
            if i > 0 and loss_value > prev_loss * 1.02:  # Reduced from 1.05 to 1.02
                archetypes = best_archetypes.copy()
                # Re-initialize optimizer
                opt_state = self.optimizer.init(archetypes)

            # Convergence check
            if i > 0 and abs(prev_loss - loss_value) < self.tol:
                self.logger.info(f"Converged at iteration {i}")
                break

            prev_loss = loss_value

            # Log progress
            if i % 50 == 0:
                outside_count = np.sum(self.is_outside_history[-1])
                self.logger.info(
                    f"Iteration {i}, Loss: {loss_value:.6f}, "
                    + f"Best loss: {best_loss:.6f}, "
                    + f"Boundary proximity: {float(boundary_proximity):.4f}, "
                    + f"Archetypes outside: {outside_count}/{self.n_archetypes}"
                )

        # Use best archetypes
        if best_loss < loss_value:
            self.logger.info(f"Using best parameters with loss: {best_loss:.6f}")
            archetypes = best_archetypes

        # Display the total change in archetypes
        initial_archetypes = self.archetype_history[0]
        total_change = np.linalg.norm(np.array(archetypes) - np.array(initial_archetypes), axis=1)
        self.logger.info("Total change in archetypes:")
        for i, change in enumerate(total_change):
            self.logger.info(f"  Archetype {i + 1}: {change:.6f}")

        # Final weights calculation
        weights = self._calculate_weights(X_jax, archetypes)

        # Store final model
        self.archetypes = np.array(archetypes) * self.X_std + self.X_mean if normalize else np.array(archetypes)
        self.weights = np.array(weights)

        # Scale history if normalized
        if normalize:
            self.archetype_history = [arch * self.X_std + self.X_mean for arch in self.archetype_history]

        return self

    def _calculate_weights(self, X, archetypes):
        """Calculate optimal weights for given archetypes using JAX."""

        @jax.jit
        def calculate_single_weight(x_sample, archetypes):
            # Initial weights
            w = jnp.ones(self.n_archetypes) / self.n_archetypes

            # Define weight update step
            def weight_update_step(w, _):
                # Calculate prediction and error
                pred = jnp.dot(w, archetypes)
                error = x_sample - pred
                grad = -2.0 * jnp.dot(error, archetypes.T)

                # Apply gradient descent and constraints
                w_new = w - 0.01 * grad
                w_new = jnp.maximum(1e-10, w_new)
                sum_w = jnp.sum(w_new)
                w_new = jnp.where(sum_w > 1e-10, w_new / sum_w, jnp.ones_like(w_new) / self.n_archetypes)
                return w_new, None

            # Run 100 iterations of optimization using scan
            final_w, _ = jax.lax.scan(weight_update_step, w, None, length=100)
            return final_w

        # Parallelize weight calculation across all samples
        return jax.vmap(calculate_single_weight, in_axes=(0, None))(X, archetypes)

    def _check_archetypes_outside(self, archetypes, X):
        """Check if archetypes are outside the convex hull."""
        centroid = jnp.mean(X, axis=0)

        def check_single_archetype(archetype):
            # Direction from centroid to archetype
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)

            # Skip near-zero norm
            normalized_direction = jnp.where(
                direction_norm > 1e-10, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Project all points onto this direction
            projections = jnp.dot(X - centroid, normalized_direction)

            # Calculate archetype projection
            max_projection = jnp.max(projections)
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

            # Check if archetype is beyond the furthest data point
            return archetype_projection > max_projection

        # Apply check to each archetype
        is_outside = jax.vmap(check_single_archetype)(archetypes)
        return np.array(is_outside)

    def _constrain_to_convex_hull_batch(self, archetypes, X):
        """Ensure all archetypes are within the convex hull."""
        return jax.vmap(lambda arch: self._constrain_to_convex_hull(arch, X))(archetypes)

    def _constrain_to_convex_hull(self, archetype, X):
        """Constrain a single archetype to be within the convex hull.

        More conservative implementation than the parent class,
        keeping archetypes slightly inside the convex hull boundary.
        """
        centroid = jnp.mean(X, axis=0)

        # Direction from centroid to archetype
        direction = archetype - centroid
        direction_norm = jnp.linalg.norm(direction)

        # Handle near-zero norm
        normalized_direction = jnp.where(direction_norm > 1e-10, direction / direction_norm, jnp.zeros_like(direction))

        # Project all points onto this direction
        projections = jnp.dot(X - centroid, normalized_direction)

        # Find extreme point in this direction
        max_projection = jnp.max(projections)

        # Calculate archetype projection
        archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

        # Scale if outside or too close to boundary
        # Use a more conservative factor for tracker (0.95 instead of 0.99) to stay further from boundary
        # This helps prevent oscillations around the boundary
        safe_factor = 0.95
        scale_factor = jnp.where(
            archetype_projection > max_projection * safe_factor,
            safe_factor * max_projection / (archetype_projection + 1e-10),
            1.0,
        )

        # Scale the offset from centroid
        return centroid + scale_factor * (archetype - centroid)

    def visualize_movement(
        self,
        feature_indices: list[int] | None = None,
        figsize: tuple[int, int] = (12, 8),
        interval: int = 1,
    ):
        """
        Visualize how archetypes move during the optimization process.

        Args:
            feature_indices: For high-dimensional data, specify which 2 features to plot.
                If None and data has more than 2D, PCA is used.
            figsize: Size of the matplotlib figure
            interval: Interval between frames for animation

        Returns:
            Matplotlib figure and animation objects (if available)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except ImportError:
            self.logger.warning("Matplotlib required for visualization.")
            return None, None

        # Check if archetype history exists
        if not self.archetype_history:
            self.logger.warning("No archetype history available. Run fit() first.")
            return None, None

        if feature_indices is None:
            feature_indices = [0, 1]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Sample history (to reduce animation complexity)
        history_sample = self.archetype_history[::interval]
        n_iters = len(history_sample)

        # Check if we need dimensionality reduction
        if feature_indices is None and self.archetypes.shape[1] > 2:
            self.logger.info("Using PCA to visualize archetype movement...")
            try:
                from sklearn.decomposition import PCA

                # Stack all archetypes from all iterations
                all_archetypes = np.vstack(list(history_sample))

                # Apply PCA
                pca = PCA(n_components=2)
                all_archetypes_2d = pca.fit_transform(all_archetypes)

                # Reshape back to iterations x archetypes x 2
                archetype_coords = all_archetypes_2d.reshape(n_iters, self.n_archetypes, 2)

                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
            except ImportError:
                self.logger.warning("Skipping PCA visualization due to missing dependencies.")
                archetype_coords = np.array([arch[:, feature_indices] for arch in history_sample])
                ax.set_xlabel(f"Feature {feature_indices[0]}")
                ax.set_ylabel(f"Feature {feature_indices[1]}")
        else:
            # Use specified feature indices
            archetype_coords = np.array([arch[:, feature_indices] for arch in history_sample])
            ax.set_xlabel(f"Feature {feature_indices[0]}")
            ax.set_ylabel(f"Feature {feature_indices[1]}")

        # Plot trajectory of each archetype
        for a in range(self.n_archetypes):
            # Extract trajectory for this archetype
            trajectory = archetype_coords[:, a, :]

            # Plot each segment with color based on iteration
            for i in range(1, n_iters):
                ax.plot(
                    trajectory[i - 1 : i + 1, 0],
                    trajectory[i - 1 : i + 1, 1],
                    color="blue",
                    alpha=0.7,
                    linewidth=2,
                )

            # Mark starting point
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                color="blue",
                s=100,
                marker="o",
                label=f"Start A{a + 1}" if a == 0 else f"A{a + 1}",
            )

            # Mark final position
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color="red",
                s=100,
                marker="*",
                label=f"Final A{a + 1}" if a == 0 else "",
            )

        # Add colorbar to show iteration progress
        sm = plt.cm.ScalarMappable(cmap="viridis")
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Optimization Progress (Iterations)")

        # Add loss curve as inset
        loss_ax = fig.add_axes((0.15, 0.15, 0.25, 0.25))  # [left, bottom, width, height]
        loss_ax.plot(range(0, n_iters * interval, interval), self.loss_history[::interval], "k-")
        loss_ax.set_title("Loss Function")
        loss_ax.set_xlabel("Iterations")
        loss_ax.set_ylabel("Loss")

        # Add title and legend
        ax.set_title("Archetype Movement During Optimization")
        ax.legend(loc="upper right")

        plt.tight_layout()

        # Create animation
        def update(frame):
            ax.clear()
            ax.set_title(f"Iteration {frame * interval}")
            ax.scatter(
                archetype_coords[frame, :, 0],
                archetype_coords[frame, :, 1],
                c=np.arange(self.n_archetypes),
                cmap="viridis",
                s=100,
                marker="o",
            )
            ax.set_xlabel(f"Feature {feature_indices[0]}" if feature_indices else "PCA Component 1")
            ax.set_ylabel(f"Feature {feature_indices[1]}" if feature_indices else "PCA Component 2")
            ax.legend(loc="upper right")
            ax.set_aspect("equal")

            # Update loss curve
            loss_ax.clear()
            loss_ax.plot(range(0, (frame + 1) * interval, interval), self.loss_history[: (frame + 1) * interval], "k-")
            loss_ax.set_title("Loss Function")
            loss_ax.set_xlabel("Iterations")
            loss_ax.set_ylabel("Loss")

        anim = FuncAnimation(fig, update, frames=range(n_iters), repeat=False)

        return fig, anim

    def visualize_boundary_proximity(self, figsize=(10, 5)):
        """Visualize how close archetypes are to the convex hull boundary.

        Args:
            figsize: Size of the figure

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("Matplotlib required for visualization.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot boundary proximity
        ax1.plot(self.boundary_proximity_history)
        ax1.set_title("Boundary Proximity Score")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Score")
        ax1.grid(True)

        # Plot count of archetypes outside convex hull
        outside_counts = [np.sum(outside) for outside in self.is_outside_history]
        ax2.plot(outside_counts, "r-")
        ax2.set_title("Archetypes Outside Convex Hull")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Count")
        ax2.set_ylim(0, self.n_archetypes)
        ax2.grid(True)

        plt.tight_layout()
        return fig

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, archetypes, weights, X):
        """Customized loss function with reduced boundary incentive for ArchetypeTracker.

        This overrides the parent class loss function to provide more stability during tracking.
        Specifically, it reduces the boundary incentive weight to prevent archetypes from
        moving too rapidly in early iterations.
        """
        archetypes_f32 = archetypes.astype(jnp.float32)
        weights_f32 = weights.astype(jnp.float32)
        X_f32 = X.astype(jnp.float32)

        X_reconstructed = jnp.matmul(weights_f32, archetypes_f32)
        reconstruction_loss = jnp.mean(jnp.sum((X_f32 - X_reconstructed) ** 2, axis=1))

        # Calculate entropy (higher values for uniform weights, lower for sparse weights)
        entropy = -jnp.sum(weights_f32 * jnp.log(weights_f32 + 1e-10), axis=1)
        entropy_reg = jnp.mean(entropy)

        # Add incentive for archetypes to stay near convex hull boundary
        # But use a much lower weight than the parent class
        boundary_incentive = self._calculate_boundary_proximity(archetypes_f32, X_f32)

        # We use a significantly reduced boundary incentive for tracking stability
        # This matches the parent class boundary incentive level
        total_loss = reconstruction_loss + self.lambda_reg * entropy_reg - 0.001 * boundary_incentive

        return total_loss.astype(jnp.float32)

    def project_archetypes_with_adaptive_strength(self, archetypes, X):
        """Modified projection function that adapts its strength based on the current iteration.

        In early iterations, the projection is very gentle to prevent large movements.
        As training progresses, it gradually increases to the normal projection strength.
        """
        # Find the centroid of the data
        centroid = jnp.mean(X, axis=0)

        # Adaptation factor that increases from 0.1 to 1.0 over the first 100 iterations
        # For stability during tracking
        adapt_factor = jnp.minimum(0.1 + self.current_iteration * 0.009, 1.0)

        def _project_to_boundary(archetype):
            # Direction from centroid to archetype
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)

            # Avoid division by zero
            normalized_direction = jnp.where(
                direction_norm > 1e-10, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Project all points onto this direction
            projections = jnp.dot(X - centroid, normalized_direction)

            # Find the most extreme point in this direction
            max_idx = jnp.argmax(projections)
            extreme_point = X[max_idx]

            # Calculate the projection of the extreme point onto the direction
            extreme_projection = jnp.dot(extreme_point - centroid, normalized_direction)
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

            # Ensure the archetype doesn't go beyond the extreme point
            # If archetype is already outside, pull it back inside
            is_outside = archetype_projection > extreme_projection

            # Even more conservative projection for tracking
            # Scaled by the adaptation factor that increases with iterations
            adaptive_alpha = adapt_factor * jnp.minimum(0.15, self.projection_alpha * 1.5)

            # Different blending strategy depending on whether archetype is inside or outside
            projected = jnp.where(
                is_outside,
                # If outside, interpolate back toward boundary (more conservative than parent)
                0.4 * extreme_point + 0.6 * archetype,
                # If inside, move very gently toward boundary
                adaptive_alpha * extreme_point + (1 - adaptive_alpha) * archetype,
            )

            return projected

        # Apply the projection to each archetype
        projected_archetypes = jax.vmap(_project_to_boundary)(archetypes)

        return jnp.asarray(projected_archetypes)

    @partial(jax.jit, static_argnums=(0,))
    def project_archetypes(self, archetypes, X):
        """Override parent class projection with adaptive version for tracking."""
        return self.project_archetypes_with_adaptive_strength(archetypes, X)
