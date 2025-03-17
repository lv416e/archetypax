"""Improved Archetypal Analysis model using JAX."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

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
        projection_method: str = "cbap",
        projection_alpha: float = 0.2,
        lambda_reg: float = 0.01,
    ):
        """Initialize the Improved Archetypal Analysis model.

        Args:
            n_archetypes: Number of archetypes to find
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            random_seed: Random seed for initialization
            learning_rate: Learning rate for optimization
            projection_method: Method for projecting archetypes
            projection_alpha: Weight for extreme point
            lambda_reg: Regularization parameter
        """
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.projection_method = (
            projection_method if projection_method != "cbap" or projection_method != "default" else "default"
        )
        self.projection_alpha = projection_alpha
        self.lambda_reg = lambda_reg
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            learning_rate=learning_rate,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data to archetype weights using JAX."""
        if self.archetypes is None:
            raise ValueError("Model must be fitted before transform")

        # Scale input data
        X_scaled = (X - self.X_mean) / self.X_std if self.X_mean is not None and self.X_std is not None else X

        # Convert to JAX array
        X_jax = jnp.array(X_scaled)

        # Scale archetypes
        archetypes_scaled = (
            (self.archetypes - self.X_mean) / self.X_std
            if self.X_mean is not None and self.X_std is not None
            else self.archetypes
        )
        archetypes_jax = jnp.array(archetypes_scaled)

        # Define per-sample optimization in JAX
        @jax.jit
        def optimize_weights(x_sample):
            # Initialize weights uniformly
            w = jnp.ones(self.n_archetypes) / self.n_archetypes

            # Define a single gradient step
            def step(w, _):
                pred = jnp.dot(w, archetypes_jax)
                error = x_sample - pred
                grad = -2 * jnp.dot(error, archetypes_jax.T)

                # Update with gradient
                w_new = w - 0.01 * grad

                # Project to constraints
                w_new = jnp.maximum(1e-10, w_new)  # Non-negativity with small epsilon
                sum_w = jnp.sum(w_new)
                # Avoid division by zero
                w_new = jnp.where(
                    sum_w > 1e-10,
                    w_new / sum_w,
                    jnp.ones_like(w_new) / self.n_archetypes,
                )

                return w_new, None

            # Run 100 steps
            final_w, _ = jax.lax.scan(step, w, jnp.arange(100))
            return final_w

        # Vectorize the optimization across all samples
        batch_optimize = jax.vmap(optimize_weights)
        weights_jax = batch_optimize(X_jax)

        return np.array(weights_jax)

    def transform_with_lbfgs(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using improved optimization for better convergence."""
        if self.archetypes is None:
            raise ValueError("Model must be fitted before transform")

        X_scaled = (X - self.X_mean) / self.X_std if self.X_mean is not None and self.X_std is not None else X

        X_jax = jnp.array(X_scaled)

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
        def grad_fn(w, x):
            return jax.grad(lambda w: objective(w, x))(w)

        @jax.jit
        def project_to_simplex(w):
            w = jnp.maximum(1e-10, w)
            sum_w = jnp.sum(w)
            # Avoid division by zero
            return jnp.where(sum_w > 1e-10, w / sum_w, jnp.ones_like(w) / self.n_archetypes)

        @jax.jit
        def optimize_single_sample(x):
            w_init = jnp.ones(self.n_archetypes) / self.n_archetypes

            optimizer = optax.adam(learning_rate=0.05)
            opt_state = optimizer.init(w_init)

            def step(state, _):
                w, opt_state = state
                loss_val, grad = jax.value_and_grad(lambda w: objective(w, x))(w)
                grad = jnp.clip(grad, -1.0, 1.0)
                updates, opt_state = optimizer.update(grad, opt_state)
                w = optax.apply_updates(w, updates)
                w = project_to_simplex(w)
                return (w, opt_state), loss_val

            (final_w, _), _ = jax.lax.scan(step, (w_init, opt_state), jnp.arange(50))

            return final_w

        batch_size = 1000
        n_samples = X_jax.shape[0]
        weights = []
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X_jax[i:end]

            batch_weights = jax.vmap(optimize_single_sample)(X_batch)
            weights.append(batch_weights)

        weights_jax = weights[0] if len(weights) == 1 else jnp.concatenate(weights, axis=0)

        return np.array(weights_jax)

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
            # Generate initial directions randomly
            self.key, subkey = jax.random.split(self.key)
            directions = jax.random.normal(subkey, (self.n_archetypes, n_features))

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

        return archetypes, jnp.zeros(self.n_archetypes, dtype=jnp.int32)

    def qhull_init(self, X_jax, n_samples, n_features):
        """Initialize archetypes using convex hull vertices via QHull algorithm."""
        # Convert to numpy for scipy compatibility
        X_np = np.array(X_jax)

        try:
            # Compute the convex hull using scipy's implementation of QHull
            from scipy.spatial import ConvexHull

            print("Computing convex hull...")
            hull = ConvexHull(X_np)
        except Exception as e:
            print(f"QHull initialization failed: {e}. Falling back to k-means++ initialization.")
            return self.kmeans_pp_init(X_jax, n_samples, n_features)

        # Get the vertices of the convex hull
        vertices = hull.vertices
        vertex_points = X_np[vertices]

        # If we have more vertices than required archetypes, select a subset
        if len(vertices) > self.n_archetypes:
            # Strategy 1: Farthest point sampling
            selected_indices = [0]  # Start with the first vertex

            for _ in range(self.n_archetypes - 1):
                # Compute distances to already selected points
                distances = []
                for i in range(len(vertex_points)):
                    if i not in selected_indices:
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

            # Sample additional points randomly
            self.key, subkey = jax.random.split(self.key)
            additional_indices = jax.random.choice(subkey, n_samples, shape=(remaining,), replace=False, p=None)
            selected_vertices.extend(additional_indices)
        else:
            # Perfect! We have exactly the right number of vertices
            selected_vertices = vertices

        # Use the selected vertices as initial archetypes
        archetypes = jnp.array(X_np[selected_vertices])

        return archetypes, jnp.array(selected_vertices)

    def kmeans_pp_init(self, X_jax, n_samples, n_features):
        """More efficient k-means++ style initialization using JAX."""
        # Randomly select the first center
        self.key, subkey = jax.random.split(self.key)
        first_idx = jax.random.randint(subkey, (), 0, n_samples)

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

            self.key, subkey = jax.random.split(self.key)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)

            # Update selected indices and centers
            chosen_indices = chosen_indices.at[i].set(next_idx)
            archetypes = archetypes.at[i].set(X_jax[next_idx])

        return archetypes, chosen_indices

    def fit(self, X: np.ndarray, normalize: bool = False) -> "ImprovedArchetypalAnalysis":
        """Fit the model with improved k-means++ initialization."""

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

            new_params["weights"] = self.project_weights(new_params["weights"])
            new_params["archetypes"] = jax.lax.cond(
                self.projection_method == "cbap" or self.projection_method == "default",
                lambda: self.project_archetypes(new_params["archetypes"], X_f32),
                lambda: jax.lax.cond(
                    self.projection_method == "convex_hull",
                    lambda: self.project_archetypes_convex_hull(new_params["archetypes"], X_f32),
                    lambda: self.project_archetypes_knn(new_params["archetypes"], X_f32),
                ),
            )
            # new_params["archetypes"] = self.project_archetypes_convex_hull(new_params["archetypes"], X_f32)
            # new_params["archetypes"] = self.project_archetypes_knn(new_params["archetypes"], X_f32)
            # new_params["archetypes"] = self.update_archetypes(new_params["archetypes"], new_params["weights"], X)

            new_params = jax.tree.map(lambda p: p.astype(jnp.float32), new_params)
            # new_params = jax.tree.map(lambda p: p.astype(jnp.float16), new_params)

            return new_params, opt_state, loss

        # Preprocess data: scale for improved stability
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        # Prevent division by zero with explicit type casting
        if self.X_std is not None:
            self.X_std = np.where(self.X_std < 1e-10, np.ones_like(self.X_std), self.X_std)
        X_scaled = (X - self.X_mean) / self.X_std if normalize else X.copy()

        # Convert to JAX array
        X_jax = jnp.array(X_scaled, dtype=jnp.float32)
        # X_jax = jnp.array(X_scaled, dtype=jnp.float16)
        n_samples, n_features = X_jax.shape

        # Convert to JAX array
        X_jax = jnp.array(X_scaled, dtype=jnp.float32)
        # X_jax = jnp.array(X_scaled, dtype=jnp.float16)
        n_samples, n_features = X_jax.shape

        # Debug information
        print(f"Data shape: {X_jax.shape}")
        print(f"Data range: min={float(jnp.min(X_jax)):.4f}, max={float(jnp.max(X_jax)):.4f}")

        # Initialize weights (more stable initialization)
        self.key, subkey = jax.random.split(self.key)
        weights_init = jax.random.uniform(
            subkey,
            (n_samples, self.n_archetypes),
            minval=0.1,
            maxval=0.9,
            dtype=jnp.float32,
            # dtype=jnp.float16,
        )
        weights_init = self.project_weights(weights_init)

        # archetype initialization
        # archetypes_init, _ = self.kmeans_pp_init(X_jax, n_samples, n_features)
        # archetypes_init, _ = self.qhull_init(X_jax, n_samples, n_features)
        archetypes_init, _ = self.directional_init(X_jax, n_samples, n_features)
        archetypes_init = archetypes_init.astype(jnp.float32)
        # archetypes_init = archetypes_init.astype(jnp.float16)

        # The rest is the same as the original fit method
        # Set up optimizer (Adam with reduced learning rate)
        optimizer: optax.GradientTransformation = optax.adam(learning_rate=self.learning_rate)

        # Initialize parameters
        params = {"archetypes": archetypes_init, "weights": weights_init}
        opt_state = optimizer.init(params)

        # Optimization loop
        prev_loss = float("inf")
        self.loss_history = []

        # Calculate initial loss for debugging
        initial_loss = float(self.loss_function(archetypes_init, weights_init, X_jax))
        print(f"Initial loss: {initial_loss:.6f}")

        for i in range(self.max_iter):
            # Execute update step
            try:
                params, opt_state, loss = update_step(params, opt_state, X_jax, i)
                loss_value = float(loss)

                # Check for NaN
                if jnp.isnan(loss_value):
                    print(f"Warning: NaN detected at iteration {i}. Stopping early.")
                    break

                # Record loss
                self.loss_history.append(loss_value)

                # Check convergence
                if i > 0 and abs(prev_loss - loss_value) < self.tol:
                    print(f"Converged at iteration {i}")
                    break

                prev_loss = loss_value

                # Show progress
                if i % 50 == 0:
                    print(f"Iteration {i}, Loss: {loss_value:.6f}")

            except Exception as e:
                print(f"Error at iteration {i}: {e!s}")
                break

        # Inverse scale transformation
        archetypes_scaled = np.array(params["archetypes"])
        self.archetypes = archetypes_scaled * self.X_std + self.X_mean
        self.weights = np.array(params["weights"])

        if len(self.loss_history) > 0:
            print(f"Final loss: {self.loss_history[-1]:.6f}")
        else:
            print("Warning: No valid loss was recorded")

        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, normalize: bool = False) -> np.ndarray:
        """
        Fit the model and return the transformed data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            y: Ignored. Present for API consistency by convention.
            normalize: Whether to normalize the data before fitting.

        Returns:
            Weight matrix representing each sample as a combination of archetypes
        """
        model = self.fit(X, normalize=normalize)
        return np.asarray(model.transform(X))

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
          a weighted average (80% extreme point, 20% original archetype) to prevent abrupt
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

            # Mix the extreme point with the original archetype to ensure stability
            # Use a higher weight for the extreme point to push towards the boundary
            projected = self.projection_alpha * X[max_idx] + (1 - self.projection_alpha) * archetype

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
        def _find_extreme_points(archetype):
            # Generate multiple random directions around the archetype direction
            key = jax.random.key(0)  # Fixed seed for deterministic behavior
            n_directions = 5

            # Direction from centroid to archetype
            main_direction = archetype - centroid
            main_direction_norm = jnp.linalg.norm(main_direction)

            # Avoid division by zero
            main_direction = jnp.where(
                main_direction_norm > 1e-10,
                main_direction / main_direction_norm,
                jax.random.normal(key, shape=main_direction.shape),
            )

            # Generate random perturbations of the main direction
            key, subkey = jax.random.split(key)
            perturbations = jax.random.normal(subkey, shape=(n_directions, main_direction.shape[0]))

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

            return projected

        # Apply the projection to each archetype
        projected_archetypes = jax.vmap(_find_extreme_points)(archetypes)

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

        entropy = -jnp.sum(weights_f32 * jnp.log(weights_f32 + 1e-10), axis=1)
        entropy_reg = -jnp.mean(entropy)

        return (reconstruction_loss + self.lambda_reg * entropy_reg).astype(jnp.float32)
        # return (reconstruction_loss + lambda_reg * entropy_reg).astype(jnp.float16)

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
        """Alternative archetype update strategy based on weighted reconstruction."""
        W_pinv = jnp.linalg.pinv(weights)
        return jnp.array(jnp.matmul(W_pinv, X))


class ArchetypeTracker(ImprovedArchetypalAnalysis):
    """A specialized subclass designed to monitor the movement of archetypes."""

    def __init__(self, *args, **kwargs):
        """Initialize the ArchetypeTracker with parameters identical to those of ImprovedArchetypalAnalysis."""
        super().__init__(*args, **kwargs)
        self.archetype_history = []
        self.optimizer: optax.GradientTransformation = optax.adam(learning_rate=self.learning_rate)
        # Specific settings for archetype updates
        self.archetype_grad_scale = (
            2.0  # Gradient scale for archetypes (values greater than 1.0 enhance archetype updates)
        )
        self.noise_scale = 0.05  # Magnitude of initial noise
        self.exploration_noise_scale = 0.1  # Magnitude of exploration noise

    def fit(self, X: np.ndarray, normalize: bool = False) -> "ArchetypeTracker":
        """Train the model while documenting the positions of archetypes at each iteration."""
        # Data preprocessing
        if normalize:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            X = (X - self.X_mean) / self.X_std

        # Convert to JAX array
        X_jax = jnp.array(X)
        n_samples, n_features = X_jax.shape

        # JIT-compiled weight calculation function
        @jax.jit
        def calculate_single_weight(x_sample, archetypes):
            """Compute the weight for a single sample (JIT optimized)."""
            # Initial weights
            w = jnp.ones(self.n_archetypes) / self.n_archetypes

            # Define the internal loop state
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

            # Use lax.scan to optimize iterative calculations (instead of a for loop)
            final_w, _ = jax.lax.scan(weight_update_step, w, None, length=100)
            return final_w

        # Use vmap to parallelize weight calculations for all samples
        calculate_all_weights = jax.vmap(calculate_single_weight, in_axes=(0, None))

        # Initialize archetypes and weights
        _, subkey = jax.random.split(jax.random.key(self.random_seed))
        archetypes, _ = self.directional_init(X_jax, n_samples, n_features)

        # Introduce a small noise after archetype initialization to encourage exploration
        self.key, subkey = jax.random.split(self.key)
        noise = jax.random.normal(subkey, archetypes.shape) * self.noise_scale
        archetypes = archetypes + noise

        # Project again after adding noise to return to the convex hull boundary
        archetypes = self.project_archetypes(archetypes, X_jax)

        # Store initial archetypes
        self.archetype_history = [np.array(archetypes)]

        # Set up optimization
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = self.optimizer.init(archetypes)

        # Optimization loop
        prev_loss = float("inf")
        self.loss_history = []
        prev_archetypes = archetypes.copy()  # Store previous archetypes for monitoring changes

        # Variables to store the best parameters throughout optimization
        best_archetypes = archetypes.copy()
        best_loss = float("inf")
        no_improvement_count = 0

        # Calculate initial loss for debugging
        initial_loss = float(self.loss_function(archetypes, None, X_jax))
        print(f"Initial loss: {initial_loss:.6f}")

        for i in range(self.max_iter):
            # Execute update step
            try:
                archetypes, opt_state, loss_value = self._update_step(archetypes, opt_state, X_jax, i)

                # Store current archetypes
                self.archetype_history.append(np.array(archetypes))

                # Check for NaN
                if jnp.isnan(loss_value):
                    print(f"Warning: NaN detected at iteration {i}. Stopping early.")
                    break

                # Record loss
                self.loss_history.append(loss_value)

                # Update best parameters if the current loss is lower
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_archetypes = archetypes.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Revert to best parameters if loss increases significantly
                # Use JAX-style functional approach for parameter restoration
                def make_restore_best_params(iteration, best_archetypes_val, best_loss_val):
                    def restore_best_params(_):
                        print(
                            f"Warning: Loss increased significantly at iteration {iteration}. Reverting to best parameters."
                        )
                        restored_params = {
                            "archetypes": best_archetypes_val.copy(),
                            "weights": None,  # Assuming weights are not updated in this method
                        }
                        return restored_params, best_loss_val

                    return restore_best_params

                def make_keep_current_params(archetypes_val, loss_val):
                    def keep_current_params(_):
                        return archetypes_val, loss_val

                    return keep_current_params

                # Only revert if loss increased significantly and not in early stages
                should_revert = (loss_value > prev_loss * 1.05) & (i > 50)
                restore_fn = make_restore_best_params(i, best_archetypes, best_loss)
                keep_fn = make_keep_current_params(archetypes, loss_value)
                archetypes, loss_value = jax.lax.cond(should_revert, restore_fn, keep_fn, operand=None)

                # Reduce learning rate after extended periods without improvement
                if no_improvement_count >= 100 and no_improvement_count % 100 == 0:
                    print(f"No improvement for {no_improvement_count} iterations. Reducing learning rate.")
                    # Create new optimizer with reduced learning rate
                    reduced_lr = self.learning_rate * 0.5
                    self.optimizer = optax.adam(learning_rate=reduced_lr)
                    opt_state = self.optimizer.init(archetypes)

                # Check for convergence
                if i > 0 and abs(prev_loss - loss_value) < self.tol:
                    print(f"Converged at iteration {i}")
                    break

                prev_loss = loss_value

                # Show progress and monitor archetype changes
                if i % 50 == 0:
                    archetype_change = jnp.mean(jnp.sum((archetypes - prev_archetypes) ** 2, axis=1))
                    print(f"Iteration {i}, Loss: {loss_value:.6f}, Avg archetype change: {float(archetype_change):.6f}")
                    prev_archetypes = archetypes.copy()

                    # Exploratory movement: occasionally perturb the archetypes randomly
                    if i > 0 and i % 200 == 0 and archetype_change < 0.001:
                        print("Introducing exploration noise to the archetypes...")

                        # Store loss and parameters before adding noise
                        pre_noise_loss = loss_value
                        pre_noise_archetypes = archetypes.copy()

                        # Add exploration noise
                        self.key, subkey = jax.random.split(self.key)
                        exploration_noise = jax.random.normal(subkey, archetypes.shape) * self.exploration_noise_scale
                        archetypes = archetypes + exploration_noise
                        archetypes = self.project_archetypes(archetypes, X_jax)

                        # Calculate loss after adding noise
                        post_noise_loss = float(self.loss_function(archetypes, None, X_jax))

                        # Revert noise if loss increases significantly using JAX-style conditionals
                        def make_revert_noise(pre_loss, post_loss, pre_archetypes):
                            def revert_noise(_):
                                print(
                                    f"Noise increased loss significantly ({pre_loss:.6f} -> {post_loss:.6f}). Reverting."
                                )
                                return pre_archetypes

                            return revert_noise

                        def make_keep_noise(current_archetypes):
                            def keep_noise(_):
                                return current_archetypes

                            return keep_noise

                        # Check if noise should be reverted (loss increased by more than 10%)
                        should_revert_noise = post_noise_loss > pre_noise_loss * 1.1
                        revert_fn = make_revert_noise(pre_noise_loss, post_noise_loss, pre_noise_archetypes)
                        keep_fn = make_keep_noise(archetypes)
                        archetypes = jax.lax.cond(should_revert_noise, revert_fn, keep_fn, operand=None)

            except Exception as e:
                print(f"Error at iteration {i}: {e!s}")
                break

        # Use the best parameters for the final model if they outperform the current ones
        # JAX-style conditional for final parameter selection
        def make_use_best_params(best_archetypes_val, best_loss_val):
            def use_best_params(_):
                print(f"Utilizing best parameters with loss: {best_loss_val:.6f}")
                return best_archetypes_val

            return use_best_params

        def make_use_current_params(current_archetypes):
            def use_current_params(_):
                return current_archetypes

            return use_current_params

        # Check if the best parameters yield a lower loss
        current_final_loss = float(self.loss_function(archetypes, None, X_jax))
        use_best = best_loss < current_final_loss
        use_best_fn = make_use_best_params(best_archetypes, best_loss)
        use_current_fn = make_use_current_params(archetypes)
        archetypes = jax.lax.cond(use_best, use_best_fn, use_current_fn, operand=None)

        # Calculate the final weights
        weights = calculate_all_weights(X_jax, archetypes)

        # Inverse scale transformation
        archetypes_scaled = np.array(archetypes)
        self.archetypes = archetypes_scaled * self.X_std + self.X_mean if normalize else archetypes_scaled
        self.weights = np.array(weights)  # Store weights
        self.archetype_history = [
            arch * self.X_std + self.X_mean if normalize else arch for arch in self.archetype_history
        ]

        return self

    def _update_step(self, archetypes, opt_state, X, iteration):
        """Execute one iteration of the update step with JAX acceleration."""

        # JIT-compiled weight calculation function
        @jax.jit
        def calculate_single_weight(x_sample, archetypes):
            """Compute the weight for a single sample (JIT optimized)."""
            # Initial weights
            w = jnp.ones(self.n_archetypes) / self.n_archetypes

            # Define the internal loop state
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

            # Use lax.scan to optimize iterative calculations (instead of a for loop)
            final_w, _ = jax.lax.scan(weight_update_step, w, None, length=100)
            return final_w

        # Use vmap to parallelize weight calculations for all samples
        calculate_all_weights = jax.vmap(calculate_single_weight, in_axes=(0, None))
        weights = calculate_all_weights(X, archetypes)  # Store the results of weight calculations

        # Define the loss function (with archetypes as the only variable)
        def loss_fn(arch):
            return self.loss_function(arch, weights, X)

        # Calculate loss and gradients
        loss_value, grads = jax.value_and_grad(loss_fn)(archetypes)

        # Adjust and clip the gradient scale
        grads = grads * self.archetype_grad_scale
        grads = jnp.clip(grads, -1.0, 1.0)

        # Execute the optimization step
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized")

        updates, opt_state = self.optimizer.update(grads, opt_state)
        archetypes = optax.apply_updates(archetypes, updates)

        # Store the current loss
        current_loss = loss_value

        # Alternating optimization: strongly update archetypes in specific iterations
        do_enhanced_update = jnp.mod(iteration, 3) == 0

        # Preserve archetypes before projection
        pre_projection_archetypes = archetypes.copy()

        # Define projection functions for JAX conditionals
        def apply_enhanced_update():
            return self._enhanced_archetype_update(archetypes, X, iteration)

        def apply_standard_projection():
            # Nested conditional for standard projection methods
            def project_default():
                return self.project_archetypes(archetypes, X)

            def project_random():
                return self.project_archetypes_random(archetypes, X)

            def project_knn():
                return self.project_archetypes_knn(archetypes, X)

            # Select projection method using JAX conditionals
            return jax.lax.cond(
                self.projection_method == "random",
                project_random,
                lambda: jax.lax.cond(self.projection_method == "knn", project_knn, project_default),
            )

        # Apply projection with JAX conditional
        projected_archetypes = jax.lax.cond(do_enhanced_update, apply_enhanced_update, apply_standard_projection)

        # Calculate loss after projection
        post_projection_loss = float(self.loss_function(projected_archetypes, weights, X))

        # If loss increases, reduce the projection intensity
        if post_projection_loss > current_loss * 1.05:  # Allow for an increase of more than 5%
            # Blend to reduce projection intensity
            blend_factor = 0.3  # Adjust projection intensity
            archetypes = (1 - blend_factor) * pre_projection_archetypes + blend_factor * projected_archetypes
            loss_value = float(self.loss_function(archetypes, weights, X))
        else:
            archetypes = projected_archetypes
            loss_value = post_projection_loss

        return archetypes, opt_state, loss_value
