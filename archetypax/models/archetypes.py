"""Improved Archetypal Analysis using JAX.

This module extends the base ArchetypalAnalysis class with enhanced optimization strategies
and boundary projection techniques.
The ImprovedArchetypalAnalysis class provides a more robust and versatile implementation
of Archetypal Analysis (AA) using JAX for GPU acceleration.

The improvements focus on:
- Multiple initialization strategies (directional, convex hull, kmeans++)
- Advanced optimization with hybrid gradient and direct update methods
- Adaptive boundary projection techniques
- Better convergence stability through regularization

Key advantages over the base implementation:
- More stable convergence across diverse datasets
- Higher quality solutions with improved boundary placement
- Richer configuration options for domain-specific tuning
- Enhanced computational efficiency for large-scale applications

Example usage:
    ```python
    from archetypax.models import ImprovedArchetypalAnalysis

    # Initialize model
    model = ImprovedArchetypalAnalysis(
        n_archetypes=5,
        normalize=True,
        archetype_init_method="directional",
        projection_method="cbap"
    )

    # Fit model and transform data
    weights = model.fit_transform(X)

    # Extract discovered archetypes
    archetypes = model.archetypes
    ```
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from archetypax.logger import get_logger, get_message
from archetypax.models.base import ArchetypalAnalysis


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
        """Initialize an enhanced archetypal analysis model with robust optimization.

        This improved implementation addresses key limitations of standard archetypal
        analysis through advanced initialization strategies, robust gradient-based
        optimization, and adaptive boundary projection techniques. These enhancements
        significantly improve convergence stability, solution quality, and computational
        efficiency across diverse datasets.

        Args:
            n_archetypes:
                Number of archetypes to discover - controls the model's
                expressiveness and dimensionality reduction ratio.
                Higher values capture more nuanced patterns
                at the cost of interpretability and potential overfitting.

            max_iter:
                Maximum optimization iterations - higher values ensure better
                convergence at the cost of computational time.
                The default (500) balances solution quality with reasonable runtime for most datasets.

            tol:
                Convergence tolerance - smaller values yield more precise solutions
                but require more iterations. The default (1e-6) is suitable for most
                applications, while scientific applications may require smaller values.

            random_seed:
                Random seed for reproducibility - ensures consistent results
                across runs with the same data and parameters.

            learning_rate:
                Gradient descent step size - critical parameter balancing
                convergence speed with stability.
                Too high risks overshooting minima, while too low causes slow convergence.

            lambda_reg:
                Regularization strength for weights - controls the balance
                between reconstruction accuracy and weight sparsity.
                Higher values promote more discrete archetype assignments.

            normalize:
                Whether to normalize features - essential when features have
                different scales to prevent dominance by high-magnitude features.
                Should be True for most real-world datasets.

            projection_method: Strategy for projecting archetypes to boundary:
                - "cbap" (default):
                    Convex boundary approximation projection
                    - balanced approach suitable for most datasets
                - "convex_hull":
                    Uses exact convex hull vertices
                    - more precise but computationally intensive for high dimensions
                - "knn":
                    K-nearest neighbors approximation
                    - faster for large datasets

            projection_alpha:
                Projection strength parameter (0-1)
                - controls how aggressively archetypes are pushed toward the boundary.
                Higher values emphasize extremeness over reconstruction.

            archetype_init_method:
                Initialization strategy for archetypes:
                - "directional" (default):
                    Directions from centroid
                    - robust general-purpose approach that balances diversity with boundary alignment
                - "qhull"/"convex_hull":
                    Exact convex hull vertices
                    - ideal when geometric extremes are well-defined
                - "kmeans"/"kmeans++":
                    K-means++ initialization
                    - beneficial when density-based initialization aligns with domain expectations

            **kwargs: Additional parameters:
                - early_stopping_patience: Iterations without improvement before stopping
                - verbose_level: Controls logging detail (0-4)
                    - 0: Critical only
                    - 1: Error level
                    - 2: Warning level
                    - 3: Info level (recommended for monitoring)
                    - 4: Debug level (verbose training details)
                - logger_level: Alternative to verbose_level with reversed mapping
        """
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            learning_rate=learning_rate,
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
        elif "logger_level" not in kwargs and "verbose_level" in kwargs and kwargs.get("verbose_level") is not None:
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
                max_iter=max_iter,
                tol=tol,
                random_seed=random_seed,
                learning_rate=learning_rate,
                lambda_reg=lambda_reg,
                normalize=normalize,
                projection_method=projection_method,
                projection_alpha=projection_alpha,
                archetype_init_method=archetype_init_method,
            )
        )

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

        self.eps = jnp.finfo(jnp.float32).eps

        self.early_stopping_patience = kwargs.get("early_stopping_patience", 100)

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Transform data into archetypal weight space with adaptive optimization.

        This method computes optimal weights representing each sample as a convex
        combination of discovered archetypes. The transformation reveals how samples
        relate to extreme patterns, offering:

        1. Dimensionality reduction while preserving interpretability
        2. Soft clustering based on meaningful archetypes rather than arbitrary centroids
        3. Insights into sample composition and relationship to extreme patterns
        4. A foundation for transfer learning when applying archetypes to new data

        Multiple optimization strategies are available, with adaptive selection based
        on dataset size to balance computational efficiency with solution quality.

        Args:
            X: Data matrix to transform (n_samples, n_features)
            y: Ignored, present for scikit-learn API compatibility
            **kwargs: Additional parameters:
                - method: Optimization approach to use:
                    - "lbfgs": Best for small datasets (<1000 samples)
                    - "adam": Balanced option for mid-sized data (default)
                    - "sgd": Memory-efficient for large datasets
                    - "adaptive": Automatically selects based on data size
                - max_iter: Maximum iterations for weight optimization
                - tol: Convergence tolerance (smaller values for more precision)

        Returns:
            Weight matrix representing each sample as a combination of the
            discovered archetypes (n_samples, n_archetypes)
        """
        if self.archetypes is None:
            raise ValueError("Model must be fitted before transform")

        method = kwargs.get("method", "adam")
        max_iter = kwargs.get("max_iter", self.max_iter)
        tol = kwargs.get("tol", self.tol)

        X_np = X.values if hasattr(X, "values") else X
        X_jax = jnp.array(X_np, dtype=jnp.float32)
        if self.normalize:
            X_scaled = jnp.asarray(
                (X_jax - self.X_mean) / self.X_std
                if self.X_mean is not None and self.X_std is not None
                else X_np.copy()
            )
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            X_scaled = X_jax.copy()

        if method == "adaptive":
            n_samples = X.shape[0]
            if n_samples > 10000:
                method = "sgd"
            elif n_samples > 1000:
                method = "adam"
            else:
                method = "lbfgs"

        self.logger.info(get_message("data", "transformation", method=method))
        transform_fn = {
            "lbfgs": self._transform_with_lbfgs,
            "sgd": self._transform_with_sgd,
            "adam": self._transform_with_adam,
            "default": self._transform_with_adam,
        }.get(method, self._transform_with_adam)

        weights = transform_fn(X_jax=X_scaled, max_iter=max_iter, tol=tol)

        return np.asarray(weights)

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
        2. Simplified workflow for immediate archetypal representation

        This method is particularly valuable in analysis pipelines or when
        integrating with scikit-learn compatible frameworks that expect this
        pattern. It ensures that the transformation is performed with the
        same preprocessing settings used during fitting.

        Args:
            X: Data matrix to fit and transform (n_samples, n_features)
            y: Ignored, present for scikit-learn API compatibility
            normalize:
                Whether to normalize features before fitting
                - essential for data with different scales or magnitudes
            **kwargs:
                Additional parameters passed to both fit() and transform(),
                including optimization settings and convergence criteria

        Returns:
            Weight matrix representing each sample as a combination of the
            discovered archetypes (n_samples, n_archetypes)
        """
        X_np = X.values if hasattr(X, "values") else X.copy()
        model = self.fit(X_np, **kwargs)
        return np.asarray(model.transform(X_np, **kwargs))

    def _transform_with_lbfgs(self, X_jax: jnp.ndarray, max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
        """Transform new data using improved L-BFGS optimization.

        Args:
            X_jax: Data matrix of shape (n_samples, n_features)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            Weight matrix of shape (n_samples, n_archetypes)
        """
        if self.normalize:
            archetypes_scaled = (
                (self.archetypes - self.X_mean) / self.X_std
                if self.X_mean is not None and self.X_std is not None
                else self.archetypes
            )
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            archetypes_scaled = self.archetypes

        archetypes_jax = jnp.array(archetypes_scaled)

        @jax.jit
        def objective(w, x):
            pred = jnp.dot(w, archetypes_jax)
            return jnp.sum((x - pred) ** 2)

        @jax.jit
        def project_to_simplex(w):
            w = jnp.maximum(self.eps, w)
            sum_w = jnp.sum(w)
            return jnp.where(sum_w > self.eps, w / sum_w, jnp.ones_like(w) / self.n_archetypes)

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

                converged = jnp.abs(prev_loss - loss_val) < tol

                return (new_w, new_opt_state, loss_val, i + 1, converged)

            init_state = (w_init, opt_state, jnp.inf, 0, False)
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

            return final_state[0]

        batch_size = min(2000, X_jax.shape[0])
        n_samples = X_jax.shape[0]
        weights = []

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X_jax[i:end]
            batch_weights = jax.vmap(optimize_single_sample)(X_batch)
            weights.append(np.array(batch_weights))

        weights_array = np.vstack(weights) if len(weights) > 1 else weights[0]

        return np.asarray(weights_array)

    def _transform_with_adam(self, X_jax: jnp.ndarray, max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
        """Transform using Adam optimizer with early stopping.

        Args:
            X_jax: Data matrix of shape (n_samples, n_features)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            Weight matrix of shape (n_samples, n_archetypes)
        """
        if self.normalize:
            archetypes_scaled = (
                (self.archetypes - self.X_mean) / self.X_std
                if self.X_mean is not None and self.X_std is not None
                else self.archetypes
            )
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            archetypes_scaled = self.archetypes

        archetypes_jax = jnp.array(archetypes_scaled)

        @jax.jit
        def objective(w, x):
            pred = jnp.dot(w, archetypes_jax)
            return jnp.sum((x - pred) ** 2)

        @jax.jit
        def project_to_simplex(w):
            w = jnp.maximum(self.eps, w)
            sum_w = jnp.sum(w)
            return jnp.where(sum_w > self.eps, w / sum_w, jnp.ones_like(w) / self.n_archetypes)

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

                converged = jnp.abs(prev_loss - loss_val) < tol

                return (new_w, new_opt_state, loss_val, i + 1, converged)

            init_state = (w_init, opt_state, jnp.inf, 0, False)
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

            return final_state[0]

        batch_size = min(1000, X_jax.shape[0])
        n_samples = X_jax.shape[0]
        weights = []

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X_jax[i:end]

            batch_weights = jax.vmap(optimize_single_sample)(X_batch)
            weights.append(np.array(batch_weights))

        weights_array = np.vstack(weights) if len(weights) > 1 else weights[0]

        return np.asarray(weights_array)

    def _transform_with_sgd(self, X_jax: jnp.ndarray, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
        """Transform using improved SGD with adaptive learning rate and convergence criteria.

        Args:
            X_jax: Data matrix of shape (n_samples, n_features)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            Weight matrix of shape (n_samples, n_archetypes)
        """
        if self.normalize:
            archetypes_scaled = (
                (self.archetypes - self.X_mean) / self.X_std
                if self.X_mean is not None and self.X_std is not None
                else self.archetypes
            )
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            archetypes_scaled = self.archetypes

        archetypes_jax = jnp.array(archetypes_scaled)

        @jax.jit
        def optimize_weights_with_convergence(x_sample):
            w_init = jnp.ones(self.n_archetypes) / self.n_archetypes

            def cond_fun(state):
                _, _, i, converged = state
                return jnp.logical_and(jnp.logical_not(converged), i < max_iter)

            def body_fun(state):
                w, prev_loss, i, _ = state

                pred = jnp.dot(w, archetypes_jax)
                error = x_sample - pred
                loss = jnp.sum(error**2)

                converged = jnp.abs(prev_loss - loss) < tol

                lr = 0.01 / (1.0 + 0.005 * i)

                grad = -2 * jnp.dot(error, archetypes_jax.T)
                w_new = w - lr * grad

                w_new = jnp.maximum(self.eps, w_new)
                sum_w = jnp.sum(w_new)
                w_new = w_new / jnp.maximum(sum_w, self.eps)

                return (w_new, loss, i + 1, converged)

            init_state = (w_init, jnp.inf, 0, False)
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

            return final_state[0]

        batch_size = min(1000, X_jax.shape[0])
        n_samples = X_jax.shape[0]
        weights = []

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X_jax[i:end]

            batch_weights = jax.vmap(optimize_weights_with_convergence)(X_batch)
            weights.append(np.array(batch_weights))

        weights_array = np.vstack(weights) if len(weights) > 1 else weights[0]
        return np.asarray(weights_array)

    def directional_init(self, X_jax: jnp.ndarray, n_samples: int, n_features: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate directions using points that are evenly distributed on a sphere.

        Args:
            X_jax: Data matrix of shape (n_samples, n_features)
            n_samples: Number of samples
            n_features: Number of features

        Returns:
            Archetypes and archetype indices
        """
        centroid = jnp.mean(X_jax, axis=0)

        if n_features == 2:
            angles = jnp.linspace(0, 2 * jnp.pi, self.n_archetypes, endpoint=False)
            directions = jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])

        elif n_features == 3:
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
            self.rng_key, subkey = jax.random.split(self.rng_key)
            directions = jax.random.normal(subkey, (self.n_archetypes, n_features))

            # Normalize the direction vectors
            norms = jnp.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / (norms + self.eps)

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
                new_directions = new_directions / (norms + self.eps)

                return new_directions, None

            # Execute the repulsion simulation
            directions, _ = jax.lax.scan(repulsion_step, directions, jnp.arange(n_iterations))

        def find_extreme_point(i, archetypes):
            # Project data points onto the direction
            projections = jnp.dot(X_jax - centroid, directions[i])
            # Find the farthest point
            max_idx = jnp.argmax(projections)
            return archetypes.at[i].set(X_jax[max_idx])

        archetypes = jnp.zeros((self.n_archetypes, n_features))
        archetypes = jax.lax.fori_loop(0, self.n_archetypes, find_extreme_point, archetypes)

        return archetypes, jnp.zeros(self.n_archetypes, dtype=jnp.int32)

    def qhull_init(self, X_jax: jnp.ndarray, n_samples: int, n_features: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize archetypes using convex hull vertices via QHull algorithm."""
        X_np = np.array(X_jax)

        try:
            from scipy.spatial import ConvexHull

            self.logger.info(get_message("init", "model_init", model_name="ConvexHull", n_archetypes=self.n_archetypes))
            hull = ConvexHull(X_np)
        except Exception as e:
            self.logger.warning(
                get_message(
                    "warning",
                    "initialization_failed",
                    strategy="QHull",
                    error_msg=str(e),
                    fallback="k-means++",
                )
            )
            return self.kmeans_pp_init(X_jax, n_samples, n_features)

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
            self.rng_key, subkey = jax.random.split(self.rng_key)
            additional_indices = jax.random.choice(subkey, n_samples, shape=(remaining,), replace=False, p=None)
            selected_vertices.extend(additional_indices)

        else:
            selected_vertices = vertices

        archetypes = jnp.array(X_np[selected_vertices])

        return archetypes, jnp.array(selected_vertices)

    def kmeans_pp_init(self, X_jax: jnp.ndarray, n_samples: int, n_features: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """More efficient k-means++ style initialization using JAX.

        Args:
            X_jax: Data matrix of shape (n_samples, n_features)
            n_samples: Number of samples
            n_features: Number of features

        Returns:
            Archetypes and archetype indices
        """
        self.rng_key, subkey = jax.random.split(self.rng_key)
        first_idx = jax.random.randint(subkey, (), 0, n_samples)

        chosen_indices = jnp.zeros(self.n_archetypes, dtype=jnp.int32)
        chosen_indices = chosen_indices.at[0].set(first_idx)

        archetypes = jnp.zeros((self.n_archetypes, n_features))
        archetypes = archetypes.at[0].set(X_jax[first_idx])

        # Select remaining archetypes using k-means++ style initialization
        for i in range(1, self.n_archetypes):
            distances = jnp.sum((X_jax[:, jnp.newaxis, :] - archetypes[jnp.newaxis, :i, :]) ** 2, axis=2)
            min_distances = jnp.min(distances, axis=1)

            mask = jnp.ones(n_samples, dtype=bool)
            for j in range(i):
                mask = mask & (jnp.arange(n_samples) != chosen_indices[j])
            min_distances = min_distances * mask

            sum_distances = jnp.sum(min_distances) + self.eps
            probs = min_distances / sum_distances

            self.rng_key, subkey = jax.random.split(self.rng_key)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)

            chosen_indices = chosen_indices.at[i].set(next_idx)
            archetypes = archetypes.at[i].set(X_jax[next_idx])

        return archetypes, chosen_indices

    def fit(
        self,
        X: np.ndarray,
        normalize: bool = False,
        **kwargs,
    ) -> "ImprovedArchetypalAnalysis":
        """Discover optimal archetypes through advanced multi-strategy optimization.

        This core method identifies the extreme patterns that define the convex hull
        of the data and serve as the building blocks for representing all observations.
        The implementation features several critical enhancements:

        1. Intelligent initialization strategies that target promising positions
        2. Hybrid optimization combining gradient-based and direct algebraic updates
        3. Adaptive boundary projection to ensure archetypes represent true extremes
        4. Improved numerical stability through strategic regularization
        5. Early stopping logic to prevent overfitting and wasted computation

        These techniques collectively address the fundamental challenges of archetypal analysis:
            sensitivity to initialization, convergence to suboptimal solutions, and computational efficiency.
            and computational efficiency.

        Args:
            X: Data matrix to analyze (n_samples, n_features)
            normalize: Whether to normalize features before fitting
                - essential for data with features of different scales or magnitudes
            **kwargs: Additional optimization parameters:
                - early_stopping_patience: Iterations without improvement before stopping (higher values ensure convergence at computational cost)
                - additional parameters specific to the initialization method

        Returns:
            Self - fitted model instance with discovered archetypes
        """

        @partial(jax.jit, static_argnums=(3))
        def update_step(
            params: dict[str, jnp.ndarray], opt_state: optax.OptState, X: jnp.ndarray, iteration: int
        ) -> tuple[dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
            """Execute a single optimization step."""

            def loss_fn(params):
                return self.loss_function(params["archetypes"], params["weights"], X_f32)

            def apply_direct_update():
                """Apply direct algebraic update to archetypes.

                Blend with gradient-based update to maintain stability
                """
                archetypes_dir = self.update_archetypes(new_params["archetypes"], new_params["weights"], X_f32)
                blend_factor = 0.2
                return blend_factor * archetypes_dir + (1 - blend_factor) * new_params["archetypes"]

            params_f32 = jax.tree.map(lambda p: p.astype(jnp.float32), params)
            X_f32 = X.astype(jnp.float32)

            loss, grads = jax.value_and_grad(loss_fn)(params_f32)
            grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params_f32, updates)
            new_params["weights"] = self.project_weights(new_params["weights"])

            # Alternating optimization:
            #     periodically use direct archetype update instead of gradient
            #     This helps break out of local minima and improves convergence characteristics
            use_direct_update = jnp.mod(iteration, 15) == 0
            new_params["archetypes"] = jax.lax.cond(
                use_direct_update, lambda: apply_direct_update(), lambda: new_params["archetypes"]
            )

            pre_projection_archetypes = new_params["archetypes"]
            pre_projection_loss = self.loss_function(pre_projection_archetypes, new_params["weights"], X_f32)

            # Intermittent projection:
            #   only project archetypes every N iterations
            #   This allows optimization to make progress between projections
            do_projection = jnp.mod(iteration, 10) == 0

            def project_archetypes():
                if self.projection_method == "cbap" or self.projection_method == "default":
                    projected = self.project_archetypes(new_params["archetypes"], X_f32)
                elif self.projection_method == "convex_hull":
                    projected = self.project_archetypes_convex_hull(new_params["archetypes"], X_f32)
                else:
                    projected = self.project_archetypes_knn(new_params["archetypes"], X_f32)

                post_projection_loss = self.loss_function(projected, new_params["weights"], X_f32)

                # Use adaptive blending based on loss differential
                # If projection increases loss, use less of the projected result
                loss_ratio = post_projection_loss / (pre_projection_loss + self.eps)
                blend_factor = jnp.where(
                    loss_ratio > 1.01,  # Loss increased by more than 1%
                    0.01,  # Use only 1% of projected result if loss increases
                    0.5,  # Otherwise use 50% of projected result
                )
                blend_factor = jnp.where(
                    loss_ratio > 1.1,  # Loss increased by more than 10%
                    0.005,  # Barely use the projection (0.5%)
                    blend_factor,  # Otherwise use standard blend factor
                )

                return blend_factor * projected + (1 - blend_factor) * pre_projection_archetypes

            # Only apply projection on designated iterations
            new_params["archetypes"] = jax.lax.cond(
                do_projection, lambda: project_archetypes(), lambda: pre_projection_archetypes
            )
            new_params = jax.tree.map(lambda p: p.astype(jnp.float32), new_params)

            return new_params, opt_state, loss

        X_np = X.values if hasattr(X, "values") else X
        self.X_mean = np.mean(X_np, axis=0)
        self.X_std = np.std(X_np, axis=0)

        if self.X_std is not None:
            self.X_std = np.where(self.X_std < self.eps, np.ones_like(self.X_std), self.X_std)

        if self.normalize:
            X_scaled = (X_np - self.X_mean) / self.X_std
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            X_scaled = X_np.copy()

        X_jax = jnp.array(X_scaled, dtype=jnp.float32)
        n_samples, n_features = X_jax.shape
        self.logger.info(
            get_message(
                "data",
                "data_shape",
                shape=X_jax.shape,
                min=float(jnp.min(X_jax)),
                max=float(jnp.max(X_jax)),
            )
        )

        self.rng_key, subkey = jax.random.split(self.rng_key)
        weights_init = jax.random.uniform(
            subkey,
            (n_samples, self.n_archetypes),
            minval=0.1,
            maxval=0.9,
            dtype=jnp.float32,
        )
        weights_init = self.project_weights(weights_init)

        archetype_init_fn = {
            "directional": self.directional_init,
            "direction": self.directional_init,
            "qhull": self.qhull_init,
            "convex_hull": self.qhull_init,
            "kmeans": self.kmeans_pp_init,
            "kmeans++": self.kmeans_pp_init,
        }.get(self.archetype_init_method, self.directional_init)
        archetypes, _ = archetype_init_fn(X_jax, n_samples, n_features)
        archetypes = archetypes.astype(jnp.float32)
        prev_archetypes = archetypes.copy()

        optimizer: optax.GradientTransformation = optax.adam(learning_rate=self.learning_rate)

        params = {"archetypes": archetypes, "weights": weights_init}
        opt_state = optimizer.init(params)
        no_improvement_count = 0
        max_no_improvement = self.early_stopping_patience

        prev_loss = float("inf")
        best_loss = float("inf")
        best_params = {k: v.copy() for k, v in params.items()}
        self.loss_history = []

        initial_loss = float(self.loss_function(archetypes, weights_init, X_jax))
        self.logger.info(f"Initial loss: {initial_loss:.6f}")

        for it in range(self.max_iter):
            try:
                params, opt_state, loss = update_step(params, opt_state, X_jax, it)
                loss_value = float(loss)

                current_archetypes = params["archetypes"]
                archetype_changes = np.array(current_archetypes) - np.array(prev_archetypes)

                change_norms = np.linalg.norm(archetype_changes, axis=1)
                avg_change = np.mean(change_norms)
                max_change = np.max(change_norms)

                if it % 50 == 0 or max_change > 1.0:
                    self.logger.debug(
                        get_message(
                            "progress",
                            "iteration_progress",
                            current=it,
                            total=self.max_iter,
                            loss=loss_value,
                            avg_change=avg_change,
                            max_change=max_change,
                        )
                    )
                    if max_change > 1.0:
                        large_changes = np.where(change_norms > 1.0)[0]
                        if len(large_changes) > 0:
                            self.logger.debug(
                                get_message(
                                    "progress",
                                    "large_changes",
                                    archetypes=large_changes,
                                    changes=change_norms[large_changes],
                                )
                            )

                if jnp.isnan(loss_value):
                    self.logger.warning(get_message("warning", "nan_detected", iteration=it))
                    break

                self.loss_history.append(loss_value)
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_params = {k: v.copy() for k, v in params.items()}
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= max_no_improvement:
                    self.logger.info(
                        get_message("progress", "early_stopping", iteration=it, patience=max_no_improvement)
                    )
                    params = best_params
                    break

                # Periodically check if loss is increasing and restore best parameters if necessary
                if it > 0 and it % 20 == 0 and loss_value > prev_loss * 1.05:
                    self.logger.debug(get_message("warning", "loss_increase", previous=prev_loss, current=loss_value))
                    params = {k: v.copy() for k, v in best_params.items()}
                    opt_state = optimizer.init(params)

                if it > 0 and abs(prev_loss - loss_value) < self.tol:
                    self.logger.info(get_message("progress", "converged", iteration=it, tolerance=self.tol))
                    break

                if it % 50 == 0:
                    boundary_weights_pct = float(
                        jnp.mean(jnp.sum(params["weights"] < 1e-5, axis=1) / self.n_archetypes)
                    )
                    self.logger.info(
                        get_message(
                            "progress",
                            "iteration_progress",
                            current=it,
                            total=self.max_iter,
                            loss=loss_value,
                            boundary_weights=f"{boundary_weights_pct:.2%}",
                        )
                    )

                prev_loss = loss_value

            except Exception as e:
                self.logger.error(get_message("error", "computation_error", error_msg=str(e)))
                params = best_params
                break

        if "loss_value" in locals() and best_loss < loss_value:
            self.logger.info(get_message("result", "final_loss", loss=best_loss, iterations=len(self.loss_history)))
            params = best_params
        else:
            self.logger.info(get_message("result", "final_loss", loss=best_loss, iterations=len(self.loss_history)))

        total_change = np.linalg.norm(np.array(params["archetypes"]) - np.array(archetypes), axis=1)
        self.logger.info("Total change in archetypes:")
        for i, change in enumerate(total_change):
            self.logger.info(f"  Archetype {i + 1}: {change:.6f}")

        archetypes_scaled = np.array(best_params["archetypes"])
        self.archetypes = archetypes_scaled * self.X_std + self.X_mean if self.normalize else archetypes_scaled
        self.weights = np.array(best_params["weights"])

        if len(self.loss_history) > 0:
            self.logger.info(
                get_message(
                    "result",
                    "final_loss",
                    loss=self.loss_history[-1],
                    iterations=len(self.loss_history),
                )
            )
        else:
            self.logger.warning(get_message("warning", "high_loss", loss=float("nan")))

        return self

    @partial(jax.jit, static_argnums=(0,))
    def project_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Strategically position archetypes on the convex hull boundary for optimal representation.

        This method is critical for meaningful archetypal analysis as it ensures archetypes remain
        at the extremes of the data distribution where they best represent distinctive patterns.

        Our implementation differs from standard projection methods by:
        1. Projecting along meaningful directions from the data centroid
        2. Identifying precise extreme points rather than using approximate methods
        3. Blending original positions with boundary points for stability
        4. Applying adaptive adjustments based on current position

        Args:
            archetypes: Current archetype positions (n_archetypes, n_features)
            X: Data matrix defining the convex hull (n_samples, n_features)

        Returns:
            Projected archetypes strategically positioned at or near the convex hull boundary
        """
        centroid = jnp.mean(X, axis=0)

        def _project_to_boundary(archetype):
            self.rng_key, subkey = jax.random.split(self.rng_key)
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)
            normalized_direction = jnp.where(
                direction_norm > self.eps,
                direction / direction_norm,
                jax.random.normal(subkey, direction.shape) / jnp.sqrt(direction.shape[0]),
            )

            # Find most extreme point along this direction
            projections = jnp.dot(X - centroid, normalized_direction)
            max_idx = jnp.argmax(projections)
            extreme_point = X[max_idx]

            # Compare projections to detect if archetype is outside boundary
            extreme_projection = jnp.dot(extreme_point - centroid, normalized_direction)
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)
            is_outside = archetype_projection > extreme_projection

            # Blend archetype with extreme point based on position
            blended = jnp.where(
                is_outside,
                self.projection_alpha * extreme_point + (1 - self.projection_alpha) * archetype,
                (1 - self.projection_alpha) * extreme_point + self.projection_alpha * archetype,
            )

            return blended

        projected_archetypes = jax.vmap(_project_to_boundary)(archetypes)

        return jnp.asarray(projected_archetypes)

    @partial(jax.jit, static_argnums=(0,))
    def project_archetypes_convex_hull(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Alternative archetype projection that uses convex combinations of extreme points.

        This method identifies potential extreme points and creates archetypes
        as sparse convex combinations of these points, ensuring they lie on the boundary.

        Technical details:
        - Multi-directional Exploration:
            Generates multiple random directions around the main archetype direction,
            allowing for more diverse extreme point discovery.
        - Sparse Convex Combinations:
            Creates archetypes as weighted combinations of extreme points found in different directions,
            with emphasis on the main direction.
        - Boundary Positioning:
            By using convex combinations of extreme points,
            archetypes are positioned on or near the convex hull boundary rather than in its interior.

        This approach offers potentially better exploration of the convex hull boundary
        at the cost of slightly higher computational complexity.

        Args:
            archetypes: Current archetype matrix of shape (n_archetypes, n_features)
            X: Original data matrix of shape (n_samples, n_features)

        Returns:
            Projected archetype matrix positioned on the convex hull boundary
        """
        centroid = jnp.mean(X, axis=0)

        def _find_extreme_points(archetype):
            self.rng_key, subkey = jax.random.split(self.rng_key)

            n_directions = 5

            # Direction from centroid to archetype
            main_direction = archetype - centroid
            main_direction_norm = jnp.linalg.norm(main_direction)
            normalized_main_direction = jnp.where(
                main_direction_norm > self.eps,
                main_direction / main_direction_norm,
                jax.random.normal(subkey, shape=main_direction.shape) / jnp.sqrt(main_direction.shape[0]),
            )

            # Generate random perturbations of the main direction
            perturbations = jax.random.normal(subkey, shape=(n_directions, normalized_main_direction.shape[0]))
            perturbation_norms = jnp.linalg.norm(perturbations, axis=1, keepdims=True)
            normalized_perturbations = perturbations / (perturbation_norms + self.eps)

            # Create directions as combinations of main direction and perturbations
            directions = jnp.vstack([
                normalized_main_direction,
                normalized_perturbations * 0.3 + normalized_main_direction * 0.7,
            ])
            direction_norms = jnp.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / (direction_norms + self.eps)

            # Find extreme points in each direction
            def _find_extreme(i, indices):
                projections = jnp.dot(X - centroid, directions[i])
                max_idx = jnp.argmax(projections)
                return indices.at[i].set(max_idx)

            extreme_indices = jnp.zeros(directions.shape[0], dtype=jnp.int32)
            extreme_indices = jax.lax.fori_loop(0, directions.shape[0], _find_extreme, extreme_indices)
            extreme_points = X[extreme_indices]

            # Create a sparse convex combination of these extreme points
            # with higher weight on the main direction's extreme point
            weights = jnp.ones(extreme_points.shape[0]) / extreme_points.shape[0]
            weights = weights.at[0].set(weights[0] * 2)
            weights = weights / jnp.sum(weights)

            return jnp.sum(weights[:, jnp.newaxis] * extreme_points, axis=0)

        projected_archetypes = jax.vmap(_find_extreme_points)(archetypes)

        return jnp.asarray(projected_archetypes)

    @partial(jax.jit, static_argnums=(0,))
    def project_archetypes_knn(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Original k-NN based archetype projection (kept for comparison).

        This method tends to pull archetypes inside the convex hull due to its averaging nature,
        which is suboptimal for archetypal analysis where archetypes should ideally lie on the convex hull boundary.

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
            weights = 1.0 / (top_k_dists + self.eps)
            weights = weights / jnp.sum(weights)
            projected = jnp.sum(weights[:, jnp.newaxis] * X[top_k_indices], axis=0)
            return projected

        dists = jnp.sum((X[:, jnp.newaxis, :] - archetypes[jnp.newaxis, :, :]) ** 2, axis=2)
        k = 10

        projected_archetypes = jax.vmap(_process_single_archetype)(jnp.arange(archetypes.shape[0]))

        return jnp.asarray(projected_archetypes)

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, archetypes: jnp.ndarray, weights: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Composite objective function balancing reconstruction with interpretability.

        This carefully designed loss function guides the optimization process
        by balancing multiple competing objectives essential for archetypal analysis:
        1. Reconstruction fidelity: Ensuring archetypes accurately represent the data
        2. Weight interpretability: Encouraging sparse, distinctive weight patterns
        3. Boundary alignment: Promoting archetypes at meaningful extremal positions

        The weighted combination of these terms creates a landscape that guides optimization
        toward solutions with both mathematical validity (convex hull representation)
        and practical utility (interpretable patterns).

        The relative weighting of these components is critical to achieving the right balance
        between reconstruction accuracy and archetypal properties.

        Args:
            archetypes:
                Candidate archetype matrix (n_archetypes, n_features)
            weights:
                Weight matrix (n_samples, n_archetypes) describing how to represent each sample
                as a combination of archetypes
            X: Original data matrix (n_samples, n_features) to reconstruct

        Returns:
            Scalar loss value combining reconstruction error with regularization
            terms - lower values indicate better solutions
        """
        archetypes_f32 = archetypes.astype(jnp.float32)
        weights_f32 = weights.astype(jnp.float32)
        X_f32 = X.astype(jnp.float32)

        # Reconstruction error
        X_reconstructed = jnp.matmul(weights_f32, archetypes_f32)
        reconstruction_loss = jnp.mean(jnp.sum((X_f32 - X_reconstructed) ** 2, axis=1))

        # entropy (higher values for uniform weights, lower for sparse weights)
        entropy = -jnp.sum(weights_f32 * jnp.log(weights_f32 + self.eps), axis=1)
        entropy_reg = jnp.mean(entropy)

        # Add incentive for archetypes to stay near convex hull boundary
        # But use a much lower weight than the parent class
        boundary_incentive = self._calculate_boundary_proximity(archetypes_f32, X_f32)

        # We use a significantly reduced boundary incentive for tracking stability
        # This matches the parent class boundary incentive level
        total_loss = reconstruction_loss + self.lambda_reg * entropy_reg - 0.001 * boundary_incentive

        return jnp.asarray(total_loss.astype(jnp.float32))

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_boundary_proximity(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Calculate a metric that rewards archetypes for being near convex hull boundary.

        A high value indicates archetypes are closer to the convex hull boundary,
        which is desirable for archetypal analysis.
        This serves as a regularization term that encourages archetypes to move toward extremal positions.

        Args:
            archetypes: Archetype matrix of shape (n_archetypes, n_features)
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Boundary proximity score as a scalar
        """
        centroid = jnp.mean(X, axis=0)

        def _boundary_score_for_archetype(archetype):
            # Direction from centroid to archetype
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)
            normalized_direction = jnp.where(
                direction_norm > self.eps, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Project all points onto this direction
            projections = jnp.dot(X - centroid, normalized_direction)
            max_projection = jnp.max(projections)
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)
            normalized_proximity = archetype_projection / (max_projection + self.eps)

            # Penalize archetypes outside the convex hull
            # This creates a peak at exactly the boundary (normalized_proximity = 1.0)
            # and penalizes positions both inside (< 1.0) and outside (> 1.0)
            boundary_score = 1.0 - jnp.abs(normalized_proximity - 1.0)
            is_outside = normalized_proximity > 1.0
            outside_penalty = jnp.where(is_outside, jnp.exp(normalized_proximity - 1.0) - 1.0, 0.0)

            return jnp.power(boundary_score, 2) - outside_penalty

        scores = jax.vmap(_boundary_score_for_archetype)(archetypes)

        return jnp.mean(scores)

    @partial(jax.jit, static_argnums=(0,))
    def project_weights(self, weights: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled weight projection function.

        Args:
            weights: Weight matrix of shape (n_samples, n_archetypes)

        Returns:
            Projected weight matrix of shape (n_samples, n_archetypes)
        """
        weights = jnp.maximum(self.eps, weights)
        sum_weights = jnp.sum(weights, axis=1, keepdims=True)
        sum_weights = jnp.maximum(self.eps, sum_weights)
        return weights / sum_weights

    @partial(jax.jit, static_argnums=(0,))
    def update_archetypes(self, archetypes: jnp.ndarray, weights: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Alternative archetype update strategy based on weighted reconstruction.

        This approach directly optimizes archetypes by computing the pseudo-inverse of weights,
        which often provides a more targeted and mathematically sound update than
        gradient descent for this specific subproblem.

        Args:
            archetypes: Archetype matrix of shape (n_archetypes, n_features)
            weights: Weight matrix of shape (n_samples, n_archetypes)
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Updated archetype matrix of shape (n_archetypes, n_features)
        """
        # Calculate weight matrix pseudoinverse with improved numerical stability
        W = weights
        WtW = jnp.dot(W.T, W) + self.eps * jnp.eye(W.shape[1])
        WtX = jnp.dot(W.T, X)

        # Solve for archetypes using the normal equations
        # This is equivalent to minimizing ||X - W*A||^2 with respect to A
        archetypes_updated = jnp.linalg.solve(WtW, WtX)

        # Process each archetype to ensure it's inside the convex hull
        centroid = jnp.mean(X, axis=0)

        def _constrain_to_convex_hull(archetype):
            # Direction from centroid to archetype
            direction = archetype - centroid
            direction_norm = jnp.linalg.norm(direction)
            normalized_direction = jnp.where(
                direction_norm > self.eps, direction / direction_norm, jnp.zeros_like(direction)
            )

            # Find max projection (extreme point in this direction)
            projections = jnp.dot(X - centroid, normalized_direction)
            max_projection = jnp.max(projections)

            # Calculate archetype projection along this direction
            archetype_projection = jnp.dot(archetype - centroid, normalized_direction)

            # Scale factor to bring the archetype inside the convex hull if it's outside
            scale_factor = jnp.where(
                archetype_projection > max_projection,
                0.99 * max_projection / (archetype_projection + self.eps),
                1.0,
            )

            # Apply the scaling to the direction vector
            constrained_archetype = centroid + scale_factor * (archetype - centroid)

            return constrained_archetype

        constrained_archetypes = jax.vmap(_constrain_to_convex_hull)(archetypes_updated)

        return jnp.asarray(constrained_archetypes)
