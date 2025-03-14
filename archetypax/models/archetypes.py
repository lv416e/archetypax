"""Improved Archetypal Analysis model using JAX."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .base import ArchetypalAnalysis


class ImprovedArchetypalAnalysis(ArchetypalAnalysis):
    """Improved Archetypal Analysis model using JAX."""

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

    def fit(self, X: np.ndarray, normalize: bool = False):
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
            new_params["archetypes"] = self.project_archetypes(new_params["archetypes"], X_f32)
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

        # Use improved k-means++ initialization
        archetypes_init, _ = self.kmeans_pp_init(X_jax, n_samples, n_features)
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

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, normalize: bool = False) -> np.ndarray:
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
        """JIT-compiled archetype projection with static k value."""

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
        lambda_reg = 0.01

        return (reconstruction_loss + lambda_reg * entropy_reg).astype(jnp.float32)
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
