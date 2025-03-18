"""GPU-accelerated Archetypal Analysis implementation using JAX."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.base import BaseEstimator, TransformerMixin


class ArchetypalAnalysis(BaseEstimator, TransformerMixin):
    """GPU-accelerated Archetypal Analysis implementation using JAX."""

    def __init__(
        self,
        n_archetypes: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_seed: int = 42,
        learning_rate: float = 0.001,
    ):
        """
        Initialize the Archetypal Analysis model.

        Args:
            n_archetypes: Number of archetypes to find
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            random_seed: Random seed for initialization
            learning_rate: Learning rate for optimizer (reduced for better stability)
        """
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.key = jax.random.key(random_seed)
        self.learning_rate = learning_rate

        # Will be set during fitting
        self.archetypes: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.loss_history: list[float] = []

        # Store scaling information
        self.X_mean: np.ndarray | None = None
        self.X_std: np.ndarray | None = None

    def loss_function(self, archetypes, weights, X):
        """Add regularization term to the loss function."""
        X_reconstructed = jnp.matmul(weights, archetypes)
        reconstruction_loss = jnp.mean(jnp.sum((X - X_reconstructed) ** 2, axis=1))
        entropy = -jnp.sum(weights * jnp.log(weights + 1e-10), axis=1)
        entropy_reg = -jnp.mean(entropy)
        lambda_reg = 0.01
        return reconstruction_loss + lambda_reg * entropy_reg

    def project_weights(self, weights):
        """
        Project weights to satisfy simplex constraints with numerical stability.

        Args:
            weights: Weight matrix

        Returns:
            Projected weight matrix
        """
        eps = 1e-10
        weights = jnp.maximum(eps, weights)
        sum_weights = jnp.sum(weights, axis=1, keepdims=True)
        sum_weights = jnp.maximum(eps, sum_weights)
        return weights / sum_weights

    def project_archetypes(self, archetypes, X):
        """
        Project archetypes using soft assignment based on k-nearest neighbors.

        Args:
            archetypes: Archetype matrix
            X: Original data matrix

        Returns:
            Projected archetype matrix
        """

        def _process_archetype(i):
            archetype_dists = dists[:, i]
            top_k_indices = jnp.argsort(archetype_dists)[:k]
            top_k_dists = archetype_dists[top_k_indices]
            weights = 1.0 / (top_k_dists + 1e-10)
            weights = weights / jnp.sum(weights)
            projected = jnp.sum(weights[:, jnp.newaxis] * X[top_k_indices], axis=0)
            return projected

        dists = jnp.sum((X[:, jnp.newaxis, :] - archetypes[jnp.newaxis, :, :]) ** 2, axis=2)
        k = min(10, X.shape[0])
        projected_archetypes = jnp.stack([_process_archetype(i) for i in range(archetypes.shape[0])])
        return projected_archetypes

    def fit(self, X: np.ndarray, normalize: bool = False, **kwargs) -> "ArchetypalAnalysis":
        """
        Fit the model to the data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            normalize: Whether to normalize the data before fitting.
            **kwargs: Additional keyword arguments for the fit method.

        Returns:
            Self
        """
        # Preprocess data: scale for improved stability
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        # Prevent division by zero with explicit type casting
        if self.X_std is not None:
            self.X_std = np.where(self.X_std < 1e-10, np.ones_like(self.X_std), self.X_std)
        X_scaled = (X - self.X_mean) / self.X_std if normalize else X.copy()

        # Convert from NumPy to JAX array
        X_jax = jnp.array(X_scaled)
        n_samples, _ = X_jax.shape

        # Debug information
        print(f"Data shape: {X_jax.shape}")
        print(f"Data range: min={jnp.min(X_jax):.4f}, max={jnp.max(X_jax):.4f}")

        # Initialize weights (more stable initialization)
        self.key, subkey = jax.random.split(self.key)
        weights_init = jax.random.uniform(subkey, (n_samples, self.n_archetypes), minval=0.1, maxval=0.9)
        weights_init = self.project_weights(weights_init)

        # Initialize archetypes (k-means++ style initialization)
        self.key, subkey = jax.random.split(self.key)
        # Select first archetype randomly
        first_idx = jax.random.randint(subkey, (), 0, n_samples)
        chosen_indices = [int(first_idx)]

        # Select remaining archetypes based on distance
        for _ in range(1, self.n_archetypes):
            self.key, subkey = jax.random.split(self.key)

            # Calculate minimum distance to already selected archetypes
            min_dists_list = []
            for i in range(n_samples):
                if i in chosen_indices:
                    min_dists_list.append(0.0)  # Don't select already chosen points
                else:
                    # Find minimum distance to selected archetypes
                    dist = float("inf")
                    for idx in chosen_indices:
                        d = np.sum((X_scaled[i] - X_scaled[idx]) ** 2)
                        dist = min(dist, d)
                    min_dists_list.append(dist)

            # Select next archetype with probability proportional to squared distance
            min_dists = np.array(min_dists_list)
            probs = min_dists / (np.sum(min_dists) + 1e-10)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)
            chosen_indices.append(int(next_idx))

        # Initialize archetypes from selected indices
        archetypes_init = X_jax[jnp.array(chosen_indices)]

        # Set up optimizer (Adam with reduced learning rate)
        optimizer = optax.adam(learning_rate=self.learning_rate)

        # Define JIT-compiled update function
        @partial(jax.jit, static_argnums=(3,))
        def update_step(params, opt_state, X, iteration):
            """Execute a single optimization step."""

            # Loss function
            def loss_fn(params):
                return self.loss_function(params["archetypes"], params["weights"], X)

            # Calculate gradient and update
            loss, grads = jax.value_and_grad(loss_fn)(params)

            # Apply gradient clipping to prevent NaNs
            for k in grads:
                grads[k] = jnp.clip(grads[k], -1.0, 1.0)

            # Get new parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            # Project to constraints
            new_params["weights"] = self.project_weights(new_params["weights"])
            new_params["archetypes"] = self.project_archetypes(new_params["archetypes"], X)

            return new_params, opt_state, loss

        # Initialize parameters
        params = {"archetypes": archetypes_init, "weights": weights_init}
        opt_state = optimizer.init(params)

        # Optimization loop
        prev_loss = float("inf")

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
                    # Use last valid parameters
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

    def transform(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """
        Transform new data to archetype weights.

        Args:
            X: New data matrix of shape (n_samples, n_features)
            y: Ignored. Present for API consistency by convention.
            **kwargs: Additional keyword arguments for the transform method.

        Returns:
            Weight matrix representing each sample as a combination of archetypes
        """
        if self.archetypes is None:
            raise ValueError("Model must be fitted before transform")

        # Scale input data
        X_scaled = (X - self.X_mean) / self.X_std if self.X_mean is not None and self.X_std is not None else X

        # Simple approach to find weights (simplified non-negative least squares)
        n_samples = X_scaled.shape[0]
        weights = np.zeros((n_samples, self.n_archetypes))

        # Scale archetypes too
        archetypes_scaled = (
            (self.archetypes - self.X_mean) / self.X_std
            if self.X_mean is not None and self.X_std is not None
            else self.archetypes
        )

        for i in range(n_samples):
            # Simple optimization for each sample
            w = np.ones(self.n_archetypes) / self.n_archetypes  # Uniform initial values

            # Simple gradient descent
            learning_rate = 0.01
            for _ in range(100):
                # Calculate gradient
                pred = np.dot(w, archetypes_scaled)
                error = X_scaled[i] - pred
                grad = -2 * np.dot(error, archetypes_scaled.T)

                # Update
                w = w - learning_rate * grad

                # Project to constraints
                w = np.maximum(1e-10, w)  # Non-negativity with small epsilon
                sum_w = np.sum(w)
                # Avoid division by zero
                w = w / sum_w if sum_w > 1e-10 else np.ones(self.n_archetypes) / self.n_archetypes

            weights[i] = w

        return weights

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, normalize: bool = False) -> np.ndarray:
        """
        Fit the model and transform the data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            y: Ignored. Present for API consistency by convention.
            normalize: Whether to normalize the data before fitting.

        Returns:
            Weight matrix representing each sample as a combination of archetypes
        """
        model = self.fit(X, normalize=normalize)
        return np.asarray(model.transform(X))

    def reconstruct(self, X: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct data using the learned archetypes.

        Args:
            X: Data to reconstruct, or None to use the training data

        Returns:
            Reconstructed data
        """
        if self.archetypes is None:
            raise ValueError("Model must be fitted before reconstruct")

        if self.weights is None and X is None:
            raise ValueError("Either weights or input data must be provided")

        weights = self.transform(X) if X is not None else self.weights
        if weights is None:
            raise ValueError("Weights must not be None")

        return np.array(np.matmul(weights, self.archetypes))

    def get_loss_history(self) -> list[float]:
        """
        Get the loss history from training.

        Returns:
            List of loss values recorded during fitting
        """
        return self.loss_history
