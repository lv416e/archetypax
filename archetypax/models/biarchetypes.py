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

    Biarchetypal Analysis uses two sets of archetypes to represent data,
    allowing for more flexible and expressive representations.
    """

    def __init__(
        self,
        n_archetypes_first: int,
        n_archetypes_second: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_seed: int = 42,
        learning_rate: float = 0.001,
        mixture_weight: float = 0.5,
        lambda_reg: float = 0.01,
    ):
        """
        Initialize the Biarchetypal Analysis model.

        Args:
            n_archetypes_first: Number of archetypes in the first set
            n_archetypes_second: Number of archetypes in the second set
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            random_seed: Random seed for initialization
            learning_rate: Learning rate for optimizer
            mixture_weight: Weight for mixing the two archetype sets (0-1)
            lambda_reg: Regularization parameter for entropy
        """
        # Initialize using parent class with the total number of archetypes
        super().__init__(
            n_archetypes=n_archetypes_first + n_archetypes_second,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            learning_rate=learning_rate,
        )

        # Store biarchetypal specific parameters
        self.n_archetypes_first = n_archetypes_first
        self.n_archetypes_second = n_archetypes_second
        self.mixture_weight = mixture_weight
        self.lambda_reg = lambda_reg

        # Will be set during fitting
        self.archetypes_first: np.ndarray | None = None
        self.archetypes_second: np.ndarray | None = None
        self.weights_first: np.ndarray | None = None
        self.weights_second: np.ndarray | None = None

    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, archetypes, weights, X):
        """Biarchetypal loss function with mixture weight."""
        # For compatibility with parent class
        if isinstance(archetypes, dict):
            # Called from our own fit method with params dict
            params = archetypes
            archetypes_first = params["archetypes_first"]
            archetypes_second = params["archetypes_second"]
            weights_first = params["weights_first"]
            weights_second = params["weights_second"]
            # Use the X parameter directly
        else:
            # This branch should not be reached in normal operation
            # but is here for API compatibility
            raise ValueError("BiarchetypalAnalysis requires different parameters than parent class")

        # Reconstruct using first set of archetypes
        X_reconstructed_first = jnp.matmul(weights_first, archetypes_first)

        # Reconstruct using second set of archetypes
        X_reconstructed_second = jnp.matmul(weights_second, archetypes_second)

        # Weighted mixture of the two reconstructions
        X_reconstructed = (
            self.mixture_weight * X_reconstructed_first + (1 - self.mixture_weight) * X_reconstructed_second
        )

        # Reconstruction loss
        reconstruction_loss = jnp.mean(jnp.sum((X - X_reconstructed) ** 2, axis=1))

        # Entropy regularization
        entropy_reg = 0.0
        if self.lambda_reg > 0:
            entropy_reg = float(
                (
                    jnp.mean(jnp.sum(-weights_first * jnp.log(jnp.maximum(weights_first, 1e-10)), axis=1))
                    + jnp.mean(jnp.sum(-weights_second * jnp.log(jnp.maximum(weights_second, 1e-10)), axis=1))
                )
                / 2.0
            )

        # Return total loss
        return (reconstruction_loss + self.lambda_reg * entropy_reg).astype(jnp.float32)

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

        # Convert to JAX array
        X_jax = jnp.array(X_scaled)
        n_samples, n_features = X_jax.shape

        # Debug information
        print(f"Data shape: {X_jax.shape}")
        print(f"Data range: min={jnp.min(X_jax):.4f}, max={jnp.max(X_jax):.4f}")
        print(f"First archetype set: {self.n_archetypes_first}")
        print(f"Second archetype set: {self.n_archetypes_second}")
        print(f"Mixture weight: {self.mixture_weight}")

        # Initialize weights for first set
        self.key, subkey = jax.random.split(self.key)
        weights_first_init = jax.random.uniform(subkey, (n_samples, self.n_archetypes_first), minval=0.1, maxval=0.9)
        weights_first_init = self.project_weights(weights_first_init)

        # Initialize weights for second set
        self.key, subkey = jax.random.split(self.key)
        weights_second_init = jax.random.uniform(subkey, (n_samples, self.n_archetypes_second), minval=0.1, maxval=0.9)
        weights_second_init = self.project_weights(weights_second_init)

        # Initialize archetypes for first set (k-means++ style)
        archetypes_first_init, _ = self._kmeans_pp_init(X_jax, n_samples, n_features, self.n_archetypes_first)

        # Initialize archetypes for second set (k-means++ style)
        archetypes_second_init, _ = self._kmeans_pp_init(X_jax, n_samples, n_features, self.n_archetypes_second)

        # Set up optimizer
        optimizer = optax.adam(learning_rate=self.learning_rate)

        # Define update function
        @partial(jax.jit, static_argnums=(3,))
        def update_step(params, opt_state, X, iteration):
            """Execute a single optimization step."""
            # Calculate loss and gradients
            loss, grads = jax.value_and_grad(lambda p: self.loss_function(p, None, X))(params)

            # Apply gradient clipping to prevent NaNs
            for k in grads:
                grads[k] = jnp.clip(grads[k], -1.0, 1.0)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            # Project weights to simplex constraints
            new_params["weights_first"] = self.project_weights(new_params["weights_first"])
            new_params["weights_second"] = self.project_weights(new_params["weights_second"])

            # Project archetypes using soft assignment
            new_params["archetypes_first"] = self._project_archetypes_set(new_params["archetypes_first"], X)
            new_params["archetypes_second"] = self._project_archetypes_set(new_params["archetypes_second"], X)

            return new_params, opt_state, loss

        # Initialize parameters
        params = {
            "archetypes_first": archetypes_first_init,
            "archetypes_second": archetypes_second_init,
            "weights_first": weights_first_init,
            "weights_second": weights_second_init,
        }
        opt_state = optimizer.init(params)

        # Optimization loop
        prev_loss = float("inf")
        self.loss_history = []

        # Calculate initial loss for debugging
        initial_loss = float(self.loss_function(params, None, X_jax))
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
        archetypes_first_scaled = np.array(params["archetypes_first"])
        archetypes_second_scaled = np.array(params["archetypes_second"])

        self.archetypes_first = archetypes_first_scaled * self.X_std + self.X_mean
        self.archetypes_second = archetypes_second_scaled * self.X_std + self.X_mean

        # Combine archetypes for compatibility with parent methods
        if self.archetypes_first is not None and self.archetypes_second is not None:
            self.archetypes = np.vstack([np.array(self.archetypes_first), np.array(self.archetypes_second)])
        else:
            self.archetypes = np.array([])

        # Store the weights
        self.weights_first = np.array(params["weights_first"])
        self.weights_second = np.array(params["weights_second"])

        # Store combined weights for compatibility (this is approximate)
        combined_weights = np.hstack([
            self.mixture_weight * self.weights_first,
            (1 - self.mixture_weight) * self.weights_second,
        ])
        # Normalize combined weights to sum to 1
        self.weights = combined_weights / np.sum(combined_weights, axis=1, keepdims=True)

        if len(self.loss_history) > 0:
            print(f"Final loss: {self.loss_history[-1]:.6f}")
        else:
            print("Warning: No valid loss was recorded")

        return self

    def _kmeans_pp_init(self, X, n_samples, n_features, n_archetypes):
        """Helper function for k-means++ initialization for a set of archetypes."""
        # Randomly select the first center
        self.key, subkey = jax.random.split(self.key)
        first_idx = jax.random.randint(subkey, (), 0, n_samples)
        chosen_indices = [int(first_idx)]

        # Select remaining archetypes based on distance
        for _ in range(1, n_archetypes):
            self.key, subkey = jax.random.split(self.key)

            # Calculate minimum distance to already selected archetypes
            min_dists_list = []
            for i in range(n_samples):
                if i in chosen_indices:
                    min_dists_list.append(0.0)  # Don't select already chosen points
                else:
                    # Find minimum distance to selected archetypes
                    dist = jnp.array(float("inf"))
                    for idx in chosen_indices:
                        d = jnp.sum((X[i] - X[idx]) ** 2)
                        dist = jnp.minimum(dist, d)
                    min_dists_list.append(float(dist))

            # Select next archetype with probability proportional to squared distance
            min_dists = jnp.array(min_dists_list)
            probs = min_dists / (jnp.sum(min_dists) + 1e-10)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)
            chosen_indices.append(int(next_idx))

        # Initialize archetypes from selected indices
        archetypes_init = X[jnp.array(chosen_indices)]

        return archetypes_init, chosen_indices

    def _project_archetypes_set(self, archetypes, X):
        """Project archetypes using soft assignment based on k-nearest neighbors."""

        def _process_archetype(i):
            archetype_dists = dists[:, i]
            top_k_indices = jnp.argsort(archetype_dists)[:k]
            top_k_dists = archetype_dists[top_k_indices]
            weights = 1.0 / (top_k_dists + 1e-10)
            weights = weights / jnp.sum(weights)
            projected = jnp.sum(weights[:, jnp.newaxis] * X[top_k_indices], axis=0)
            return projected

        dists = jnp.sum((X[:, jnp.newaxis, :] - archetypes[jnp.newaxis, :, :]) ** 2, axis=2)
        k = jnp.minimum(10, X.shape[0])
        projected_archetypes = jnp.stack([_process_archetype(i) for i in range(archetypes.shape[0])])

        return projected_archetypes

    def transform(self, X: np.ndarray) -> Any:
        """
        Transform new data to archetype weights for both sets.

        Args:
            X: New data matrix of shape (n_samples, n_features)

        Returns:
            Tuple of (weights_first, weights_second) representing each sample as combinations
            of archetypes from the two sets
        """
        if self.archetypes_first is None or self.archetypes_second is None:
            raise ValueError("Model must be fitted before transform")

        # Scale input data
        X_scaled = (X - self.X_mean) / self.X_std if self.X_mean is not None and self.X_std is not None else X
        X_jax = jnp.array(X_scaled)

        # Scale archetypes
        archetypes_first_scaled = (
            (self.archetypes_first - self.X_mean) / self.X_std
            if self.X_mean is not None and self.X_std is not None
            else self.archetypes_first
        )
        archetypes_first_jax = jnp.array(archetypes_first_scaled)

        archetypes_second_scaled = (
            (self.archetypes_second - self.X_mean) / self.X_std
            if self.X_mean is not None and self.X_std is not None
            else self.archetypes_second
        )
        archetypes_second_jax = jnp.array(archetypes_second_scaled)

        # Define optimization for first set
        @jax.jit
        def optimize_weights_first(x_sample):
            w = jnp.ones(self.n_archetypes_first) / self.n_archetypes_first

            def step(w, _):
                pred = jnp.dot(w, archetypes_first_jax)
                error = x_sample - pred
                grad = -2 * jnp.dot(error, archetypes_first_jax.T)
                w_new = w - 0.01 * grad
                w_new = jnp.maximum(1e-10, w_new)
                sum_w = jnp.sum(w_new)
                w_new = jnp.where(sum_w > 1e-10, w_new / sum_w, jnp.ones_like(w_new) / self.n_archetypes_first)
                return w_new, None

            final_w, _ = jax.lax.scan(step, w, jnp.arange(100))
            return final_w

        # Define optimization for second set
        @jax.jit
        def optimize_weights_second(x_sample):
            w = jnp.ones(self.n_archetypes_second) / self.n_archetypes_second

            def step(w, _):
                pred = jnp.dot(w, archetypes_second_jax)
                error = x_sample - pred
                grad = -2 * jnp.dot(error, archetypes_second_jax.T)
                w_new = w - 0.01 * grad
                w_new = jnp.maximum(1e-10, w_new)
                sum_w = jnp.sum(w_new)
                w_new = jnp.where(sum_w > 1e-10, w_new / sum_w, jnp.ones_like(w_new) / self.n_archetypes_second)
                return w_new, None

            final_w, _ = jax.lax.scan(step, w, jnp.arange(100))
            return final_w

        # Vectorize the optimization across all samples
        batch_optimize_first = jax.vmap(optimize_weights_first)
        batch_optimize_second = jax.vmap(optimize_weights_second)

        weights_first_jax = batch_optimize_first(X_jax)
        weights_second_jax = batch_optimize_second(X_jax)

        return np.array(weights_first_jax), np.array(weights_second_jax)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, normalize: bool = False) -> Any:
        """
        Fit the model and transform the data.

        Args:
            X: Data matrix
            y: Ignored (for compatibility with sklearn)
            normalize: Whether to normalize the data

        Returns:
            Tuple of (weights_first, weights_second)
        """
        self.fit(X, normalize=normalize)
        weights_first, weights_second = self.transform(X)
        return np.asarray(weights_first), np.asarray(weights_second)

    def reconstruct(self, X: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct data from archetype weights.

        Args:
            X: Optional data matrix to reconstruct. If None, uses the training data.

        Returns:
            Reconstructed data matrix
        """
        if X is not None:
            # Transform new data and reconstruct
            weights_first, weights_second = self.transform(X)
        else:
            # Use stored weights from training
            if self.weights_first is None or self.weights_second is None:
                raise ValueError("Model must be fitted before reconstruction")
            weights_first, weights_second = self.weights_first, self.weights_second

        if self.archetypes_first is None or self.archetypes_second is None:
            raise ValueError("Model must be fitted before reconstruction")

        # Reconstruct using both archetype sets
        recon_first = np.matmul(weights_first, self.archetypes_first)
        recon_second = np.matmul(weights_second, self.archetypes_second)

        # Combine reconstructions using mixture weight
        reconstructed = self.mixture_weight * recon_first + (1 - self.mixture_weight) * recon_second

        return np.array(reconstructed)

    def get_all_archetypes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get both sets of archetypes.

        Returns:
            Tuple of (archetypes_first, archetypes_second)
        """
        if self.archetypes_first is None or self.archetypes_second is None:
            raise ValueError("Model must be fitted before accessing archetypes")

        return np.array(self.archetypes_first), np.array(self.archetypes_second)

    def get_all_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get weights for both sets of archetypes.

        Returns:
            Tuple of (weights_first, weights_second)
        """
        if self.weights_first is None or self.weights_second is None:
            raise ValueError("Model must be fitted before getting weights")

        return np.array(self.weights_first), np.array(self.weights_second)
