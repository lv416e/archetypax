"""Archetypal Analysis using JAX.

This module provides the foundational implementation of Archetypal Analysis (AA) optimized for GPU acceleration via JAX.
It serves as the base class for more advanced implementations in the archetypax package.

Archetypal Analysis identifies extreme patterns (archetypes) in data that can represent the entire dataset
through convex combinations, offering both dimensionality reduction and interpretable insights into data structure.

Core Features:
- JAX-based implementation for GPU/TPU acceleration
- Scikit-learn compatible API (BaseEstimator, TransformerMixin)
- Standard k-means++ style initialization
- Gradient-based optimization with Adam
- Basic weight and archetype projection methods

This base implementation provides a solid foundation with standard AA features,
while more advanced techniques are available in derived classes such as ImprovedArchetypalAnalysis.

Example usage:
    ```python
    from archetypax.models import ArchetypalAnalysis

    # Initialize model
    model = ArchetypalAnalysis(
        n_archetypes=5,
        normalize=True
    )

    # Fit model and transform data
    weights = model.fit_transform(X)
    ```
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.base import BaseEstimator, TransformerMixin

from archetypax.logger import get_logger, get_message


class ArchetypalAnalysis(BaseEstimator, TransformerMixin):
    """GPU-accelerated Archetypal Analysis implementation using JAX.

    This class provides the core functionality for identifying archetypes
    - extreme points that can represent data through convex combinations,
        offering interpretable and meaningful insights into data structure.

    Leverages JAX for efficient GPU computation and automatic differentiation.
    """

    def __init__(
        self,
        n_archetypes: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_seed: int = 42,
        learning_rate: float = 0.001,
        normalize: bool = False,
        **kwargs,
    ):
        """
        Initialize the Archetypal Analysis model.

        Args:
            n_archetypes:
                Number of archetypes to find
                - determines the dimensionality of the representation space
            max_iter:
                Maximum number of iterations for optimization convergence
            tol:
                Convergence tolerance for early stopping
            random_seed:
                Random seed for reproducible results
            learning_rate:
                Learning rate for optimizer
                - lower values provide better stability at the cost of slower convergence
            normalize:
                Whether to normalize the data before fitting
                - essential for features with different scales
            **kwargs: Additional keyword arguments including:
                early_stopping_patience:
                    Number of iterations without improvement before stopping optimization
                logger_level/verbose_level:
                    Control for logging granularity
        """
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
                learning_rate=learning_rate,
                max_iter=max_iter,
                tol=tol,
                normalize=normalize,
                random_seed=random_seed,
            )
        )

        self.eps = jnp.finfo(jnp.float32).eps
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.rng_key = jax.random.key(random_seed)
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.archetypes: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.loss_history: list[float] = []
        self.X_mean: np.ndarray | None = None
        self.X_std: np.ndarray | None = None

        self.early_stopping_patience = kwargs.get("early_stopping_patience", 100)

    def loss_function(self, archetypes: jnp.ndarray, weights: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Calculate reconstruction loss with entropy regularization.

        Computes the fundamental objective: minimize reconstruction error while
        encouraging more discriminative weights through entropy regularization.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)
            weights: Weight matrix (n_samples, n_archetypes)
            X: Data matrix (n_samples, n_features)

        Returns:
            Combined loss value as a scalar
        """
        X_reconstructed = jnp.matmul(weights, archetypes)
        reconstruction_loss = jnp.mean(jnp.sum((X - X_reconstructed) ** 2, axis=1))
        entropy = -jnp.sum(weights * jnp.log(weights + self.eps), axis=1)
        entropy_reg = -jnp.mean(entropy)
        lambda_reg = 0.01
        return reconstruction_loss + lambda_reg * entropy_reg

    def project_weights(self, weights: jnp.ndarray) -> jnp.ndarray:
        """
        Project weights to satisfy simplex constraints with numerical stability.

        Ensures that weights form valid convex combinations (non-negative and sum to 1)
        while avoiding numerical underflow/overflow issues.

        Args:
            weights: Weight matrix (n_samples, n_archetypes)

        Returns:
            Projected weight matrix (n_samples, n_archetypes)
        """
        weights = jnp.maximum(self.eps, weights)
        sum_weights = jnp.sum(weights, axis=1, keepdims=True)
        sum_weights = jnp.maximum(self.eps, sum_weights)
        return weights / sum_weights

    def project_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """
        Project archetypes using soft assignment based on k-nearest neighbors.

        Ensures archetypes remain within the convex hull of data points by creating soft assignments based on proximity.
        This approach offers better stability than hard assignment methods.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)
            X: Original data matrix (n_samples, n_features)

        Returns:
            Projected archetype matrix (n_archetypes, n_features)
        """

        def _process_archetype(i: int) -> jnp.ndarray:
            archetype_dists = dists[:, i]
            top_k_indices = jnp.argsort(archetype_dists)[:k]
            top_k_dists = archetype_dists[top_k_indices]
            weights = 1.0 / (top_k_dists + self.eps)
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

        Identifies optimal archetypes and weights through iterative optimization.
        Uses Adam optimizer with projection steps to ensure constraints are satisfied.

        Args:
            X: Data matrix (n_samples, n_features)
            normalize: Whether to normalize the data before fitting
            **kwargs: Additional keyword arguments for fine-tuning the fitting process

        Returns:
            Self - fitted model instance
        """
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

        X_jax = jnp.array(X_scaled)
        n_samples, _ = X_jax.shape

        self.logger.info(f"Data shape: {X_jax.shape}")
        self.logger.info(f"Data range: min={jnp.min(X_jax):.4f}, max={jnp.max(X_jax):.4f}")

        self.rng_key, subkey = jax.random.split(self.rng_key)
        weights_init = jax.random.uniform(subkey, (n_samples, self.n_archetypes), minval=0.1, maxval=0.9)
        weights_init = self.project_weights(weights_init)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        first_idx = jax.random.randint(subkey, (), 0, n_samples)
        chosen_indices = [int(first_idx)]

        for _ in range(1, self.n_archetypes):
            self.rng_key, subkey = jax.random.split(self.rng_key)

            min_dists_list = []
            for i in range(n_samples):
                # Don't select already chosen points
                if i in chosen_indices:
                    min_dists_list.append(0.0)

                # Find minimum distance to selected archetypes
                else:
                    dist = float("inf")
                    for idx in chosen_indices:
                        d = np.sum((X_scaled[i] - X_scaled[idx]) ** 2)
                        dist = min(dist, d)
                    min_dists_list.append(dist)

            min_dists = np.array(min_dists_list)
            probs = min_dists / (np.sum(min_dists) + self.eps)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)
            chosen_indices.append(int(next_idx))

        archetypes_init = X_jax[jnp.array(chosen_indices)]

        optimizer = optax.adam(learning_rate=self.learning_rate)

        @partial(jax.jit, static_argnums=(3,))
        def update_step(
            params: dict[str, jnp.ndarray], opt_state: optax.OptState, X: jnp.ndarray, iteration: int
        ) -> tuple[dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
            """Execute a single optimization step."""

            def loss_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
                return self.loss_function(params["archetypes"], params["weights"], X)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            for k in grads:
                # Apply gradient clipping to prevent NaNs
                grads[k] = jnp.clip(grads[k], -1.0, 1.0)

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            new_params["weights"] = self.project_weights(new_params["weights"])
            new_params["archetypes"] = self.project_archetypes(new_params["archetypes"], X)

            return new_params, opt_state, loss

        params = {"archetypes": archetypes_init, "weights": weights_init}
        opt_state = optimizer.init(params)

        prev_loss = float("inf")

        initial_loss = float(self.loss_function(archetypes_init, weights_init, X_jax))
        self.logger.info(f"Initial loss: {initial_loss:.6f}")

        for it in range(self.max_iter):
            try:
                params, opt_state, loss = update_step(params, opt_state, X_jax, it)

                loss_value = float(loss)
                self.loss_history.append(loss_value)

                if jnp.isnan(loss_value):
                    self.logger.warning(get_message("warning", "nan_detected", iteration=it))
                    break

                if it > 0 and abs(prev_loss - loss_value) < self.tol:
                    self.logger.info(f"Converged at iteration {it}")
                    break

                if it % 50 == 0:
                    self.logger.info(f"Iteration {it}, Loss: {loss_value:.6f}")

                prev_loss = loss_value

            except Exception as e:
                self.logger.error(f"Error at iteration {it}: {e!s}")
                break

        archetypes_scaled = np.array(params["archetypes"])
        self.archetypes = archetypes_scaled * self.X_std + self.X_mean
        self.weights = np.array(params["weights"])

        if len(self.loss_history) > 0:
            self.logger.info(f"Final loss: {self.loss_history[-1]:.6f}")
        else:
            self.logger.warning("No valid loss was recorded")

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

        X_np = X.values if hasattr(X, "values") else X
        X_jax = jnp.array(X_np)

        if self.normalize:
            X_scaled = (
                (X_jax - self.X_mean) / self.X_std if self.X_mean is not None and self.X_std is not None else X_jax
            )
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            X_scaled = X_jax.copy()

        n_samples = X_scaled.shape[0]
        weights = np.zeros((n_samples, self.n_archetypes))

        if self.normalize:
            archetypes_scaled = (
                (self.archetypes - self.X_mean) / self.X_std
                if self.X_mean is not None and self.X_std is not None
                else self.archetypes
            )
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            archetypes_scaled = self.archetypes

        for i in range(n_samples):
            w = np.ones(self.n_archetypes) / self.n_archetypes

            for _ in range(100):
                pred = np.dot(w, archetypes_scaled)
                error = X_scaled[i] - pred
                grad = -2 * np.dot(error, archetypes_scaled.T)

                w = w - self.learning_rate * grad
                w = np.maximum(1e-10, w)
                sum_w = np.sum(w)
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
        X_np = X.values if hasattr(X, "values") else X
        model = self.fit(X_np, normalize=normalize)
        return np.asarray(model.transform(X_np))

    def reconstruct(self, X: np.ndarray | None = None) -> np.ndarray:
        """
        Reconstruct data using the learned archetypes.

        Args:
            X: Data to reconstruct, or None to use the training data

        Returns:
            Reconstructed data of shape (n_samples, n_features)
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
