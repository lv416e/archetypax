"""
ArchetypeTracker: Advanced monitoring system for archetypal optimization dynamics.

This specialized extension captures the complete evolutionary journey of archetypes
during model training, addressing the critical need to understand optimization behavior
beyond final results alone. By recording archetype positions, boundary relationships,
and loss trajectories at each iteration, it enables:

1. Model debugging and verification by revealing optimization bottlenecks
2. Scientific insights into how archetypes discover and adapt to data structures
3. Improved hyperparameter selection based on convergence patterns
4. Compelling visualizations of archetype evolution for stakeholder communications

This capability is particularly valuable when fine-tuning models, understanding
complex datasets, or diagnosing unexpected analysis results.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..logger import get_logger, get_message
from ..models.archetypes import ImprovedArchetypalAnalysis


class ArchetypeTracker(ImprovedArchetypalAnalysis):
    """Advanced archetype monitoring system for optimization trajectory analysis.

    This extension of ImprovedArchetypalAnalysis maintains a complete history of
    archetype positions, loss values, and boundary relationships throughout the
    optimization process. Unlike standard models that only preserve final states,
    this tracker enables in-depth analysis of optimization dynamics, including:

    - Identification of degenerate solutions or local minima traps
    - Visualization of archetype movement paths during convergence
    - Measurement of boundary proximity and stability over time
    - Detection of oscillations or under/over-projection issues

    These capabilities are essential for model debugging, hyperparameter tuning,
    and gaining scientific insights into how archetypes adapt to data structures.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ArchetypeTracker with comprehensive monitoring capabilities.

        Creates a tracker with identical initialization parameters to ImprovedArchetypalAnalysis
        but adds specialized tracking arrays and optimization settings tuned for
        detailed movement analysis. These settings provide more conservative updates
        to better capture transitional states during optimization.

        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class, with optional
                      early_stopping_patience parameter (default: 100)
        """
        super().__init__(*args, **kwargs)
        if isinstance(kwargs.get("logger_level"), str) and kwargs.get("logger_level") is not None:
            logger_level = kwargs["logger_level"].lower()
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
                n_archetypes=self.n_archetypes,
            )
        )

        self.archetype_history = []
        self.loss_history = []
        self.optimizer: optax.GradientTransformation = optax.adam(learning_rate=self.learning_rate)
        # Specific settings for archetype updates
        self.archetype_grad_scale = 1.0  # Gradient scale for archetypes
        self.noise_scale = 0.02  # Magnitude of initial noise
        self.exploration_noise_scale = 0.05  # Magnitude of exploration noise
        # Track position metrics
        self.boundary_proximity_history = []  # History of boundary proximity scores
        self.is_outside_history = []  # History of whether archetypes are outside the convex hull

        self.early_stopping_patience = kwargs.get("early_stopping_patience", 100)

    def fit(self, X: np.ndarray, normalize: bool = False, **kwargs) -> "ArchetypeTracker":
        """Train the model while capturing detailed archetype evolution data.

        This enhanced fitting process extends standard training with comprehensive
        tracking of archetype positions, boundary relationships, and optimization
        metrics at each iteration. The method uses more conservative update settings
        to ensure smooth transitions are captured, while maintaining comparable
        convergence properties to the parent implementation.

        During training, the tracker stores:
        - Complete history of archetype positions at each iteration
        - Loss value progression throughout optimization
        - Boundary proximity measurements over time
        - Detection of archetypes outside the convex hull
        - Detailed movement metrics for stability analysis

        Args:
            X: Data matrix (n_samples, n_features)
            normalize: Whether to normalize features before fitting
            **kwargs: Additional parameters for customizing the fitting process

        Returns:
            Self - fitted model instance with complete archetype history
        """
        # Data preprocessing
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)

        X_np = X.values if hasattr(X, "values") else X
        if self.normalize:
            X_scaled = (X_np - self.X_mean) / self.X_std
            self.logger.info(get_message("data", "normalization", mean=self.X_mean, std=self.X_std))
        else:
            X_scaled = X_np.copy()

        # Convert to JAX array
        X_jax = jnp.array(X_scaled)
        n_samples, n_features = X_jax.shape

        # Store current iteration for use in adaptive projection
        self.current_iteration = 0

        # Initialize archetypes based on selected method
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
            self.current_iteration = i

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

            if i % 50 == 0 or max_change > 1.0:
                self.logger.debug(
                    f"Iteration {i}, Archetype changes: Average={avg_change:.6f}, Maximum={max_change:.6f}"
                )
                # Display indices of archetypes with significant changes
                if max_change > 1.0:  # Set threshold
                    large_changes = np.where(change_norms > 1.0)[0]
                    if len(large_changes) > 0:
                        self.logger.debug(
                            f"  Archetypes with significant changes: {large_changes}, Changes: {change_norms[large_changes]}"
                        )

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
                self.logger.info(
                    get_message(
                        "progress",
                        "early_stopping",
                        iteration=len(self.loss_history),
                        patience=self.early_stopping_patience,
                    )
                )
                break

            # More sensitive loss increase detection for tracker
            if i > 0 and loss_value > prev_loss * 1.02:  # Reduced from 1.05 to 1.02
                archetypes = best_archetypes.copy()
                # Re-initialize optimizer
                opt_state = self.optimizer.init(archetypes)

            # Convergence check
            if i > 0 and abs(prev_loss - loss_value) < self.tol:
                self.logger.info(get_message("progress", "converged", iteration=i, tolerance=self.tol))
                break

            prev_loss = loss_value

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
            self.logger.info(get_message("result", "final_loss", loss=best_loss, iterations=len(self.loss_history)))
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
        self.archetypes = np.array(archetypes) * self.X_std + self.X_mean if self.normalize else np.array(archetypes)
        self.weights = np.array(weights)

        # Scale history if normalized
        if self.normalize:
            self.archetype_history = [arch * self.X_std + self.X_mean for arch in self.archetype_history]

        return self

    def _calculate_weights(self, X: jnp.ndarray, archetypes: jnp.ndarray) -> jnp.ndarray:
        """Calculate optimal weights for given archetypes using JAX.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            archetypes: Archetype matrix of shape (n_archetypes, n_features)

        Returns:
            Weight matrix of shape (n_samples, n_archetypes)
        """

        @jax.jit
        def calculate_single_weight(x_sample: jnp.ndarray, archetypes: jnp.ndarray) -> jnp.ndarray:
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
        return jnp.asarray(jax.vmap(calculate_single_weight, in_axes=(0, None))(X, archetypes))

    def _check_archetypes_outside(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> np.ndarray:
        """Check if archetypes are outside the convex hull.

        Args:
            archetypes: Archetype matrix of shape (n_archetypes, n_features)
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Boolean array of shape (n_archetypes,)
        """
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

    def _constrain_to_convex_hull_batch(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Ensure all archetypes are within the convex hull.

        Args:
            archetypes: Archetype matrix of shape (n_archetypes, n_features)
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Constrained archetype matrix of shape (n_archetypes, n_features)
        """
        return jax.vmap(lambda arch: self._constrain_to_convex_hull(arch, X))(archetypes)

    def _constrain_to_convex_hull(self, archetype: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Constrain a single archetype to be within the convex hull.

        More conservative implementation than the parent class,
        keeping archetypes slightly inside the convex hull boundary.

        Args:
            archetype: Archetype matrix of shape (n_features,)
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Constrained archetype matrix of shape (n_features,)
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
        self, feature_indices: list[int] | None = None, figsize=(12, 8), interval: int = 1
    ) -> Any | None:
        """Visualize how archetypes moved during optimization.

        Args:
            feature_indices: Indices of features to use for 2D plot. If None, PCA is used.
            figsize: Figure size.
            interval: Plot every nth iteration to avoid overcrowding.

        Returns:
            matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.cm import get_cmap

            history_sample = self.archetype_history[::interval]
            loss_sample = self.loss_history[::interval]
            n_iters = len(history_sample)

            # Set up colormap for iterations
            cmap = get_cmap("viridis")
            colors = [cmap(i / max(1, n_iters - 1)) for i in range(n_iters)]

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # If feature indices not provided, use PCA to reduce to 2D
            if feature_indices is None or len(feature_indices) != 2:
                from sklearn.decomposition import PCA

                self.logger.info(get_message("data", "transformation", method="PCA"))

                # Flatten all archetypes from all iterations
                all_archetypes = np.vstack(history_sample)

                # Apply PCA
                pca = PCA(n_components=2)
                all_archetypes_2d = pca.fit_transform(all_archetypes)

                # Reshape back to iterations x archetypes x 2
                archetype_coords = all_archetypes_2d.reshape(n_iters, self.n_archetypes, 2)

                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
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
                        color=colors[i],
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
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Optimization Progress (Iterations)")

            # Add loss curve as inset
            loss_ax = fig.add_axes((0.15, 0.15, 0.25, 0.25))  # [left, bottom, width, height]
            loss_ax.plot(range(0, n_iters * interval, interval), loss_sample, "k-")
            loss_ax.set_title("Loss Function")
            loss_ax.set_xlabel("Iterations")
            loss_ax.set_ylabel("Loss")

            # Add title and legend
            ax.set_title("Archetype Movement During Optimization")
            ax.legend(loc="upper right")

            plt.tight_layout()
            return fig

        except ImportError:
            self.logger.error(
                get_message(
                    "error",
                    "computation_error",
                    error_msg="Matplotlib library is required for visualization",
                )
            )
            return None

    def visualize_boundary_proximity(self, figsize=(10, 5)) -> Any | None:
        """Visualize how close archetypes stayed to the convex hull boundary.

        Args:
            figsize: Figure size.

        Returns:
            matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt

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

        except ImportError:
            self.logger.error(
                get_message(
                    "error",
                    "computation_error",
                    error_msg="Matplotlib library is required for visualization",
                )
            )
            return None

    def project_archetypes_with_adaptive_strength(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Modified projection function that adapts its strength based on the current iteration.

        In early iterations, the projection is very gentle to prevent large movements.
        As training progresses, it gradually increases to the normal projection strength.

        Args:
            archetypes: Archetype matrix of shape (n_archetypes, n_features)
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Projected archetype matrix of shape (n_archetypes, n_features)
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
    def project_archetypes(self, archetypes: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Override parent class projection with adaptive version for tracking.

        Args:
            archetypes: Archetype matrix of shape (n_archetypes, n_features)
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Projected archetype matrix of shape (n_archetypes, n_features)
        """
        return self.project_archetypes_with_adaptive_strength(archetypes, X)
