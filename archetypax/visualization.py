"""Visualization utilities for Archetypal Analysis."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .base import ArchetypalAnalysis


class ArchetypalAnalysisVisualizer:
    """Visualization utilities for Archetypal Analysis."""

    @staticmethod
    def plot_loss(model: ArchetypalAnalysis) -> None:
        """
        Plot the loss history from training.

        Args:
            model: Fitted ArchetypalAnalysis model
        """
        loss_history = model.get_loss_history()
        if not loss_history:
            print("No loss history to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Archetypal Analysis Loss History")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_archetypes_2d(model: ArchetypalAnalysis, X: np.ndarray, feature_names: list[str] | None = None) -> None:
        """
        Plot data and archetypes in 2D.

        Args:
            model: Fitted ArchetypalAnalysis model
            X: Original data
            feature_names: Optional feature names for axis labels
        """
        from scipy.spatial import ConvexHull

        if model.archetypes is None:
            raise ValueError("Model must be fitted before plotting")

        if model.weights is None:
            raise ValueError("Model must be fitted before plotting")

        if X.shape[1] != 2:
            raise ValueError("This plotting function is only for 2D data")

        weights: np.ndarray = model.weights

        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Data")
        plt.scatter(
            model.archetypes[:, 0],
            model.archetypes[:, 1],
            c="red",
            s=100,
            marker="*",
            label="Archetypes",
        )

        # Add arrows from data points to their dominant archetypes
        for i in range(min(100, len(X))):  # Show max 100 arrows for performance
            # Find the archetype with the highest weight
            if weights is not None and model.archetypes is not None:
                max_idx = np.argmax(weights[i])
                if weights[i, max_idx] > 0.5:  # Only draw if weight is significant
                    plt.arrow(
                        X[i, 0],
                        X[i, 1],
                        model.archetypes[max_idx, 0] - X[i, 0],
                        model.archetypes[max_idx, 1] - X[i, 1],
                        head_width=0.01,
                        head_length=0.02,
                        alpha=0.1,
                        color="grey",
                    )

        # Show convex hull
        if len(model.archetypes) >= 3:
            try:
                hull = ConvexHull(model.archetypes)
                for simplex in hull.simplices:
                    plt.plot(model.archetypes[simplex, 0], model.archetypes[simplex, 1], "r-")
            except Exception as e:
                print(f"Could not plot convex hull: {e!s}")

        # Add feature names if provided
        if feature_names is not None and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")

        plt.legend()
        plt.title("Data and Archetypes")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_reconstruction_comparison(model: ArchetypalAnalysis, X: np.ndarray) -> None:
        """
        Plot original vs reconstructed data.

        Args:
            model: Fitted ArchetypalAnalysis model
            X: Original data matrix
        """
        if model.archetypes is None:
            raise ValueError("Model must be fitted before plotting")

        if X.shape[1] != 2:
            raise ValueError("This plotting function is only for 2D data")

        # Reconstruct data
        X_reconstructed = model.reconstruct()

        # Calculate reconstruction error
        error = np.linalg.norm(X - X_reconstructed, ord="fro")
        print(f"Reconstruction error: {error:.6f}")

        # Plot reconstruction
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7, label="Original")
        plt.title("Original Data")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(
            X_reconstructed[:, 0],
            X_reconstructed[:, 1],
            alpha=0.7,
            label="Reconstructed",
        )
        plt.scatter(
            model.archetypes[:, 0],
            model.archetypes[:, 1],
            c="red",
            s=100,
            marker="*",
            label="Archetypes",
        )
        plt.title("Reconstructed Data")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_membership_weights(model: ArchetypalAnalysis, n_samples: int | None = None) -> None:
        """
        Plot membership weights for samples.

        Args:
            model: Fitted ArchetypalAnalysis model
            n_samples: Optional number of samples to visualize (default: all)
        """
        if model.archetypes is None or model.weights is None:
            raise ValueError("Model must be fitted before plotting membership weights")

        weights = model.weights

        if n_samples is not None:
            # Select a subset of samples if specified
            n_samples = min(n_samples, weights.shape[0])
            # Sort samples by their max weight for better visualization
            max_weight_idx = np.argmax(weights, axis=1)
            sorted_indices = np.argsort(max_weight_idx)
            sample_indices = sorted_indices[:n_samples]
            weights_subset = weights[sample_indices]
        else:
            # Use all samples, but sort them for better visualization
            max_weight_idx = np.argmax(weights, axis=1)
            sorted_indices = np.argsort(max_weight_idx)
            weights_subset = weights[sorted_indices]
            n_samples = weights.shape[0]

        plt.figure(figsize=(12, 8))

        # Create a heatmap of the membership weights
        ax = sns.heatmap(
            weights_subset,
            cmap="viridis",
            annot=True,
            vmin=0,
            vmax=1,
            yticklabels=False,
        )
        ax.set_xlabel("Archetypes")
        ax.set_ylabel("Samples")
        ax.set_title(f"Membership Weights for {n_samples} Samples")

        # Add archetype indices as x-tick labels
        plt.xticks(
            np.arange(model.n_archetypes) + 0.5,
            labels=[f"A{i}" for i in range(model.n_archetypes)],
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_archetype_profiles(model: ArchetypalAnalysis, feature_names: list[str] | None = None) -> None:
        """
        Plot feature profiles of each archetype.

        Args:
            model: Fitted ArchetypalAnalysis model
            feature_names: Optional list of feature names for axis labels
        """
        if model.archetypes is None:
            raise ValueError("Model must be fitted before plotting archetype profiles")

        n_archetypes, n_features = model.archetypes.shape

        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]

        # Prepare feature indices for the x-axis
        x = np.arange(n_features)

        plt.figure(figsize=(12, 8))

        # Plot each archetype as a line
        for i in range(n_archetypes):
            plt.plot(x, model.archetypes[i], marker="o", label=f"Archetype {i}")

        plt.xlabel("Features")
        plt.ylabel("Feature Value")
        plt.title("Archetype Feature Profiles")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Set feature names as x-tick labels if not too many
        if n_features <= 20:
            plt.xticks(x, feature_names, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_archetype_distribution(model: ArchetypalAnalysis) -> None:
        """
        Plot the distribution of dominant archetypes across samples.

        Args:
            model: Fitted ArchetypalAnalysis model
        """
        if model.weights is None:
            raise ValueError("Model must be fitted before plotting archetype distribution")

        # Find the dominant archetype for each sample
        dominant_archetypes = np.argmax(model.weights, axis=1)

        # Count occurrences of each archetype as dominant
        unique, counts = np.unique(dominant_archetypes, return_counts=True)

        plt.figure(figsize=(10, 6))

        # Create a bar plot
        bars = plt.bar(
            range(model.n_archetypes),
            [counts[list(unique).index(i)] if i in unique else 0 for i in range(model.n_archetypes)],
            color="skyblue",
            alpha=0.7,
        )

        # Add labels and percentages
        total_samples = len(dominant_archetypes)
        for bar in bars:
            height = bar.get_height()
            percentage = 100 * height / total_samples
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height} ({percentage:.1f}%)",
                ha="center",
                va="bottom",
                rotation=0,
            )

        plt.xlabel("Archetype")
        plt.ylabel("Number of Samples")
        plt.title("Distribution of Dominant Archetypes")
        plt.xticks(range(model.n_archetypes), [f"A{i}" for i in range(model.n_archetypes)])
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_simplex_2d(model: ArchetypalAnalysis, n_samples: int | None = 500) -> None:
        """
        Plot samples in 2D simplex space (only works for 3 archetypes).

        Args:
            model: Fitted ArchetypalAnalysis model
            n_samples: Number of samples to plot (default: 500)
        """
        if model.archetypes is None or model.weights is None:
            raise ValueError("Model must be fitted before plotting simplex")

        if model.n_archetypes != 3:
            raise ValueError("Simplex plot only works for exactly 3 archetypes")

        # Select a subset of samples if specified
        weights = model.weights
        if n_samples is not None and n_samples < weights.shape[0]:
            indices = np.random.choice(weights.shape[0], n_samples, replace=False)
            weights_subset = weights[indices]
        else:
            weights_subset = weights

        # Convert barycentric coordinates to 2D for visualization
        # For a 3-simplex, we can use an equilateral triangle
        # Where each vertex represents an archetype
        sqrt3_2 = np.sqrt(3) / 2
        triangle_vertices = np.array([
            [0, 0],  # Archetype 0 at origin
            [1, 0],  # Archetype 1 at (1,0)
            [0.5, sqrt3_2],  # Archetype 2 at (0.5, sqrt(3)/2)
        ])

        # Transform weights to 2D coordinates
        points_2d = np.dot(weights_subset, triangle_vertices)

        # Create a colormap based on which archetype has the highest weight
        dominant_archetypes = np.argmax(weights_subset, axis=1)

        plt.figure(figsize=(10, 8))

        # Plot the simplex boundaries
        plt.plot([0, 1, 0.5, 0], [0, 0, sqrt3_2, 0], "k-")

        # Add vertex labels
        plt.text(-0.05, -0.05, "Archetype 0", ha="right")
        plt.text(1.05, -0.05, "Archetype 1", ha="left")
        plt.text(0.5, sqrt3_2 + 0.05, "Archetype 2", ha="center")

        # Plot points colored by dominant archetype
        scatter = plt.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            c=dominant_archetypes,
            alpha=0.6,
            cmap="viridis",
        )

        # Add a color legend
        legend1 = plt.legend(*scatter.legend_elements(), title="Dominant Archetype")
        plt.gca().add_artist(legend1)

        # Add grid lines for the simplex
        for i in range(1, 10):
            p = i / 10
            # Line parallel to the bottom edge
            plt.plot(
                [p * 0.5, p + (1 - p) * 0.5],
                [p * sqrt3_2, (1 - p) * 0],
                "gray",
                alpha=0.3,
            )
            # Line parallel to the left edge
            plt.plot([0, p * 0.5], [p * 0, p * sqrt3_2], "gray", alpha=0.3)
            # Line parallel to the right edge
            plt.plot(
                [p * 1, 0.5 + (1 - p) * 0.5],
                [p * 0, (1 - p) * sqrt3_2],
                "gray",
                alpha=0.3,
            )

        plt.axis("equal")
        plt.title("Samples in Simplex Space")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
