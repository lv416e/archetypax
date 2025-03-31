"""Advanced visualization tools for extracting insights from archetypal models.

This module provides specialized visualization capabilities that transform abstract
archetypal representations into intuitive visual insights. These visualizations
bridge the gap between mathematical models and human understanding by:

1. Revealing geometric relationships between data points and archetypes
2. Exposing patterns in feature utilization across different archetypes
3. Demonstrating reconstruction quality and model performance
4. Enabling exploration of relationships in both standard and biarchetypal space

These capabilities are essential for model interpretation, result communication,
and extracting actionable insights from archetypal analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..models.base import ArchetypalAnalysis
from ..models.biarchetypes import BiarchetypalAnalysis


class ArchetypalAnalysisVisualizer:
    """Comprehensive visualization suite for archetypal analysis insights.

    This class provides specialized visualization methods that transform abstract
    archetypal models into intuitive visual representations. Rather than just plotting
    data, these methods reveal the underlying structures and relationships discovered
    by archetypal analysis, enabling:

    - Interpretation of archetype meaning and significance
    - Assessment of model quality and reconstruction fidelity
    - Communication of results to technical and non-technical audiences
    - Discovery of patterns in high-dimensional archetypal space

    These visualizations bridge the critical gap between mathematical models and
    human understanding, making archetypal analysis results accessible and actionable.
    """

    @staticmethod
    def plot_loss(model: ArchetypalAnalysis) -> None:
        """Visualize convergence behavior through loss trajectory analysis.

        This diagnostic visualization reveals the optimization dynamics of the model
        by tracking loss values across iterations. It provides critical insights into:

        - Convergence speed and stability
        - Potential issues with learning rates or initialization
        - Evidence of premature convergence or local minima traps
        - Effectiveness of early stopping criteria

        Understanding these dynamics is essential for hyperparameter tuning,
        model validation, and diagnosing unexpected results.

        Args:
            model: Fitted ArchetypalAnalysis model with loss history
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
        """Reveal geometric relationships between data and archetypes in 2D space.

        This visualization exposes the fundamental geometrical interpretation of
        archetypal analysis by showing how archetypes position themselves at the
        extremes of the data distribution and form a convex hull. The plot reveals:

        - Position of archetypes relative to the data cloud
        - Dominance relationships between data points and archetypes
        - The convex hull structure formed by the archetypes
        - Feature-specific patterns that define each archetype

        This representation is particularly valuable for initial model validation,
        intuitive explanation of what archetypes represent, and identification of
        outliers or unexpected patterns.

        Args:
            model: Fitted ArchetypalAnalysis model with discovered archetypes
            X: Original data matrix in 2D space
            feature_names: Optional feature names for meaningful axis labels
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
        """Assess model fidelity through side-by-side reconstruction comparison.

        This visualization provides a direct assessment of how well the archetypal
        model captures the underlying data structure by comparing original and
        reconstructed data points. This comparison reveals:

        - Overall reconstruction quality and information preservation
        - Specific regions where the model performs well or poorly
        - Distortion patterns introduced by dimensionality reduction
        - Evidence of potential overfitting or underfitting

        This assessment is critical for validating model quality, determining
        an appropriate number of archetypes, and communicating the tradeoff
        between interpretability and accuracy.

        Args:
            model: Fitted ArchetypalAnalysis model for reconstruction
            X: Original data matrix to be reconstructed
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
        """Visualize how samples relate to archetypes through weight distribution patterns.

        This heatmap visualization reveals the fundamental composition patterns
        in the data by showing how each sample leverages different archetypes.
        The visualization exposes:

        - Dominant archetypes for each sample
        - Patterns of archetype co-utilization
        - Samples with similar composition profiles
        - Evidence of archetype redundancy or specialization

        These insights are valuable for clustering analysis, identifying
        representative samples, detecting subpopulations, and understanding
        how archetypes interact to represent the data.

        Args:
            model: Fitted ArchetypalAnalysis model with weights
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


class BiarchetypalAnalysisVisualizer:
    """Visualization utilities for Biarchetypal Analysis."""

    @staticmethod
    def plot_dual_archetypes_2d(
        model: BiarchetypalAnalysis, X: np.ndarray, feature_names: list[str] | None = None
    ) -> None:
        """
        Plot data and both sets of archetypes in 2D.

        Args:
            model: Fitted BiarchetypalAnalysis model
            X: Original data
            feature_names: Optional feature names for axis labels
        """
        if X.shape[1] != 2:
            raise ValueError("This plotting function is only for 2D data")

        archetypes_first, archetypes_second = model.get_all_archetypes()
        weights_first, weights_second = model.get_all_weights()

        plt.figure(figsize=(12, 8))

        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], alpha=0.4, color="gray", label="Data")

        # Plot first set of archetypes
        plt.scatter(
            archetypes_first[:, 0],
            archetypes_first[:, 1],
            c="blue",
            s=150,
            marker="*",
            label=f"Archetypes Set 1 (n={model.n_row_archetypes})",
        )

        # Plot second set of archetypes
        plt.scatter(
            archetypes_second[:, 0],
            archetypes_second[:, 1],
            c="red",
            s=150,
            marker="^",
            label=f"Archetypes Set 2 (n={model.n_col_archetypes})",
        )

        # Connect points to their dominant archetypes in each set
        for i in range(min(50, X.shape[0])):  # Limit to 50 arrows for visual clarity
            # Find dominant archetype in first set
            max_idx_first = np.argmax(weights_first[i])
            if weights_first[i, max_idx_first] > 0.5:
                plt.arrow(
                    X[i, 0],
                    X[i, 1],
                    archetypes_first[max_idx_first, 0] - X[i, 0],
                    archetypes_first[max_idx_first, 1] - X[i, 1],
                    head_width=0.01,
                    head_length=0.02,
                    alpha=0.2,
                    color="blue",
                    linestyle="--",
                )

            # Find dominant archetype in second set
            # Handle different shapes of weights_second
            if len(weights_second.shape) == 2 and weights_second.shape[0] == model.n_col_archetypes:
                # If weights_second has shape (n_col_archetypes, n_features)
                weights_second_transposed = weights_second.T
                if i < weights_second_transposed.shape[0]:
                    max_idx_second = np.argmax(weights_second_transposed[i])
                    if weights_second_transposed[i, max_idx_second] > 0.5:
                        plt.arrow(
                            X[i, 0],
                            X[i, 1],
                            archetypes_second[max_idx_second, 0] - X[i, 0],
                            archetypes_second[max_idx_second, 1] - X[i, 1],
                            head_width=0.01,
                            head_length=0.02,
                            alpha=0.2,
                            color="red",
                            linestyle=":",
                        )
            elif len(weights_second.shape) == 2 and i < weights_second.shape[0]:
                # For standard shape
                max_idx_second = np.argmax(weights_second[i])
                if weights_second[i, max_idx_second] > 0.5:
                    plt.arrow(
                        X[i, 0],
                        X[i, 1],
                        archetypes_second[max_idx_second, 0] - X[i, 0],
                        archetypes_second[max_idx_second, 1] - X[i, 1],
                        head_width=0.01,
                        head_length=0.02,
                        alpha=0.2,
                        color="red",
                        linestyle=":",
                    )

        # Add feature names if provided
        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")

        plt.title("Data and Dual Archetype Sets")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_biarchetypal_reconstruction(model: BiarchetypalAnalysis, X: np.ndarray) -> None:
        """
        Plot original data vs. reconstructions from each archetype set and combined.

        Args:
            model: Fitted BiarchetypalAnalysis model
            X: Original data matrix
        """
        if X.shape[1] != 2:
            raise ValueError("This plotting function is only for 2D data")

        # Get weights for both archetype sets
        weights_first, weights_second = model.get_all_weights()
        archetypes_first, archetypes_second = model.get_all_archetypes()

        # Create reconstructions
        X_recon_first = np.matmul(weights_first, archetypes_first)

        # Handle different shapes of weights_second
        if len(weights_second.shape) == 2 and weights_second.shape[0] == model.n_col_archetypes:
            # If weights_second has shape (n_col_archetypes, n_features)
            weights_second_transposed = weights_second.T
            if weights_second_transposed.shape[0] == weights_first.shape[0]:
                X_recon_second = np.matmul(weights_second_transposed, archetypes_second)
            else:
                # Create dummy data with matching shape if shapes don't match
                X_recon_second = np.zeros_like(X_recon_first)
        else:
            # For standard shape
            X_recon_second = np.matmul(weights_second, archetypes_second)

        # Check and adjust shape of X_recon_second if necessary
        if X_recon_second.shape != X_recon_first.shape:
            # Create dummy data with matching shape if shapes don't match
            X_recon_second_temp = X_recon_second.copy()
            X_recon_second = np.zeros_like(X_recon_first)

            # Use original data where possible
            min_rows = min(X_recon_second_temp.shape[0], X_recon_first.shape[0])
            min_cols = min(X_recon_second_temp.shape[1], X_recon_first.shape[1])
            X_recon_second[:min_rows, :min_cols] = X_recon_second_temp[:min_rows, :min_cols]

        # Create combined reconstruction using mixture weight
        if hasattr(model, "mixture_weight"):
            X_recon_combined = model.mixture_weight * X_recon_first + (1 - model.mixture_weight) * X_recon_second
        else:
            # Use equal weights if mixture_weight doesn't exist
            X_recon_combined = 0.5 * X_recon_first + 0.5 * X_recon_second

        # Calculate reconstruction errors
        error_first = np.linalg.norm(X - X_recon_first, ord="fro")
        error_second = np.linalg.norm(X - X_recon_second, ord="fro")
        error_combined = np.linalg.norm(X - X_recon_combined, ord="fro")

        # Create plot with subplots
        _, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original data
        axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.7, label="Original")
        axes[0, 0].set_title("Original Data")
        axes[0, 0].grid(True, alpha=0.3)

        # First archetype set reconstruction
        axes[0, 1].scatter(X_recon_first[:, 0], X_recon_first[:, 1], alpha=0.7, color="blue")
        axes[0, 1].scatter(
            archetypes_first[:, 0],
            archetypes_first[:, 1],
            c="blue",
            s=100,
            marker="*",
            label="Archetypes Set 1",
        )
        axes[0, 1].set_title(f"First Set Reconstruction\nError: {error_first:.4f}")
        axes[0, 1].grid(True, alpha=0.3)

        # Second archetype set reconstruction
        axes[1, 0].scatter(X_recon_second[:, 0], X_recon_second[:, 1], alpha=0.7, color="red")
        axes[1, 0].scatter(
            archetypes_second[:, 0],
            archetypes_second[:, 1],
            c="red",
            s=100,
            marker="^",
            label="Archetypes Set 2",
        )
        axes[1, 0].set_title(f"Second Set Reconstruction\nError: {error_second:.4f}")
        axes[1, 0].grid(True, alpha=0.3)

        # Combined reconstruction
        axes[1, 1].scatter(X_recon_combined[:, 0], X_recon_combined[:, 1], alpha=0.7, color="purple")
        axes[1, 1].scatter(
            archetypes_first[:, 0],
            archetypes_first[:, 1],
            c="blue",
            s=100,
            marker="*",
            label="Archetypes Set 1",
        )
        axes[1, 1].scatter(
            archetypes_second[:, 0],
            archetypes_second[:, 1],
            c="red",
            s=100,
            marker="^",
            label="Archetypes Set 2",
        )

        # Check if mixture_weight exists
        mixture_weight = model.mixture_weight if hasattr(model, "mixture_weight") else 0.5

        axes[1, 1].set_title(f"Combined Reconstruction (w={mixture_weight:.2f})\nError: {error_combined:.4f}")
        axes[1, 1].grid(True, alpha=0.3)

        # Add legend to the last plot
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dual_membership_heatmap(model: BiarchetypalAnalysis, n_samples: int = 50) -> None:
        """
        Plot heatmap of membership weights for both sets of archetypes.

        Args:
            model: Fitted BiarchetypalAnalysis model
            n_samples: Number of samples to visualize
        """
        weights_first, weights_second = model.get_all_weights()

        # Select a subset of samples
        n_samples = min(n_samples, weights_first.shape[0])

        # Sort samples by their dominant archetype in first set only
        max_weight_first_idx = np.argmax(weights_first, axis=1)
        sorted_indices = np.argsort(max_weight_first_idx)[:n_samples]

        # Get the subsets for plotting
        weights_first_subset = weights_first[sorted_indices]

        # Create figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Plot first set heatmap
        sns.heatmap(
            weights_first_subset,
            ax=ax1,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            xticklabels=[f"A1_{i}" for i in range(model.n_row_archetypes)],
            yticklabels=False,
        )
        ax1.set_title(f"First Set Membership Weights (n={model.n_row_archetypes})")
        ax1.set_xlabel("Archetypes (First Set)")
        ax1.set_ylabel("Samples")

        # Plot second set heatmap
        if len(weights_second.shape) == 2 and weights_second.shape[0] == model.n_col_archetypes:
            # Use transposed weights_second
            weights_second_transposed = weights_second.T
            if weights_second_transposed.shape[0] == weights_first.shape[0]:
                weights_second_subset = weights_second_transposed[sorted_indices]
            else:
                # Create matrix of ones if shapes don't match
                weights_second_subset = np.ones((len(sorted_indices), model.n_col_archetypes))
        else:
            # Create matrix of ones for n_col_archetypes=1 case
            weights_second_subset = np.ones((len(sorted_indices), 1))

        sns.heatmap(
            weights_second_subset,
            ax=ax2,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            xticklabels=[f"A2_{i}" for i in range(model.n_col_archetypes)],
            yticklabels=False,
        )
        ax2.set_title(f"Second Set Membership Weights (n={model.n_col_archetypes})")
        ax2.set_xlabel("Archetypes (Second Set)")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mixture_effect(model: BiarchetypalAnalysis, X: np.ndarray, mixture_steps: int = 5) -> None:
        """
        Plot the effect of different mixture weights between the two archetype sets.

        Args:
            model: Fitted BiarchetypalAnalysis model
            X: Original data matrix
            mixture_steps: Number of different mixture weights to try
        """
        if X.shape[1] != 2:
            raise ValueError("This plotting function is only for 2D data")

        # Get weights and archetypes for both sets
        weights_first, weights_second = model.get_all_weights()
        archetypes_first, archetypes_second = model.get_all_archetypes()

        # Create reconstructions for first and second sets
        X_recon_first = np.matmul(weights_first, archetypes_first)

        # Handle different shapes of weights_second
        if len(weights_second.shape) == 2 and weights_second.shape[0] == model.n_col_archetypes:
            # If weights_second has shape (n_col_archetypes, n_features)
            weights_second_transposed = weights_second.T
            if weights_second_transposed.shape[0] == weights_first.shape[0]:
                X_recon_second = np.matmul(weights_second_transposed, archetypes_second)
            else:
                # Create dummy data with matching shape if shapes don't match
                X_recon_second = np.zeros_like(X_recon_first)
                # Display warning
                print("Warning: Shape mismatch in weights_second. Using dummy data for second reconstruction.")
        else:
            # For standard shape
            X_recon_second = np.matmul(weights_second, archetypes_second)
        # Check and adjust shape of X_recon_second if necessary
        if X_recon_second.shape != X_recon_first.shape:
            # Create dummy data with matching shape if shapes don't match
            X_recon_second_temp = X_recon_second.copy()
            X_recon_second = np.zeros_like(X_recon_first)

            # Use original data where possible
            min_rows = min(X_recon_second_temp.shape[0], X_recon_first.shape[0])
            min_cols = min(X_recon_second_temp.shape[1], X_recon_first.shape[1])
            X_recon_second[:min_rows, :min_cols] = X_recon_second_temp[:min_rows, :min_cols]

            # Display warning message
            print(
                f"Warning: Shape mismatch between reconstructions. X_recon_first: {X_recon_first.shape}, X_recon_second: {X_recon_second_temp.shape}"
            )

        # Create figure with subplots
        n_rows = (mixture_steps + 2) // 3  # Ceiling division
        _, axes = plt.subplots(n_rows, min(3, mixture_steps), figsize=(15, 4 * n_rows))

        # Flatten axes if necessary
        if mixture_steps > 3:
            axes = axes.flatten()
        elif mixture_steps == 1:
            axes = [axes]  # Convert to list for single subplot case

        # Original data for reference
        X_original = X.copy()

        # Try different mixture weights
        for i in range(mixture_steps):
            # Calculate mixture weight
            mix_weight = i / (mixture_steps - 1) if mixture_steps > 1 else 0.5

            # Create mixed reconstruction
            X_mixed = mix_weight * X_recon_first + (1 - mix_weight) * X_recon_second

            # Calculate error
            error = np.linalg.norm(X_original - X_mixed, ord="fro")

            # Plot
            ax = axes[i] if mixture_steps > 1 else axes
            ax.scatter(X_original[:, 0], X_original[:, 1], alpha=0.2, color="gray", label="Original")
            ax.scatter(X_mixed[:, 0], X_mixed[:, 1], alpha=0.7, color="purple", label="Reconstructed")
            ax.scatter(
                archetypes_first[:, 0],
                archetypes_first[:, 1],
                c="blue",
                s=80,
                marker="*",
                label="Set 1",
            )
            ax.scatter(
                archetypes_second[:, 0],
                archetypes_second[:, 1],
                c="red",
                s=80,
                marker="^",
                label="Set 2",
            )
            ax.set_title(f"w={mix_weight:.2f}, Error={error:.4f}")

            # Only add legend to the first plot
            if i == 0:
                ax.legend()

            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dual_simplex_2d(model: BiarchetypalAnalysis, n_samples: int = 200) -> None:
        """
        Plot samples in separate 2D simplex spaces for each archetype set (only works for 3 archetypes per set).

        Args:
            model: Fitted BiarchetypalAnalysis model
            n_samples: Number of samples to plot
        """
        # Relaxed condition: at least one of the archetype sets must have exactly 3 archetypes
        if model.n_row_archetypes != 3 and model.n_col_archetypes != 3:
            raise ValueError("This simplex plot requires at least one set to have exactly 3 archetypes")

        # Get weights for both sets
        weights_first, weights_second = model.get_all_weights()

        # Select a subset of samples if needed
        if n_samples < weights_first.shape[0]:
            indices = np.random.choice(weights_first.shape[0], n_samples, replace=False)
            weights_first_subset = weights_first[indices]

            # Handle different shapes of weights_second
            if len(weights_second.shape) == 2 and weights_second.shape[0] == model.n_col_archetypes:
                # If weights_second has shape (n_col_archetypes, n_features)
                weights_second_transposed = weights_second.T
                if weights_second_transposed.shape[0] == weights_first.shape[0]:
                    weights_second_subset = weights_second_transposed[indices]
                else:
                    # Create dummy data if shapes don't match
                    weights_second_subset = np.ones((len(indices), model.n_col_archetypes)) / model.n_col_archetypes
            else:
                # For standard shape
                weights_second_subset = weights_second[indices]
        else:
            weights_first_subset = weights_first

            # Handle different shapes of weights_second
            if len(weights_second.shape) == 2 and weights_second.shape[0] == model.n_col_archetypes:
                # If weights_second has shape (n_col_archetypes, n_features)
                weights_second_transposed = weights_second.T
                if weights_second_transposed.shape[0] == weights_first.shape[0]:
                    weights_second_subset = weights_second_transposed
                else:
                    # Create dummy data if shapes don't match
                    weights_second_subset = (
                        np.ones((weights_first.shape[0], model.n_col_archetypes)) / model.n_col_archetypes
                    )
            else:
                # For standard shape
                weights_second_subset = weights_second

        # Handle case where first archetype set doesn't have exactly 3 archetypes
        if model.n_row_archetypes != 3:
            # Create dummy data
            weights_first_subset = np.ones((weights_first_subset.shape[0], 3)) / 3

        # Handle case where second archetype set doesn't have exactly 3 archetypes
        if model.n_col_archetypes != 3:
            # Create dummy data
            weights_second_subset = np.ones((weights_second_subset.shape[0], 3)) / 3

        # Set up the figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Convert barycentric coordinates to 2D for visualization
        sqrt3_2 = np.sqrt(3) / 2
        triangle_vertices = np.array([
            [0, 0],  # Archetype 0 at origin
            [1, 0],  # Archetype 1 at (1,0)
            [0.5, sqrt3_2],  # Archetype 2 at (0.5, sqrt(3)/2)
        ])

        # Transform weights to 2D coordinates for both sets
        points_2d_first = np.dot(weights_first_subset, triangle_vertices)
        points_2d_second = np.dot(weights_second_subset, triangle_vertices)

        # Create dominant archetype colormaps for each set
        dominant_archetypes_first = np.argmax(weights_first_subset, axis=1)
        dominant_archetypes_second = np.argmax(weights_second_subset, axis=1)

        # Plot first set simplex
        ax1.plot([0, 1, 0.5, 0], [0, 0, sqrt3_2, 0], "k-")
        ax1.scatter(
            points_2d_first[:, 0],
            points_2d_first[:, 1],
            c=dominant_archetypes_first,
            alpha=0.6,
            cmap="Blues",
        )
        ax1.text(-0.05, -0.05, "A1_0", ha="right", color="blue")
        ax1.text(1.05, -0.05, "A1_1", ha="left", color="blue")
        ax1.text(0.5, sqrt3_2 + 0.05, "A1_2", ha="center", color="blue")
        ax1.set_title("First Archetype Set Simplex" + (" (Dummy)" if model.n_row_archetypes != 3 else ""))
        ax1.axis("equal")
        ax1.axis("off")

        # Plot second set simplex
        ax2.plot([0, 1, 0.5, 0], [0, 0, sqrt3_2, 0], "k-")
        ax2.scatter(
            points_2d_second[:, 0],
            points_2d_second[:, 1],
            c=dominant_archetypes_second,
            alpha=0.6,
            cmap="Reds",
        )
        ax2.text(-0.05, -0.05, "A2_0", ha="right", color="red")
        ax2.text(1.05, -0.05, "A2_1", ha="left", color="red")
        ax2.text(0.5, sqrt3_2 + 0.05, "A2_2", ha="center", color="red")
        ax2.set_title("Second Archetype Set Simplex" + (" (Dummy)" if model.n_col_archetypes != 3 else ""))
        ax2.axis("equal")
        ax2.axis("off")

        # Add grid lines for the simplex
        for ax in [ax1, ax2]:
            for i in range(1, 10):
                p = i / 10
                # Line parallel to the bottom edge
                ax.plot(
                    [p * 0.5, p + (1 - p) * 0.5],
                    [p * sqrt3_2, (1 - p) * 0],
                    "gray",
                    alpha=0.3,
                )
                # Line parallel to the left edge
                ax.plot([0, p * 0.5], [p * 0, p * sqrt3_2], "gray", alpha=0.3)
                # Line parallel to the right edge
                ax.plot(
                    [p * 1, 0.5 + (1 - p) * 0.5],
                    [p * 0, (1 - p) * sqrt3_2],
                    "gray",
                    alpha=0.3,
                )

        plt.tight_layout()
        plt.show()
