"""Evaluation metrics for Archetypal Analysis."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.metrics import davies_bouldin_score, silhouette_score

from ..models.base import ArchetypalAnalysis


class ArchetypalAnalysisEvaluator:
    """
    Evaluator for Archetypal Analysis results, especially for high-dimensional data.

    Provides metrics and visualizations to assess model quality.
    """

    def __init__(self, model: ArchetypalAnalysis):
        """
        Initialize the evaluator.

        Args:
            model: Fitted ArchetypalAnalysis model
        """
        self.model = model
        if model.archetypes is None or model.weights is None:
            raise ValueError("Model must be fitted before evaluation")

        # Cache some frequently used values
        self.n_archetypes = model.archetypes.shape[0]
        self.n_features = model.archetypes.shape[1]
        self.dominant_archetypes = np.argmax(model.weights, axis=1)

    def reconstruction_error(self, X: np.ndarray, metric: str = "frobenius") -> float:
        """
        Calculate the reconstruction error of the model.

        Args:
            X: Data matrix
            metric: Error metric to use ('frobenius', 'mae', 'mse', or 'relative')

        Returns:
            Reconstruction error value
        """
        X_reconstructed = self.model.reconstruct()

        if metric == "frobenius":
            # Frobenius norm (default)
            return float(np.linalg.norm(X - X_reconstructed, ord="fro"))
        elif metric == "mae":
            # Mean Absolute Error
            return float(np.mean(np.abs(X - X_reconstructed)))
        elif metric == "mse":
            # Mean Squared Error
            return float(np.mean((X - X_reconstructed) ** 2))
        elif metric == "relative":
            # Relative error
            return float(np.linalg.norm(X - X_reconstructed, ord="fro") / np.linalg.norm(X, ord="fro"))
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'frobenius', 'mae', 'mse', or 'relative'.")

    def explained_variance(self, X: np.ndarray) -> float:
        """
        Calculate the explained variance of the model.

        Args:
            X: Data matrix

        Returns:
            Explained variance (0-1)
        """
        X_reconstructed = self.model.reconstruct(X)

        # Calculate total variance
        total_variance = np.var(X, axis=0).sum()

        # Calculate residual variance
        residual_variance = np.var(X - X_reconstructed, axis=0).sum()

        # Calculate explained variance
        explained_var = 1.0 - (residual_variance / total_variance)

        return float(explained_var)

    def dominant_archetype_purity(self) -> dict[str, Any]:
        """
        Analyze how dominant each archetype is for its assigned samples.

        Returns:
            Dictionary with purity metrics
        """
        if self.model.weights is None:
            raise ValueError("Model must be fitted before evaluating purity")

        # Get weights for each sample
        weights: np.ndarray = self.model.weights

        # Get maximum weight for each sample
        max_weights = np.max(weights, axis=1)

        # Calculate average purity for each archetype
        archetype_purity = {}
        for i in range(self.n_archetypes):
            archetype_mask = self.dominant_archetypes == i
            if np.sum(archetype_mask) > 0:  # Check if archetype has any assigned samples
                avg_purity = np.mean(max_weights[archetype_mask])
                archetype_purity[f"Archetype_{i}"] = avg_purity

        # Calculate overall purity metrics
        overall_purity = np.mean(max_weights)
        purity_std = np.std(max_weights)

        return {
            "archetype_purity": archetype_purity,
            "overall_purity": overall_purity,
            "purity_std": purity_std,
            "max_weights": max_weights,
        }

    def archetype_separation(self) -> dict[str, float]:
        """
        Measure how well-separated the archetypes are.

        Returns:
            Dictionary with separation metrics
        """
        # Calculate all pairwise distances between archetypes
        archetype_distances = cdist(self.model.archetypes, self.model.archetypes)

        # Fill diagonal with NaN to ignore self-distances
        np.fill_diagonal(archetype_distances, np.nan)

        # Calculate metrics
        min_distance = np.nanmin(archetype_distances)
        max_distance = np.nanmax(archetype_distances)
        mean_distance = np.nanmean(archetype_distances)

        return {
            "min_distance": min_distance,
            "max_distance": max_distance,
            "mean_distance": mean_distance,
            "distance_ratio": min_distance / max_distance if max_distance > 0 else 0,
        }

    def clustering_metrics(self, X: np.ndarray) -> dict[str, float]:
        """
        Calculate clustering quality metrics by using dominant archetypes as cluster assignments.

        Args:
            X: Original data matrix

        Returns:
            Dictionary with clustering metrics
        """
        # Need at least 2 archetypes and more samples than archetypes
        if self.n_archetypes < 2 or X.shape[0] <= self.n_archetypes:
            return {"silhouette": np.nan, "davies_bouldin": np.nan}

        try:
            # Silhouette score (higher is better)
            silhouette = silhouette_score(X, self.dominant_archetypes)

            # Davies-Bouldin index (lower is better)
            davies_bouldin = davies_bouldin_score(X, self.dominant_archetypes)

            return {"silhouette": silhouette, "davies_bouldin": davies_bouldin}
        except Exception as e:
            print(f"Could not compute clustering metrics: {e!s}")
            return {"silhouette": np.nan, "davies_bouldin": np.nan}

    def archetype_feature_importance(self) -> pd.DataFrame:
        """
        Analyze which features are most important for each archetype.

        Returns:
            DataFrame with feature importance for each archetype
        """
        # Get archetypes
        archetypes = self.model.archetypes

        if archetypes is None:
            raise ValueError("Model archetypes must not be None")

        # Calculate feature-wise z-scores for each archetype
        feature_means = np.mean(archetypes, axis=0)
        feature_stds = np.std(archetypes, axis=0)

        # Avoid division by zero
        feature_stds = np.where(feature_stds < 1e-10, 1.0, feature_stds)

        # Calculate z-scores
        feature_importance = np.abs((archetypes - feature_means) / feature_stds)

        # Create DataFrame
        archetype_names = [f"Archetype_{i}" for i in range(self.n_archetypes)]
        feature_names = [f"Feature_{i}" for i in range(self.n_features)]

        return pd.DataFrame(feature_importance, index=archetype_names, columns=feature_names)

    def weight_diversity(self) -> dict[str, float]:
        """
        Measure how diverse the weight distributions are across samples.

        Returns:
            Dictionary with diversity metrics
        """
        weights = self.model.weights

        if weights is None:
            raise ValueError("Model weights must not be None")

        # Calculate entropy for each sample's weight distribution
        sample_entropy = np.array([entropy(w) for w in weights])

        # Theoretical maximum entropy for uniform distribution
        max_entropy = np.log(self.n_archetypes)

        # Normalize entropy (0-1 scale)
        normalized_entropy = sample_entropy / max_entropy

        return {
            "mean_entropy": np.mean(sample_entropy),
            "mean_normalized_entropy": np.mean(normalized_entropy),
            "entropy_std": np.std(sample_entropy),
            "min_entropy": np.min(sample_entropy),
            "max_entropy": np.max(sample_entropy),
        }

    def comprehensive_evaluation(self, X: np.ndarray) -> dict[str, Any]:
        """
        Run all evaluation metrics and return comprehensive results.

        Args:
            X: Original data matrix

        Returns:
            Dictionary with all evaluation metrics
        """
        results = {
            "reconstruction": {
                "frobenius": self.reconstruction_error(X, "frobenius"),
                "mae": self.reconstruction_error(X, "mae"),
                "mse": self.reconstruction_error(X, "mse"),
                "relative": self.reconstruction_error(X, "relative"),
            },
            "explained_variance": self.explained_variance(X),
            "purity": self.dominant_archetype_purity(),
            "separation": self.archetype_separation(),
            "clustering": self.clustering_metrics(X),
            "diversity": self.weight_diversity(),
        }

        return results

    def print_evaluation_report(self, X: np.ndarray) -> None:
        """
        Print a human-readable evaluation report.

        Args:
            X: Original data matrix
        """
        results = self.comprehensive_evaluation(X)

        print("=" * 50)
        print(f"ARCHETYPAL ANALYSIS EVALUATION REPORT ({self.n_archetypes} archetypes, {self.n_features} features)")
        print("=" * 50)

        print("\n1. RECONSTRUCTION QUALITY:")
        print(f"   - Relative Error: {results['reconstruction']['relative']:.4f}")
        print(f"   - Frobenius Norm: {results['reconstruction']['frobenius']:.4f}")
        print(f"   - Mean Absolute Error: {results['reconstruction']['mae']:.4f}")
        print(f"   - Mean Squared Error: {results['reconstruction']['mse']:.4f}")
        print(f"   - Explained Variance: {results['explained_variance']:.4f} (higher is better)")

        print("\n2. ARCHETYPE QUALITY:")
        print(f"   - Average Archetype Purity: {results['purity']['overall_purity']:.4f}")
        print(f"   - Purity Std. Deviation: {results['purity']['purity_std']:.4f}")
        print("   - Per-archetype Purity:")
        for arch, purity in results["purity"]["archetype_purity"].items():
            print(f"     * {arch}: {purity:.4f}")

        print("\n3. ARCHETYPE SEPARATION:")
        print(f"   - Minimum Distance: {results['separation']['min_distance']:.4f}")
        print(f"   - Maximum Distance: {results['separation']['max_distance']:.4f}")
        print(f"   - Mean Distance: {results['separation']['mean_distance']:.4f}")
        print(f"   - Distance Ratio (min/max): {results['separation']['distance_ratio']:.4f}")

        print("\n4. CLUSTERING QUALITY:")
        print(f"   - Silhouette Score: {results['clustering']['silhouette']:.4f} (higher is better)")
        print(f"   - Davies-Bouldin Index: {results['clustering']['davies_bouldin']:.4f} (lower is better)")

        print("\n5. WEIGHT DIVERSITY:")
        print(f"   - Mean Entropy: {results['diversity']['mean_entropy']:.4f}")
        print(f"   - Normalized Entropy: {results['diversity']['mean_normalized_entropy']:.4f} (0-1 scale)")
        print(f"   - Min Entropy: {results['diversity']['min_entropy']:.4f}")
        print(f"   - Max Entropy: {results['diversity']['max_entropy']:.4f}")

        print("\n" + "=" * 50)

    # Visualization methods for high-dimensional data

    def plot_feature_importance_heatmap(self, feature_names: list[str] | None = None) -> None:
        """
        Plot heatmap of feature importance across archetypes.

        Args:
            feature_names: Optional list of feature names
        """
        importance_df = self.archetype_feature_importance()

        # Rename columns if feature names provided
        if feature_names is not None and len(feature_names) == self.n_features:
            importance_df = pd.DataFrame(importance_df.values, index=importance_df.index, columns=feature_names)

        plt.figure(figsize=(12, 8))
        sns.heatmap(importance_df, cmap="viridis", annot=True)
        plt.title("Feature Importance Across Archetypes")
        plt.xlabel("Features")
        plt.ylabel("Archetypes")
        plt.tight_layout()
        plt.show()

    def plot_archetype_feature_comparison(self, top_n: int = 5, feature_names: list[str] | None = None) -> None:
        """
        Plot radar chart or bar chart comparing top N most important features for each archetype.

        Args:
            top_n: Number of top features to display
            feature_names: Optional list of feature names
        """
        importance_df = self.archetype_feature_importance()

        # Rename columns if feature names provided
        if feature_names is not None and len(feature_names) == self.n_features:
            importance_df = pd.DataFrame(importance_df.values, index=importance_df.index, columns=feature_names)

        # For each archetype, get the top N most important features
        plt.figure(figsize=(15, 4 * ((self.n_archetypes + 1) // 2)))

        for i in range(self.n_archetypes):
            # Sort features by importance for this archetype
            archetype_importance = importance_df.iloc[i].sort_values(ascending=False)
            top_features = archetype_importance.head(top_n)

            plt.subplot(((self.n_archetypes + 1) // 2), 2, i + 1)
            bars = plt.bar(
                np.arange(len(top_features)),
                top_features.values.astype(float),
                tick_label=top_features.index,
                color="skyblue",
            )

            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                )

            plt.title(f"Archetype {i}: Top {top_n} Features")
            plt.ylim(0, max(top_features.values) * 1.2)  # Add headroom for text
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        plt.tight_layout()
        plt.show()

    def plot_weight_distributions(self, bins: int = 20) -> None:
        """
        Plot histograms of weight distributions for each archetype.

        Args:
            bins: Number of histogram bins
        """
        weights = self.model.weights
        if weights is None:
            raise ValueError("Model weights must not be None")

        plt.figure(figsize=(15, 4 * ((self.n_archetypes + 1) // 2)))

        for i in range(self.n_archetypes):
            plt.subplot(((self.n_archetypes + 1) // 2), 2, i + 1)

            # Get weights for this archetype
            archetype_weights = weights[:, i]

            # Plot histogram
            plt.hist(archetype_weights, bins=bins, alpha=0.7, color="skyblue")
            plt.title(f"Archetype {i} Weight Distribution")
            plt.xlabel("Weight")
            plt.ylabel("Number of Samples")

            # Add statistics
            plt.axvline(
                np.mean(archetype_weights),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(archetype_weights):.3f}",
            )
            plt.axvline(
                np.median(archetype_weights),
                color="g",
                linestyle="-",
                label=f"Median: {np.median(archetype_weights):.3f}",
            )
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_purity_distribution(self) -> None:
        """Plot the distribution of dominant archetype weights (purity)."""
        purity_data = self.dominant_archetype_purity()
        if "max_weights" not in purity_data:
            raise ValueError("Max weights data is missing")

        max_weights = purity_data["max_weights"]
        if max_weights is None:
            raise ValueError("Max weights is None")

        plt.figure(figsize=(10, 6))

        # Plot histogram
        plt.hist(max_weights, bins=20, alpha=0.7, color="skyblue")
        plt.title("Distribution of Dominant Archetype Weights (Purity)")
        plt.xlabel("Maximum Weight")
        plt.ylabel("Number of Samples")

        # Add statistics
        plt.axvline(
            np.mean(max_weights),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(max_weights):.3f}",
        )
        plt.axvline(
            np.median(max_weights),
            color="g",
            linestyle="-",
            label=f"Median: {np.median(max_weights):.3f}",
        )

        # Theoretical threshold for uniform weights
        uniform_weight = 1.0 / self.n_archetypes
        plt.axvline(
            uniform_weight,
            color="k",
            linestyle=":",
            label=f"Uniform: {uniform_weight:.3f}",
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_distance_matrix(self) -> None:
        """Plot distance matrix between archetypes."""
        # Calculate pairwise distances
        distances = cdist(self.model.archetypes, self.model.archetypes)

        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            distances,
            annot=True,
            cmap="viridis",
            xticklabels=[f"A{i}" for i in range(self.n_archetypes)],
            yticklabels=[f"A{i}" for i in range(self.n_archetypes)],
        )

        plt.title("Distance Matrix Between Archetypes")
        plt.tight_layout()
        plt.show()

    def plot_entropy_vs_reconstruction(self, X: np.ndarray, n_samples: int = 1000) -> None:
        """
        Plot relationship between sample entropy and reconstruction error.

        Args:
            X: Original data matrix
            n_samples: Number of samples to plot (random subset)
        """
        weights = self.model.weights
        X_reconstructed = self.model.reconstruct()

        if weights is None:
            raise ValueError("Model weights must not be None")

        # Calculate point-wise reconstruction error
        point_errors = np.sqrt(np.sum((X - X_reconstructed) ** 2, axis=1))

        # Calculate entropy for each point
        entropies = np.array([entropy(w) for w in weights])

        # Normalize to maximum possible entropy
        max_entropy = np.log(self.n_archetypes)
        normalized_entropies = entropies / max_entropy

        # Select subset if needed
        if n_samples < len(entropies) and n_samples > 0:
            indices = np.random.choice(len(entropies), size=n_samples, replace=False)
            point_errors = point_errors[indices]
            normalized_entropies = normalized_entropies[indices]
            dominant_archetypes = self.dominant_archetypes[indices]
        else:
            dominant_archetypes = self.dominant_archetypes

        plt.figure(figsize=(10, 8))

        # Scatter plot colored by dominant archetype
        scatter = plt.scatter(
            normalized_entropies,
            point_errors,
            c=dominant_archetypes,
            cmap="viridis",
            alpha=0.6,
            s=30,
        )

        # Add color legend
        legend = plt.legend(*scatter.legend_elements(), title="Dominant Archetype")
        plt.gca().add_artist(legend)

        # Add correlation coefficient
        corr = np.corrcoef(normalized_entropies, point_errors)[0, 1]
        plt.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=plt.gca().transAxes,
            bbox={"facecolor": "white", "alpha": 0.8},
        )

        plt.xlabel("Normalized Entropy (Diversity)")
        plt.ylabel("Reconstruction Error")
        plt.title("Relationship Between Weight Diversity and Reconstruction Error")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class BiarchetypalAnalysisEvaluator:
    """
    Evaluator for Biarchetypal Analysis results.

    Provides metrics and visualizations to assess model quality for biarchetypal models,
    which use two sets of archetypes to represent data.
    """

    def __init__(self, model):
        """
        Initialize the evaluator.

        Args:
            model: Fitted BiarchetypalAnalysis model
        """
        from ..models.biarchetypes import BiarchetypalAnalysis

        if not isinstance(model, BiarchetypalAnalysis):
            raise TypeError("Model must be a BiarchetypalAnalysis instance")

        self.model = model

        # Check if model is fitted
        if (
            model.archetypes_first is None
            or model.archetypes_second is None
            or model.weights_first is None
            or model.weights_second is None
        ):
            raise ValueError("Model must be fitted before evaluation")

        # Cache some frequently used values
        self.n_archetypes_first = model.archetypes_first.shape[0]
        self.n_archetypes_second = model.archetypes_second.shape[0]
        self.n_features = model.archetypes_first.shape[1]

        # Calculate dominant archetypes for each set
        self.dominant_archetypes_first = np.argmax(model.weights_first, axis=1)
        self.dominant_archetypes_second = np.argmax(model.weights_second, axis=1)

    def reconstruction_error(self, X: np.ndarray, metric: str = "frobenius") -> float:
        """
        Calculate the reconstruction error of the model.

        Args:
            X: Data matrix
            metric: Error metric to use ('frobenius', 'mae', 'mse', or 'relative')

        Returns:
            Reconstruction error value
        """
        X_reconstructed = self.model.reconstruct(X)

        if metric == "frobenius":
            return float(np.linalg.norm(X - X_reconstructed, ord="fro") / np.sqrt(X.shape[0]))
        elif metric == "mse":
            return float(np.mean((X - X_reconstructed) ** 2))
        elif metric == "mae":
            return float(np.mean(np.abs(X - X_reconstructed)))
        elif metric == "relative":
            return float(np.linalg.norm(X - X_reconstructed, ord="fro") / np.linalg.norm(X, ord="fro"))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def explained_variance(self, X: np.ndarray) -> float:
        """
        Calculate the explained variance of the model.

        Args:
            X: Data matrix

        Returns:
            Explained variance (0-1)
        """
        X_reconstructed = self.model.reconstruct(X)

        # Calculate total variance
        total_variance = np.var(X, axis=0).sum()

        # Calculate residual variance
        residual_variance = np.var(X - X_reconstructed, axis=0).sum()

        # Calculate explained variance
        explained_var = 1.0 - (residual_variance / total_variance)

        return float(explained_var)

    def archetype_separation(self) -> dict:
        """
        Calculate separation metrics between archetypes.

        Returns:
            Dictionary of separation metrics
        """
        # Calculate distances between first set archetypes
        distances_first = cdist(self.model.archetypes_first, self.model.archetypes_first)
        np.fill_diagonal(distances_first, np.inf)  # Ignore self-distances

        # Calculate distances between second set archetypes
        distances_second = cdist(self.model.archetypes_second, self.model.archetypes_second)
        np.fill_diagonal(distances_second, np.inf)  # Ignore self-distances

        # Calculate cross-distances between first and second sets
        cross_distances = cdist(self.model.archetypes_first, self.model.archetypes_second)

        # Calculate metrics for first set
        metrics_first = {
            "mean_distance_first": np.mean(distances_first[distances_first != np.inf]),
            "min_distance_first": np.min(distances_first[distances_first != np.inf]),
            "max_distance_first": np.max(distances_first[distances_first != np.inf]),
        }

        # Calculate metrics for second set
        metrics_second = {
            "mean_distance_second": np.mean(distances_second[distances_second != np.inf])
            if distances_second[distances_second != np.inf].size > 0
            else 0,
            "min_distance_second": np.min(distances_second[distances_second != np.inf])
            if distances_second[distances_second != np.inf].size > 0
            else 0,
            "max_distance_second": np.max(distances_second[distances_second != np.inf])
            if distances_second[distances_second != np.inf].size > 0
            else 0,
        }

        # Calculate cross-set metrics
        metrics_cross = {
            "mean_cross_distance": np.mean(cross_distances),
            "min_cross_distance": np.min(cross_distances),
            "max_cross_distance": np.max(cross_distances),
        }

        # Combine all metrics
        return {**metrics_first, **metrics_second, **metrics_cross}

    def dominant_archetype_purity(self) -> dict:
        """
        Calculate purity metrics for dominant archetypes.

        Returns:
            Dictionary of purity metrics
        """
        # Calculate purity for first set
        archetype_counts_first = np.bincount(self.dominant_archetypes_first, minlength=self.n_archetypes_first)
        archetype_purity_first = archetype_counts_first / np.sum(archetype_counts_first)

        # Calculate purity for second set
        archetype_counts_second = np.bincount(self.dominant_archetypes_second, minlength=self.n_archetypes_second)
        archetype_purity_second = archetype_counts_second / np.sum(archetype_counts_second)

        # Calculate overall metrics
        return {
            "archetype_purity_first": archetype_purity_first,
            "archetype_purity_second": archetype_purity_second,
            "overall_purity_first": np.max(archetype_purity_first) if archetype_purity_first.size > 0 else 0,
            "overall_purity_second": np.max(archetype_purity_second) if archetype_purity_second.size > 0 else 0,
            "purity_std_first": np.std(archetype_purity_first),
            "purity_std_second": np.std(archetype_purity_second),
        }

    def weight_diversity(self) -> dict:
        """
        Calculate diversity metrics for archetype weights.

        Returns:
            Dictionary of diversity metrics
        """
        # Check if weights are None
        if self.model.weights_first is None or self.model.weights_second is None:
            raise ValueError("Model weights must not be None")

        # Calculate entropy for each sample's weights in first set
        entropies_first = np.array([entropy(w) for w in self.model.weights_first])
        max_entropy_first = np.log(self.n_archetypes_first)
        normalized_entropies_first = entropies_first / max_entropy_first

        # Calculate entropy for each sample's weights in second set
        entropies_second = np.array([entropy(w) for w in self.model.weights_second])
        max_entropy_second = np.log(self.n_archetypes_second)
        normalized_entropies_second = entropies_second / max_entropy_second

        return {
            "mean_entropy_first": np.mean(entropies_first),
            "entropy_std_first": np.std(entropies_first),
            "max_entropy_first": np.max(entropies_first),
            "mean_normalized_entropy_first": np.mean(normalized_entropies_first),
            "mean_entropy_second": np.mean(entropies_second),
            "entropy_std_second": np.std(entropies_second),
            "max_entropy_second": np.max(entropies_second),
            "mean_normalized_entropy_second": np.mean(normalized_entropies_second),
        }

    def comprehensive_evaluation(self, X: np.ndarray) -> dict:
        """
        Perform a comprehensive evaluation of the model.

        Args:
            X: Data matrix

        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate reconstruction metrics
        reconstruction_metrics = {
            "frobenius": self.reconstruction_error(X, metric="frobenius"),
            "mse": self.reconstruction_error(X, metric="mse"),
            "mae": self.reconstruction_error(X, metric="mae"),
            "relative": self.reconstruction_error(X, metric="relative"),
            "explained_variance": self.explained_variance(X),
        }

        # Get other metrics
        separation_metrics = self.archetype_separation()
        purity_metrics = self.dominant_archetype_purity()
        diversity_metrics = self.weight_diversity()

        # Combine all metrics
        return {
            "reconstruction": reconstruction_metrics,
            "separation": separation_metrics,
            "purity": purity_metrics,
            "diversity": diversity_metrics,
        }

    def print_evaluation_report(self, X: np.ndarray) -> None:
        """
        Print a comprehensive evaluation report.

        Args:
            X: Data matrix
        """
        results = self.comprehensive_evaluation(X)

        print("===== Biarchetypal Analysis Evaluation Report =====")
        print(f"Number of first set archetypes: {self.n_archetypes_first}")
        print(f"Number of second set archetypes: {self.n_archetypes_second}")
        print(f"Number of features: {self.n_features}")
        print("\n--- Reconstruction Metrics ---")
        print(f"Frobenius Error: {results['reconstruction']['frobenius']:.4f}")
        print(f"MSE: {results['reconstruction']['mse']:.4f}")
        print(f"MAE: {results['reconstruction']['mae']:.4f}")
        print(f"Relative Error: {results['reconstruction']['relative']:.4f}")
        print(f"Explained Variance: {results['reconstruction']['explained_variance']:.4f}")

        print("\n--- Archetype Separation Metrics ---")
        print(f"Mean Distance (First Set): {results['separation']['mean_distance_first']:.4f}")
        print(f"Mean Distance (Second Set): {results['separation']['mean_distance_second']:.4f}")
        print(f"Mean Cross-Set Distance: {results['separation']['mean_cross_distance']:.4f}")

        print("\n--- Purity Metrics ---")
        print(f"Overall Purity (First Set): {results['purity']['overall_purity_first']:.4f}")
        print(f"Overall Purity (Second Set): {results['purity']['overall_purity_second']:.4f}")

        print("\n--- Weight Diversity Metrics ---")
        print(f"Mean Normalized Entropy (First Set): {results['diversity']['mean_normalized_entropy_first']:.4f}")
        print(f"Mean Normalized Entropy (Second Set): {results['diversity']['mean_normalized_entropy_second']:.4f}")
