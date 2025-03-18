"""Interpretability metrics for Archetypal Analysis."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..models.archetypes import ImprovedArchetypalAnalysis
from ..models.base import ArchetypalAnalysis
from ..models.biarchetypes import BiarchetypalAnalysis
from ..tools.evaluation import ArchetypalAnalysisEvaluator


class ArchetypalAnalysisInterpreter:
    """
    Interpreter for Archetypal Analysis results, focusing on interpretability metrics.

    Provides quantitative measures for archetype interpretability and optimal number selection.
    """

    def __init__(self, models_dict: dict[int, ArchetypalAnalysis] | None = None) -> None:
        """
        Initialize the interpreter.

        Args:
            models_dict: Optional dictionary of {n_archetypes: model} pairs
        """
        self.models_dict: dict[int, ArchetypalAnalysis] | None = models_dict or None
        self.results: dict[int, dict[str, Any]] | None = None

    def add_model(
        self, n_archetypes: int, model: ArchetypalAnalysis
    ) -> "ArchetypalAnalysisInterpreter":
        """Add a fitted model to the interpreter."""
        if model.archetypes is None or model.weights is None:
            raise ValueError("Model must be fitted before adding to interpreter")

        if self.models_dict is None:
            self.models_dict = {}

        self.models_dict[n_archetypes] = model
        return self

    def feature_distinctiveness(self, archetypes: np.ndarray) -> np.ndarray:
        """
        Calculate how distinctive each archetype is in terms of feature values.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)

        Returns:
            Array of distinctiveness scores for each archetype
        """
        n_archetypes, n_features = archetypes.shape
        distinctiveness_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Calculate the difference between this archetype's values and the maximum values of other archetypes
            other_archetypes = np.delete(archetypes, i, axis=0)
            max_others = (
                np.max(other_archetypes, axis=0)
                if len(other_archetypes) > 0
                else np.zeros(n_features)
            )
            distinctiveness = archetypes[i] - max_others

            # Sum the positive differences (features that are particularly prominent in this archetype)
            distinctiveness_scores[i] = np.sum(np.maximum(0, distinctiveness))

        return distinctiveness_scores

    def sparsity_coefficient(self, archetypes: np.ndarray, percentile: float = 80) -> np.ndarray:
        """
        Calculate sparsity of each archetype's feature representation.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)
            percentile: Percentile threshold for considering features as prominent

        Returns:
            Array of sparsity scores for each archetype (higher is more interpretable)
        """
        n_archetypes, n_features = archetypes.shape
        sparsity_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Calculate feature importance (e.g., Z-scores)
            importance = np.abs(archetypes[i])
            if np.std(importance) > 1e-10:  # Standardize only if variance is non-zero
                importance = (importance - np.mean(importance)) / np.std(importance)

            # Calculate the proportion of features above a certain percentile
            threshold = np.percentile(importance, percentile)
            prominent_features = np.sum(importance >= threshold)
            sparsity_scores[i] = prominent_features / n_features

        # Lower scores indicate higher sparsity (better interpretability)
        return 1 - sparsity_scores

    def cluster_purity(
        self, weights: np.ndarray, threshold: float = 0.6
    ) -> tuple[np.ndarray, float]:
        """
        Calculate purity of each archetype's associated data points.

        Args:
            weights: Weight matrix (n_samples, n_archetypes)
            threshold: Threshold for considering an archetype as dominant

        Returns:
            Tuple of (purity scores per archetype, average purity)
        """
        n_samples, n_archetypes = weights.shape
        purity_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Count samples where this archetype is dominant
            dominant_samples = np.sum(weights[:, i] >= threshold)
            purity_scores[i] = dominant_samples / n_samples

        return np.asarray(purity_scores), float(np.mean(purity_scores))

    def information_gain(self, X: np.ndarray) -> list[tuple[int, float]]:
        """
        Calculate information gain when adding each additional archetype.

        Args:
            X: Original data matrix

        Returns:
            List of (n_archetypes, gain) pairs
        """
        if not self.models_dict:
            raise ValueError("No models available for information gain calculation")

        ks = sorted(self.models_dict.keys())
        gains = []

        for i in range(1, len(ks)):
            prev_k = ks[i - 1]
            curr_k = ks[i]

            prev_error = self.models_dict[prev_k].reconstruct(X)
            prev_error = np.mean(np.sum((X - prev_error) ** 2, axis=1))

            curr_error = self.models_dict[curr_k].reconstruct(X)
            curr_error = np.mean(np.sum((X - curr_error) ** 2, axis=1))

            # Calculate error reduction rate (information gain)
            gain = (prev_error - curr_error) / prev_error if prev_error > 0 else 0
            gains.append((curr_k, gain))

        return gains

    def feature_consistency(
        self,
        X: np.ndarray,
        n_archetypes: int,
        n_trials: int = 5,
        top_k: int = 5,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Calculate feature importance consistency across multiple initializations.

        Args:
            X: Original data matrix
            n_archetypes: Number of archetypes to evaluate
            n_trials: Number of different initializations to try
            top_k: Number of top features to consider
            random_seed: Base random seed (will be incremented for each trial)

        Returns:
            Array of consistency scores for each archetype
        """
        importance_matrices = []

        for i in range(n_trials):
            model = ImprovedArchetypalAnalysis(
                n_archetypes=n_archetypes, random_seed=random_seed + i
            )
            model.fit(X)

            # Calculate feature importance matrix
            evaluator = ArchetypalAnalysisEvaluator(model)
            importance = evaluator.archetype_feature_importance().values

            # Store feature importance rankings for each archetype
            rankings = np.argsort(-importance, axis=1)
            importance_matrices.append(rankings)

        # Calculate consistency of top-K features
        consistency_scores = np.zeros(n_archetypes)
        top_k = min(top_k, X.shape[1])

        for i in range(n_archetypes):
            overlap_count = 0.0
            for j in range(n_trials):
                for k in range(j + 1, n_trials):
                    set1 = set(importance_matrices[j][i, :top_k])
                    set2 = set(importance_matrices[k][i, :top_k])
                    overlap = len(set1.intersection(set2))
                    overlap_count += overlap / top_k

            total_comparisons = (n_trials * (n_trials - 1)) / 2
            consistency_scores[i] = (
                overlap_count / total_comparisons if total_comparisons > 0 else 0
            )

        return consistency_scores

    def evaluate_all_models(self, X: np.ndarray) -> dict[int, dict[str, Any]]:
        """
        Evaluate interpretability metrics for all models.

        Args:
            X: Original data matrix

        Returns:
            Dictionary of results per number of archetypes
        """
        if not self.models_dict:
            raise ValueError("No models available for evaluation")

        if self.results is None:
            self.results = {}

        for k, model in self.models_dict.items():
            if model.archetypes is None:
                raise ValueError(f"Model with {k} archetypes must be fitted before evaluation")

            if model.weights is None:
                raise ValueError(f"Model with {k} archetypes must be fitted before evaluation")

            if k not in self.results:
                self.results[k] = {}

            # Calculate various interpretability metrics
            distinctiveness = self.feature_distinctiveness(model.archetypes)
            sparsity = self.sparsity_coefficient(model.archetypes)
            purity, avg_purity = self.cluster_purity(model.weights)

            # Calculate average metrics
            avg_distinctiveness = np.mean(distinctiveness)
            avg_sparsity = np.mean(sparsity)

            # Calculate overall interpretability score (higher is better)
            interpretability_score = (avg_distinctiveness + avg_sparsity + avg_purity) / 3

            self.results[k] = {
                "distinctiveness": distinctiveness,
                "sparsity": sparsity,
                "purity": purity,
                "avg_distinctiveness": avg_distinctiveness,
                "avg_sparsity": avg_sparsity,
                "avg_purity": avg_purity,
                "interpretability_score": interpretability_score,
            }

        # Calculate information gain
        try:
            gains = self.information_gain(X)
            for k, gain in gains:
                if k in self.results:
                    self.results[k]["information_gain"] = gain

            # Evaluate balance between interpretability and information gain
            for k in list(self.models_dict.keys())[1:]:  # Skip the first model
                if "information_gain" in self.results[k]:
                    gain = self.results[k]["information_gain"]
                    interp = self.results[k]["interpretability_score"]

                    # Calculate balance score using harmonic mean
                    if gain + interp > 0:
                        self.results[k]["balance_score"] = 2 * (gain * interp) / (gain + interp)
                    else:
                        self.results[k]["balance_score"] = 0
        except Exception as e:
            print(f"Warning: Could not compute information gain: {e}")

        return self.results

    def suggest_optimal_archetypes(self, method: str = "balance") -> int:
        """
        Suggest optimal number of archetypes based on interpretability metrics.

        Args:
            method: Method to use for selection ('balance', 'interpretability', or 'information_gain')

        Returns:
            Optimal number of archetypes
        """
        if not self.results:
            raise ValueError("Must run evaluate_all_models() first")

        if self.models_dict is None:
            raise ValueError("No models available for optimal archetype selection")

        if method == "balance" and all(
            "balance_score" in self.results[k] for k in list(self.models_dict.keys())[1:]
        ):
            scores = {
                k: self.results[k]["balance_score"] for k in list(self.models_dict.keys())[1:]
            }
            best_k = max(scores, key=lambda k: scores[k])

        elif method == "interpretability":
            scores = {
                k: self.results[k]["interpretability_score"] for k in list(self.models_dict.keys())
            }
            best_k = max(scores, key=lambda k: scores[k])

        elif method == "information_gain" and all(
            "information_gain" in self.results[k] for k in list(self.models_dict.keys())[1:]
        ):
            # Detect decay in information gain (elbow method)
            ks = sorted(
                k for k in list(self.models_dict.keys()) if k > min(list(self.models_dict.keys()))
            )
            gains = [self.results[k]["information_gain"] for k in ks]

            # Calculate differences in information gain
            gain_diffs = np.diff(gains)
            if len(gain_diffs) > 0:
                # Detect the largest decrease
                elbow_idx = np.argmin(gain_diffs)
                best_k = ks[elbow_idx + 1]  # +1 because diff reduces array size by 1
            else:
                best_k = min(self.models_dict.keys())
        else:
            raise ValueError(f"Method '{method}' not applicable with current results")

        return int(best_k)

    def plot_interpretability_metrics(self):
        """Plot interpretability metrics for different numbers of archetypes."""
        if not self.results:
            raise ValueError("Must run evaluate_all_models() first")

        ks = sorted(self.results.keys())

        # Prepare metrics for plotting
        avg_distinctiveness = [self.results[k]["avg_distinctiveness"] for k in ks]
        avg_sparsity = [self.results[k]["avg_sparsity"] for k in ks]
        avg_purity = [self.results[k]["avg_purity"] for k in ks]
        interpretability = [self.results[k]["interpretability_score"] for k in ks]

        information_gain = []
        for k in ks[1:]:  # Skip first k as it has no information gain
            information_gain.append(self.results[k].get("information_gain", np.nan))

        balance_scores = []
        for k in ks[1:]:  # Skip first k
            balance_scores.append(self.results[k].get("balance_score", np.nan))

        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        # Plot interpretability metrics
        axes[0].plot(ks, avg_distinctiveness, "o-", label="Distinctiveness")
        axes[0].plot(ks, avg_sparsity, "s-", label="Sparsity")
        axes[0].plot(ks, avg_purity, "^-", label="Purity")
        axes[0].plot(ks, interpretability, "D-", label="Overall Interpretability")
        axes[0].set_xlabel("Number of Archetypes")
        axes[0].set_ylabel("Score")
        axes[0].set_title("Interpretability Metrics vs Number of Archetypes")
        axes[0].legend()
        axes[0].grid(True)

        # Plot information gain
        if len(information_gain) > 0 and not all(np.isnan(information_gain)):
            axes[1].plot(ks[1:], information_gain, "o-")
            axes[1].set_xlabel("Number of Archetypes")
            axes[1].set_ylabel("Information Gain")
            axes[1].set_title("Information Gain from Adding Archetypes")
            axes[1].grid(True)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No information gain data available",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Plot balance score
        if len(balance_scores) > 0 and not all(np.isnan(balance_scores)):
            axes[2].plot(ks[1:], balance_scores, "o-")
            axes[2].set_xlabel("Number of Archetypes")
            axes[2].set_ylabel("Balance Score")
            axes[2].set_title("Interpretability-Information Gain Balance")
            axes[2].grid(True)

            # Highlight best k according to balance score
            if not all(np.isnan(balance_scores)):
                best_idx = np.nanargmax(balance_scores)
                best_k = ks[1:][best_idx]
                axes[2].axvline(best_k, color="r", linestyle="--", label=f"Optimal k={best_k}")
                axes[2].legend()
        else:
            axes[2].text(
                0.5,
                0.5,
                "No balance score data available",
                horizontalalignment="center",
                verticalalignment="center",
            )
        plt.tight_layout()
        plt.show()


class BiarchetypalAnalysisInterpreter:
    """
    Interpreter for Biarchetypal Analysis results, focusing on interpretability metrics.

    Provides quantitative measures for biarchetype interpretability and optimal number selection.
    """

    def __init__(
        self, models_dict: dict[tuple[int, int], BiarchetypalAnalysis] | None = None
    ) -> None:
        """
        Initialize the interpreter.

        Args:
            models_dict: Optional dictionary of {(n_archetypes_first, n_archetypes_second): model} pairs
        """
        self.models_dict: dict[tuple[int, int], BiarchetypalAnalysis] = models_dict or {}
        self.results: dict[tuple[int, int], dict[str, Any]] = {}

    def add_model(
        self, n_archetypes_first: int, n_archetypes_second: int, model: BiarchetypalAnalysis
    ) -> "BiarchetypalAnalysisInterpreter":
        """Add a fitted model to the interpreter."""
        # Verify that the model is fitted by using the get_all_archetypes method
        try:
            model.get_all_archetypes()
        except ValueError as e:
            raise ValueError(f"Model must be fitted before adding to interpreter: {e}") from e

        self.models_dict[(n_archetypes_first, n_archetypes_second)] = model
        return self

    def feature_distinctiveness(self, archetypes: np.ndarray) -> np.ndarray:
        """
        Calculate how distinctive each archetype is in terms of feature values.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)

        Returns:
            Array of distinctiveness scores for each archetype
        """
        n_archetypes, n_features = archetypes.shape
        distinctiveness_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Calculate the difference between this archetype's values and the maximum values of other archetypes
            other_archetypes = np.delete(archetypes, i, axis=0)
            max_others = (
                np.max(other_archetypes, axis=0)
                if len(other_archetypes) > 0
                else np.zeros(n_features)
            )
            distinctiveness = archetypes[i] - max_others

            # Sum the positive differences (features that are particularly prominent in this archetype)
            distinctiveness_scores[i] = np.sum(np.maximum(0, distinctiveness))

        return distinctiveness_scores

    def sparsity_coefficient(self, archetypes: np.ndarray, percentile: float = 80) -> np.ndarray:
        """
        Calculate sparsity of each archetype's feature representation.

        Args:
            archetypes: Archetype matrix (n_archetypes, n_features)
            percentile: Percentile threshold for considering features as prominent

        Returns:
            Array of sparsity scores for each archetype (higher is more interpretable)
        """
        n_archetypes, n_features = archetypes.shape
        sparsity_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Calculate feature importance (e.g., Z-scores)
            importance = np.abs(archetypes[i])
            if np.std(importance) > 1e-10:  # Standardize only if variance is non-zero
                importance = (importance - np.mean(importance)) / np.std(importance)

            # Calculate proportion of features above the specified percentile
            threshold = np.percentile(importance, percentile)
            prominent_features = np.sum(importance >= threshold)
            sparsity_scores[i] = prominent_features / n_features

        # Lower scores indicate higher sparsity (better interpretability)
        return 1 - sparsity_scores

    def cluster_purity(
        self, weights: np.ndarray, threshold: float = 0.6
    ) -> tuple[np.ndarray, float]:
        """
        Calculate purity of each archetype's associated data points.

        Args:
            weights: Weight matrix (n_samples, n_archetypes)
            threshold: Threshold for considering an archetype as dominant

        Returns:
            Tuple of (purity scores per archetype, average purity)
        """
        n_samples, n_archetypes = weights.shape
        purity_scores = np.zeros(n_archetypes)

        for i in range(n_archetypes):
            # Count samples where this archetype is dominant
            dominant_samples = np.sum(weights[:, i] >= threshold)
            purity_scores[i] = dominant_samples / n_samples

        return purity_scores, float(np.mean(purity_scores))

    def evaluate_all_models(self, X: np.ndarray) -> dict[tuple[int, int], dict[str, Any]]:
        """
        Evaluate interpretability metrics for all models.

        Args:
            X: Original data matrix

        Returns:
            Dictionary of results per combination of archetypes
        """
        if not self.models_dict:
            raise ValueError("No models available for evaluation")

        self.results = {}

        for (k1, k2), model in self.models_dict.items():
            try:
                # Retrieve row and column archetypes using the get_all_archetypes method
                archetypes_first, archetypes_second = model.get_all_archetypes()

                # Retrieve row and column weights using the get_all_weights method
                weights_first, weights_second = model.get_all_weights()
            except ValueError as e:
                raise ValueError(
                    f"Model with archetypes ({k1}, {k2}) must be fitted before evaluation: {e}"
                ) from e

            # First archetype set interpretability metrics
            distinctiveness_first = self.feature_distinctiveness(np.array(archetypes_first))
            sparsity_first = self.sparsity_coefficient(np.array(archetypes_first))
            purity_first, avg_purity_first = self.cluster_purity(np.array(weights_first))

            # Second archetype set interpretability metrics
            distinctiveness_second = self.feature_distinctiveness(np.array(archetypes_second))
            sparsity_second = self.sparsity_coefficient(np.array(archetypes_second))
            purity_second, avg_purity_second = self.cluster_purity(np.array(weights_second))

            # Calculate averages
            avg_distinctiveness_first = np.mean(distinctiveness_first)
            avg_sparsity_first = np.mean(sparsity_first)

            avg_distinctiveness_second = np.mean(distinctiveness_second)
            avg_sparsity_second = np.mean(sparsity_second)

            # Interpretability scores (higher is better)
            interpretability_first = (
                avg_distinctiveness_first + avg_sparsity_first + avg_purity_first
            ) / 3
            interpretability_second = (
                avg_distinctiveness_second + avg_sparsity_second + avg_purity_second
            ) / 3

            # Combined score for both sets
            combined_interpretability = (interpretability_first + interpretability_second) / 2

            # Calculate reconstruction error
            X_recon = model.reconstruct(X)
            recon_error = np.mean(np.sum((X - X_recon) ** 2, axis=1))

            self.results[(k1, k2)] = {
                # First archetype set
                "distinctiveness_first": distinctiveness_first,
                "sparsity_first": sparsity_first,
                "purity_first": purity_first,
                "avg_distinctiveness_first": avg_distinctiveness_first,
                "avg_sparsity_first": avg_sparsity_first,
                "avg_purity_first": avg_purity_first,
                "interpretability_first": interpretability_first,
                # Second archetype set
                "distinctiveness_second": distinctiveness_second,
                "sparsity_second": sparsity_second,
                "purity_second": purity_second,
                "avg_distinctiveness_second": avg_distinctiveness_second,
                "avg_sparsity_second": avg_sparsity_second,
                "avg_purity_second": avg_purity_second,
                "interpretability_second": interpretability_second,
                # Combined scores
                "combined_interpretability": combined_interpretability,
                "reconstruction_error": recon_error,
            }

        # Calculate information gain
        self.compute_information_gain(X)

        return self.results

    def compute_information_gain(self, X: np.ndarray) -> None:
        """
        Calculate information gain between different archetype number combinations.

        Args:
            X: Original data matrix
        """
        if len(self.models_dict) <= 1:
            return  # At least two models are needed for comparison

        # Find the combination with minimum number of archetypes
        min_k1 = min(k1 for k1, _ in self.models_dict)
        min_k2 = min(k2 for _, k2 in self.models_dict)

        # Error of the baseline model
        if (min_k1, min_k2) in self.models_dict:
            base_model = self.models_dict[(min_k1, min_k2)]
            base_recon = base_model.reconstruct(X)
            base_error = np.mean(np.sum((X - base_recon) ** 2, axis=1))
        else:
            print("Warning: Base model not found for information gain calculation")
            return

        # Calculate information gain for each model
        for (k1, k2), _model in self.models_dict.items():
            if (k1, k2) == (min_k1, min_k2):
                continue  # Skip the baseline model

            model_error = self.results[(k1, k2)]["reconstruction_error"]
            gain = (base_error - model_error) / base_error if base_error > 0 else 0
            self.results[(k1, k2)]["information_gain"] = gain

            # Balance score between information gain and interpretability
            interp = self.results[(k1, k2)]["combined_interpretability"]
            if gain + interp > 0:
                self.results[(k1, k2)]["balance_score"] = (
                    2 * (gain * interp) / (gain + interp)
                )  # Harmonic mean
            else:
                self.results[(k1, k2)]["balance_score"] = 0

    def suggest_optimal_biarchetypes(self, method: str = "balance") -> tuple[int, int]:
        """
        Suggest optimal archetype number combination based on interpretability metrics.

        Args:
            method: Method to use for selection ('balance', 'interpretability', or 'information_gain')

        Returns:
            Optimal combination of (n_archetypes_first, n_archetypes_second)
        """
        if not self.results:
            raise ValueError("Must run evaluate_all_models() first")

        if method == "balance":
            # Only use models that have a balance_score
            scores: dict[tuple[int, int], float] = {}
            for k in self.models_dict:
                if "balance_score" in self.results[k]:
                    scores[k] = self.results[k]["balance_score"]

            if scores:  # Ensure scores is not empty
                best_k = max(scores.items(), key=lambda x: x[1])[0]
            else:
                # Fall back to interpretability if balance scores aren't available
                return self.suggest_optimal_biarchetypes(method="interpretability")

        elif method == "interpretability":
            scores = {k: self.results[k]["combined_interpretability"] for k in self.models_dict}
            best_k = max(scores.items(), key=lambda x: x[1])[0]

        elif method == "information_gain":
            # Only use models that have information_gain
            scores = {}
            min_k = min(self.models_dict.keys(), key=lambda x: x[0] + x[1])

            for k in self.models_dict:
                if k != min_k and "information_gain" in self.results[k]:
                    scores[k] = self.results[k]["information_gain"]

            if scores:  # Ensure scores is not empty
                best_k = max(scores.items(), key=lambda x: x[1])[0]
            else:
                # Fall back to interpretability if information gain isn't available
                return self.suggest_optimal_biarchetypes(method="interpretability")
        else:
            raise ValueError(f"Method '{method}' not applicable with current results")

        return best_k

    def plot_interpretability_heatmap(self) -> plt.Figure:
        """
        Plot heatmaps of interpretability metrics for different archetype number combinations.

        Returns:
            The matplotlib figure object
        """
        if not self.results:
            raise ValueError("Must run evaluate_all_models() first")

        # Get available archetype number combinations
        k1_values = sorted({k1 for k1, _ in self.models_dict})
        k2_values = sorted({k2 for _, k2 in self.models_dict})

        # Store interpretability scores in matrix form
        interpretability_matrix = np.zeros((len(k1_values), len(k2_values)))
        balance_matrix = np.zeros((len(k1_values), len(k2_values)))
        error_matrix = np.zeros((len(k1_values), len(k2_values)))

        # Prepare data
        for i, k1 in enumerate(k1_values):
            for j, k2 in enumerate(k2_values):
                if (k1, k2) in self.results:
                    interpretability_matrix[i, j] = self.results[(k1, k2)][
                        "combined_interpretability"
                    ]
                    if "balance_score" in self.results[(k1, k2)]:
                        balance_matrix[i, j] = self.results[(k1, k2)]["balance_score"]
                    error_matrix[i, j] = self.results[(k1, k2)]["reconstruction_error"]

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Interpretability score heatmap
        sns.heatmap(
            interpretability_matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            xticklabels=k2_values,
            yticklabels=k1_values,
            ax=axes[0],
        )
        axes[0].set_xlabel("Number of Second Archetypes")
        axes[0].set_ylabel("Number of First Archetypes")
        axes[0].set_title("Combined Interpretability Score")

        # Balance score heatmap
        sns.heatmap(
            balance_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            xticklabels=k2_values,
            yticklabels=k1_values,
            ax=axes[1],
        )
        axes[1].set_xlabel("Number of Second Archetypes")
        axes[1].set_ylabel("Number of First Archetypes")
        axes[1].set_title("Interpretability-Information Gain Balance")

        # Reconstruction error heatmap
        sns.heatmap(
            error_matrix,
            annot=True,
            fmt=".3f",
            cmap="rocket_r",
            xticklabels=k2_values,
            yticklabels=k1_values,
            ax=axes[2],
        )
        axes[2].set_xlabel("Number of Second Archetypes")
        axes[2].set_ylabel("Number of First Archetypes")
        axes[2].set_title("Reconstruction Error")

        plt.tight_layout()
        return fig
