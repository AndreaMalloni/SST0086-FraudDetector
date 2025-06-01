# src/fraud_detector/evaluation/stability.py
from pathlib import Path
from typing import Any

import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

from fraud_detector.core.logger import LoggingManager


class StabilityAnalyzer:
    def __init__(
        self, model_path: Path, training_data: pd.DataFrame, noise_multiplier: float = 0.01
    ):
        """Initialize the stability analyzer.

        Args:
            model_path: Path to the trained XGBoost model
            training_data: Training dataset used to calculate feature statistics
            noise_multiplier: Multiplier for the standard deviation when adding noise
        """
        self.logger = LoggingManager().get_logger("Explanation")
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        self.noise_multiplier = noise_multiplier

        # Calculate feature statistics from training data
        self.feature_stds = training_data.drop(columns=["Class", "id"]).std()

    def apply_small_perturbation(self, instance: pd.Series) -> pd.Series:
        """Create a perturbed copy of the instance by adding Gaussian noise.

        Args:
            instance: Original instance to perturb

        Returns:
            Perturbed instance
        """
        perturbed = instance.copy()

        # Add Gaussian noise to each feature
        for feature in perturbed.index:
            if feature in ["Class", "id"]:
                continue

            noise = np.random.normal(
                0,  # mean
                self.noise_multiplier * self.feature_stds[feature],  # std
            )
            perturbed[feature] += noise

        return perturbed

    def get_top_k_features(self, feature_importance: dict[str, float], k: int) -> set:
        """Get the set of top k features based on absolute importance values.

        Args:
            feature_importance: Dictionary mapping feature names to their importance scores
            k: Number of top features to select

        Returns:
            Set of top k feature names
        """
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        return {feature for feature, _ in sorted_features[:k]}

    def calculate_jaccard_index(self, set1: set, set2: set) -> float:
        """Calculate the Jaccard index between two sets.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Jaccard index (intersection over union)
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def calculate_stability_metrics(
        self,
        original_importance: dict[str, float],
        perturbed_importance: dict[str, float],
        k_values: tuple[int] = (5, 10, 20),
    ) -> dict[str, float]:
        """Calculate stability metrics for a pair of explanations.

        Args:
            original_importance: Dictionary mapping feature names to their importance scores
            perturbed_importance: Dictionary mapping feature names to their importance scores
            k_values: List of k values for top-k feature comparison

        Returns:
            Dictionary containing stability metrics
        """
        # Calculate cosine similarity
        original_values = np.array(list(original_importance.values()))
        perturbed_values = np.array(list(perturbed_importance.values()))
        cosine_sim = cosine_similarity(
            original_values.reshape(1, -1), perturbed_values.reshape(1, -1)
        )[0][0]

        # Calculate Jaccard indices for different k values
        jaccard_indices = {}
        for k in k_values:
            original_top_k = self.get_top_k_features(original_importance, k)
            perturbed_top_k = self.get_top_k_features(perturbed_importance, k)
            jaccard_indices[f"jaccard_k{k}"] = self.calculate_jaccard_index(
                original_top_k, perturbed_top_k
            )

        return {"cosine_similarity": cosine_sim, **jaccard_indices}

    def analyze_stability(
        self,
        instances: pd.DataFrame,
        explanations: dict[str, Any],
        method: str,
        n_perturbations: int = 5,
    ) -> dict[str, Any]:
        """Analyze stability for multiple instances.

        Args:
            instances: DataFrame containing instances to analyze
            explanations: Dictionary containing explanation results
            method: Explanation method used ('shap' or 'lime')
            n_perturbations: Number of perturbations to perform per instance

        Returns:
            Dictionary containing stability analysis results
        """
        results = {
            "cosine_similarities": [],
            "jaccard_indices": {k: [] for k in [5, 10, 20]},
            "plots": [],
        }

        for idx, instance in instances.iterrows():
            instance_cosine_sims = []
            instance_jaccard_indices = {k: [] for k in [5, 10, 20]}

            # Get original feature importance
            if method == "shap":
                original_importance = dict(
                    zip(explanations["feature_names"], explanations["shap_values"][idx])
                )
            else:  # LIME
                original_importance = dict(explanations["explanations"][idx]["explanation"])

            # Perform multiple perturbations
            for _ in range(n_perturbations):
                # Create perturbed instance
                perturbed = self.apply_small_perturbation(instance)

                # Get prediction for perturbed instance
                # dmatrix = xgb.DMatrix(perturbed.drop(['Class', 'id']).to_frame().T)
                # perturbed_pred = self.model.predict(dmatrix)[0]

                # Generate explanation for perturbed instance
                if method == "shap":
                    explainer = shap.TreeExplainer(self.model)
                    perturbed_values = explainer.shap_values(
                        perturbed.drop(["Class", "id"]).to_frame().T
                    )[0]
                    perturbed_importance = dict(
                        zip(explanations["feature_names"], perturbed_values)
                    )
                else:  # LIME
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        instances.drop(columns=["Class", "id"]).values,
                        feature_names=explanations["feature_names"],
                        class_names=["Not Fraud", "Fraud"],
                        mode="classification",
                    )
                    perturbed_exp = explainer.explain_instance(
                        perturbed.drop(["Class", "id"]).values,
                        lambda x: self.model.predict(xgb.DMatrix(x)),
                        num_features=10,
                    )
                    perturbed_importance = dict(perturbed_exp.as_list())

                # Calculate stability metrics
                metrics = self.calculate_stability_metrics(
                    original_importance, perturbed_importance
                )

                instance_cosine_sims.append(metrics["cosine_similarity"])
                for k in [5, 10, 20]:
                    instance_jaccard_indices[k].append(metrics[f"jaccard_k{k}"])

            # Average metrics over perturbations
            results["cosine_similarities"].append(np.mean(instance_cosine_sims))
            for k in [5, 10, 20]:
                results["jaccard_indices"][k].append(np.mean(instance_jaccard_indices[k]))

            # Create stability plot
            self._plot_stability_metrics(instance_cosine_sims, instance_jaccard_indices, idx)

        return results

    def _plot_stability_metrics(
        self, cosine_sims: list[float], jaccard_indices: dict[int, list[float]], idx: int
    ):
        """Create and save stability metrics plot.

        Args:
            cosine_sims: List of cosine similarities
            jaccard_indices: Dictionary mapping k to lists of Jaccard indices
            idx: Instance index
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot cosine similarities
        ax1.hist(cosine_sims, bins=10, alpha=0.7)
        ax1.set_title("Cosine Similarity Distribution")
        ax1.set_xlabel("Cosine Similarity")
        ax1.set_ylabel("Frequency")

        # Plot Jaccard indices
        for k, indices in jaccard_indices.items():
            ax2.hist(indices, bins=10, alpha=0.7, label=f"k={k}")
        ax2.set_title("Jaccard Index Distribution")
        ax2.set_xlabel("Jaccard Index")
        ax2.set_ylabel("Frequency")
        ax2.legend()

        plt.tight_layout()

        # Save plot
        log_dirs = sorted(Path("logs").glob("*"))
        if not log_dirs:
            raise RuntimeError("No log directory found")
        log_dir = log_dirs[-1]
        plots_dir = log_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plots_dir / f"stability_metrics_{idx}.png"
        plt.savefig(plot_path)
        plt.close()
