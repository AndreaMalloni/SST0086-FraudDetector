from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from fraud_detector.core.logger import LoggingManager
from fraud_detector.evaluation.discriminative_power import DiscriminativePowerAnalyzer
from fraud_detector.evaluation.fidelity import FidelityAnalyzer
from fraud_detector.evaluation.stability import StabilityAnalyzer


class ExplanationMethod(ABC):
    """Abstract base class for explanation methods."""

    @abstractmethod
    def explain(self, model_path: Path, data: pd.DataFrame) -> dict[str, Any]:
        """Generate explanations for the given model and data.

        Args:
            model_path: Path to the model file
            data: DataFrame containing the data to explain

        Returns:
            Dictionary containing explanation results
        """


class SHAPExplanation(ExplanationMethod):
    """SHAP-based explanation method."""

    def explain(self, model_path: Path, data: pd.DataFrame) -> dict[str, Any]:
        logger = LoggingManager().get_logger("Explanation")

        # Load the model
        booster = xgb.Booster()
        booster.load_model(str(model_path))

        # Prepare data excluding target and ID columns
        x = data.drop(columns=["Class", "id"])

        # Generate SHAP explanations
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(x[:100])

        # Create and save the summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x[:100], show=False)
        plt.tight_layout()

        # Get the current log directory
        log_dirs = sorted(Path("logs").glob("*"))
        if not log_dirs:
            raise RuntimeError(
                "No log directory found. Make sure the application is properly initialized."
            )
        log_dir = log_dirs[-1]

        # Ensure plots directory exists
        plots_dir = log_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save the plot
        plot_path = plots_dir / f"shap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"SHAP summary plot saved to: {plot_path}")

        # Log feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame(
            {"Feature": x.columns, "Importance": mean_shap_values}
        ).sort_values("Importance", ascending=False)

        logger.info("\nFeature Importance (SHAP values):")
        for _, row in feature_importance.iterrows():
            logger.info(f"{row['Feature']}: {row['Importance']:.4f}")

        # Fidelity analysis
        fidelity_analyzer = FidelityAnalyzer(model_path, data)
        fidelity_results = fidelity_analyzer.analyze_fidelity(
            data[:100],  # Analyze first 100 instances
            {"feature_names": x.columns.tolist(), "shap_values": shap_values},
            "shap",
        )

        # Stability analysis
        stability_analyzer = StabilityAnalyzer(model_path, data)
        stability_results = stability_analyzer.analyze_stability(
            data[:100],  # Analyze first 100 instances
            {"feature_names": x.columns.tolist(), "shap_values": shap_values},
            "shap",
        )

        # Discriminative power analysis
        discriminative_analyzer = DiscriminativePowerAnalyzer()

        # Get balanced sample of instances for discriminative power analysis
        fraud_samples = data[data["Class"] == 1].sample(n=50, random_state=42)
        non_fraud_samples = data[data["Class"] == 0].sample(n=50, random_state=42)
        balanced_samples = pd.concat([fraud_samples, non_fraud_samples])

        discriminative_results = discriminative_analyzer.analyze_discriminative_power(
            {"feature_names": x.columns.tolist(), "shap_values": shap_values},
            balanced_samples["Class"],  # Use balanced samples
            "shap",
        )

        logger.info("\nFidelity Analysis Results:")
        logger.info(f"Mean AUFC: {np.mean(fidelity_results['aufc_scores']):.4f}")
        logger.info(f"Mean Correlation: {np.mean(fidelity_results['correlations']):.4f}")

        logger.info("\nStability Analysis Results:")
        logger.info(
            f"Mean Cosine Similarity: \
                    {np.mean(stability_results['cosine_similarities']):.4f}"
        )
        for k in [5, 10, 20]:
            logger.info(
                f"Mean Jaccard Index (k={k}): \
                        {np.mean(stability_results['jaccard_indices'][k]):.4f}"
            )

        logger.info("\nDiscriminative Power Analysis Results:")
        logger.info("Logistic Regression Metrics:")
        for metric, value in discriminative_results["logistic_regression"]["metrics"].items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("\nDecision Tree Metrics:")
        for metric, value in discriminative_results["decision_tree"]["metrics"].items():
            logger.info(f"{metric}: {value:.4f}")

        return {
            "method": "shap",
            "shap_values": shap_values,
            "feature_names": x.columns.tolist(),
            "plot_path": str(plot_path),
            "fidelity_analysis": {
                "mean_aufc": float(np.mean(fidelity_results["aufc_scores"])),
                "mean_correlation": float(np.mean(fidelity_results["correlations"])),
                "aufc_scores": fidelity_results["aufc_scores"],
                "correlations": fidelity_results["correlations"],
            },
            "stability_analysis": {
                "mean_cosine_similarity": float(np.mean(stability_results["cosine_similarities"])),
                "mean_jaccard_indices": {
                    str(k): float(np.mean(stability_results["jaccard_indices"][k]))
                    for k in [5, 10, 20]
                },
                "cosine_similarities": stability_results["cosine_similarities"],
                "jaccard_indices": stability_results["jaccard_indices"],
            },
            "discriminative_power_analysis": discriminative_results,
        }


class LIMEExplanation(ExplanationMethod):
    """LIME-based explanation method."""

    def explain(self, model_path: Path, data: pd.DataFrame) -> dict[str, Any]:
        logger = LoggingManager().get_logger("Explanation")

        # Load the model
        booster = xgb.Booster()
        booster.load_model(str(model_path))

        # Prepare data excluding target and ID columns
        x = data.drop(columns=["Class", "id"])

        # Create a prediction function that LIME can use
        def predict_fn(x):
            dmatrix = xgb.DMatrix(x)
            # Get probability predictions for positive class
            pos_probs = booster.predict(dmatrix, output_margin=False)
            # Create array with probabilities for both classes
            probs = np.zeros((len(pos_probs), 2))
            probs[:, 0] = 1 - pos_probs  # Probability for negative class
            probs[:, 1] = pos_probs  # Probability for positive class
            return probs

        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            x.values,
            feature_names=x.columns,
            class_names=["Not Fraud", "Fraud"],
            mode="classification",
        )

        # Get the current log directory
        log_dirs = sorted(Path("logs").glob("[*]"))
        if not log_dirs:
            raise RuntimeError(
                "No log directory found. Make sure the application is properly initialized."
            )
        log_dir = log_dirs[-1]

        # Ensure plots directory exists
        plots_dir = log_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate explanations for a few examples
        explanations = []
        for i in range(min(5, len(x))):
            # Get the predicted probabilities
            probs = predict_fn(x.iloc[i : i + 1].values)[0]
            pred_class = np.argmax(probs)

            # Generate explanation for the predicted class
            exp = explainer.explain_instance(
                x.iloc[i].values,
                predict_fn,
                num_features=10,
                labels=[pred_class],  # Explain only the predicted class
            )

            # Create and save the explanation plot
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure(label=pred_class)
            plt.tight_layout()

            # Save the plot
            plot_path = (
                plots_dir / f"lime_explanation_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"\nExplanation for instance {i}:")
            logger.info(f"Predicted class: {'Fraud' if pred_class == 1 else 'Not Fraud'}")
            logger.info(f"Prediction probability: {probs[pred_class]:.4f}")
            logger.info("Feature contributions:")
            for feature, weight in exp.as_list(label=pred_class):
                logger.info(f"{feature}: {weight:.4f}")
            logger.info(f"Explanation plot saved to: {plot_path}")

            explanations.append(
                {
                    "instance": i,
                    "prediction": {
                        "class": "Fraud" if pred_class == 1 else "Not Fraud",
                        "probability": float(probs[pred_class]),
                    },
                    "explanation": exp.as_list(label=pred_class),
                    "plot_path": str(plot_path),
                }
            )

        # Fidelity analysis
        fidelity_analyzer = FidelityAnalyzer(model_path, data)
        fidelity_results = fidelity_analyzer.analyze_fidelity(
            data[:5],  # Analyze the same 5 instances we explained
            {"explanations": explanations},
            "lime",
        )

        # Stability analysis
        stability_analyzer = StabilityAnalyzer(model_path, data)
        stability_results = stability_analyzer.analyze_stability(
            data[:5],  # Analyze the same 5 instances we explained
            {"explanations": explanations},
            "lime",
        )

        # Discriminative power analysis
        discriminative_analyzer = DiscriminativePowerAnalyzer()
        discriminative_results = discriminative_analyzer.analyze_discriminative_power(
            {"explanations": explanations, "feature_names": x.columns.tolist()},
            data[:5]["Class"],  # Use the same 5 instances
            "lime",
        )

        logger.info("\nFidelity Analysis Results:")
        logger.info(f"Mean AUFC: {np.mean(fidelity_results['aufc_scores']):.4f}")
        logger.info(f"Mean Correlation: {np.mean(fidelity_results['correlations']):.4f}")

        logger.info("\nStability Analysis Results:")
        logger.info(
            f"Mean Cosine Similarity: \
                    {np.mean(stability_results['cosine_similarities']):.4f}"
        )
        for k in [5, 10, 20]:
            logger.info(
                f"Mean Jaccard Index (k={k}): \
                        {np.mean(stability_results['jaccard_indices'][k]):.4f}"
            )

        logger.info("\nDiscriminative Power Analysis Results:")
        logger.info("Logistic Regression Metrics:")
        for metric, value in discriminative_results["logistic_regression"]["metrics"].items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("\nDecision Tree Metrics:")
        for metric, value in discriminative_results["decision_tree"]["metrics"].items():
            logger.info(f"{metric}: {value:.4f}")

        return {
            "method": "lime",
            "explanations": explanations,
            "feature_names": x.columns.tolist(),
            "fidelity_analysis": {
                "mean_aufc": float(np.mean(fidelity_results["aufc_scores"])),
                "mean_correlation": float(np.mean(fidelity_results["correlations"])),
                "aufc_scores": fidelity_results["aufc_scores"],
                "correlations": fidelity_results["correlations"],
            },
            "stability_analysis": {
                "mean_cosine_similarity": float(np.mean(stability_results["cosine_similarities"])),
                "mean_jaccard_indices": {
                    str(k): float(np.mean(stability_results["jaccard_indices"][k]))
                    for k in [5, 10, 20]
                },
                "cosine_similarities": stability_results["cosine_similarities"],
                "jaccard_indices": stability_results["jaccard_indices"],
            },
            "discriminative_power_analysis": discriminative_results,
        }


def get_explanation_method(method: str) -> ExplanationMethod:
    """Factory function to get the appropriate explanation method.

    Args:
        method: Name of the explanation method ('shap' or 'lime')

    Returns:
        Instance of the appropriate ExplanationMethod

    Raises:
        ValueError: If the method is not supported
    """
    methods = {"shap": SHAPExplanation(), "lime": LIMEExplanation()}

    if method.lower() not in methods:
        raise ValueError(
            f"Unsupported explanation method: {method}. "
            f"Supported methods are: {', '.join(methods.keys())}"
        )

    return methods[method.lower()]
