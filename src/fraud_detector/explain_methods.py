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
        log_dirs = sorted(Path("logs").glob("[*]"))
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

        return {
            "method": "shap",
            "shap_values": shap_values,
            "feature_names": x.columns.tolist(),
            "plot_path": str(plot_path),
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

        return {
            "method": "lime",
            "explanations": explanations,
            "feature_names": x.columns.tolist(),
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
