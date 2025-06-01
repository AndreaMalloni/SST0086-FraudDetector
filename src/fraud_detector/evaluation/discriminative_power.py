from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from fraud_detector.core.logger import LoggingManager


class DiscriminativePowerAnalyzer:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize the discriminative power analyzer.

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.logger = LoggingManager().get_logger("Explanation")
        self.test_size = test_size
        self.random_state = random_state

    def prepare_explanation_features(
        self, explanations: dict[str, Any], method: str
    ) -> pd.DataFrame:
        """Convert explanations into feature matrix.

        Args:
            explanations: Dictionary containing explanation results
            method: Explanation method used ('shap' or 'lime')

        Returns:
            DataFrame containing explanation features
        """
        if method == "shap":
            # For SHAP, we already have the values in the correct format
            return pd.DataFrame(explanations["shap_values"], columns=explanations["feature_names"])
        else:  # LIME
            # For LIME, we need to convert the list of (feature, weight) pairs
            # into a feature matrix
            feature_matrix = []
            for exp in explanations["explanations"]:
                # Create a dictionary of feature weights
                feature_weights = dict(exp["explanation"])
                # Ensure all features are present; some might be missing in individual explanations
                row = dict.fromkeys(explanations["feature_names"], 0.0)
                row.update(feature_weights)
                feature_matrix.append(row)
            return pd.DataFrame(feature_matrix)

    def train_and_evaluate(
        self, x_explanations: pd.DataFrame, y_true: pd.Series, model_type: str = "logistic"
    ) -> dict[str, float]:
        """Train and evaluate a simple model on explanation features.

        Args:
            x_explanations: DataFrame containing explanation features
            y_true: Series containing true labels
            model_type: Type of model to use ('logistic' or 'tree')

        Returns:
            Dictionary containing evaluation metrics
        """
        # Verify we have both classes
        unique_classes = y_true.unique()
        if len(unique_classes) < 2:
            self.logger.warning(
                f"Only found {len(unique_classes)} class(es) in the data. \
                    Skipping discriminative power analysis."
            )
            return {
                "f1_score": 0.0,
                "auc_pr": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "accuracy": 0.0,
            }

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(
            x_explanations,
            y_true,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_true,
        )

        # Initialize and train the model
        if model_type == "logistic":
            model = LogisticRegression(class_weight="balanced", random_state=self.random_state)
        else:  # tree
            model = DecisionTreeClassifier(
                max_depth=3,  # Keep it simple and interpretable
                class_weight="balanced",
                random_state=self.random_state,
            )

        try:
            model.fit(x_train, y_train)

            # Make predictions
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)[:, 1]

            # Calculate metrics
            metrics = {
                "f1_score": f1_score(y_test, y_pred),
                "auc_pr": average_precision_score(y_test, y_pred_proba),
                "recall": recall_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "accuracy": accuracy_score(y_test, y_pred),
            }

            # If using logistic regression, add feature importance
            if model_type == "logistic":
                metrics["feature_importance"] = dict(
                    zip(x_explanations.columns, np.abs(model.coef_[0]))
                )

            return metrics
        except Exception as e:
            self.logger.error(f"Error during model training: {e!s}")
            return {
                "f1_score": 0.0,
                "auc_pr": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "accuracy": 0.0,
            }

    def analyze_discriminative_power(
        self, explanations: dict[str, Any], y_true: pd.Series, method: str
    ) -> dict[str, Any]:
        """Analyze discriminative power of explanations.

        Args:
            explanations: Dictionary containing explanation results
            y_true: Series containing true labels
            method: Explanation method used ('shap' or 'lime')

        Returns:
            Dictionary containing analysis results
        """
        # Prepare explanation features
        x_explanations = self.prepare_explanation_features(explanations, method)

        # Train and evaluate both models
        logistic_metrics = self.train_and_evaluate(x_explanations, y_true, model_type="logistic")
        tree_metrics = self.train_and_evaluate(x_explanations, y_true, model_type="tree")

        # Create visualization
        self._plot_metrics_comparison(logistic_metrics, tree_metrics, method)

        return {
            "logistic_regression": {
                "metrics": {
                    k: v for k, v in logistic_metrics.items() if k != "feature_importance"
                },
                "feature_importance": logistic_metrics.get("feature_importance", {}),
            },
            "decision_tree": {"metrics": tree_metrics},
        }

    def _plot_metrics_comparison(
        self, logistic_metrics: dict[str, float], tree_metrics: dict[str, float], method: str
    ):
        """Create and save metrics comparison plot.

        Args:
            logistic_metrics: Metrics from logistic regression
            tree_metrics: Metrics from decision tree
            method: Explanation method used
        """
        # Prepare data for plotting
        metrics = ["f1_score", "auc_pr", "recall", "precision", "accuracy"]
        logistic_values = [logistic_metrics[m] for m in metrics]
        tree_values = [tree_metrics[m] for m in metrics]

        # Create plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width / 2, logistic_values, width, label="Logistic Regression")
        plt.bar(x + width / 2, tree_values, width, label="Decision Tree")

        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title(f"Discriminative Power Metrics - {method.upper()}")
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save plot
        log_dirs = sorted(Path("logs").glob("*"))
        if not log_dirs:
            raise RuntimeError("No log directory found")
        log_dir = log_dirs[-1]
        plots_dir = log_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plots_dir / f"discriminative_power_{method}.png"
        plt.savefig(plot_path)
        plt.close()
