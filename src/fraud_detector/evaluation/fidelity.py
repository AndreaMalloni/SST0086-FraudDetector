from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_detector.core.logger import LoggingManager


class FidelityAnalyzer:
	def __init__(self, model_path: Path, training_data: pd.DataFrame):
		"""Initialize the fidelity analyzer.

		Args:
			model_path: Path to the trained XGBoost model
			training_data: Training dataset used to calculate feature statistics
		"""
		self.logger = LoggingManager().get_logger("Explanation")
		self.model = xgb.Booster()
		self.model.load_model(str(model_path))

		# Calculate feature statistics from training data
		self.feature_means = training_data.drop(columns=["Class", "id"]).mean()
		self.feature_medians = training_data.drop(columns=["Class", "id"]).median()

	def perturb_features(
		self, instance: pd.Series, feature_importance: dict[str, float], k: int
	) -> pd.Series:
		"""Create a perturbed copy of the instance by replacing top k features.

		Args:
			instance: Original instance to perturb
			feature_importance: Dictionary mapping feature names to their importance scores
			k: Number of top features to perturb

		Returns:
			Perturbed instance
		"""
		# Sort features by importance
		sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

		# Create a copy of the instance
		perturbed = instance.copy()

		# Replace top k features with their median values
		for feature, _ in sorted_features[:k]:
			perturbed[feature] = self.feature_medians[feature]

		return perturbed

	def calculate_fidelity_metrics(
		self, instance: pd.Series, feature_importance: dict, original_pred: float
	) -> tuple[float, float]:
		"""Calculate fidelity metrics for a single instance.

		Args:
			instance: The instance to analyze
			feature_importance: Dictionary mapping feature names to their importance values
			original_pred: Original prediction for the instance

		Returns:
			Tuple of (AUFC score, correlation coefficient)
		"""
		# Get features excluding 'id' and 'Class'
		features = [col for col in instance.index if col not in ["id", "Class"]]
		instance = instance[features]

		# Sort features by absolute importance
		sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

		# Calculate impact of each feature
		feature_impacts = []
		for feature, _importance in sorted_features:
			# Create perturbed instance by setting feature to mean value
			perturbed = instance.copy()
			perturbed[feature] = self.feature_means[feature]

			# Get prediction for perturbed instance
			dmatrix = xgb.DMatrix(perturbed.to_frame().T)
			perturbed_pred = self.model.predict(dmatrix)[0]

			# Calculate impact
			impact = abs(original_pred - perturbed_pred)
			feature_impacts.append((feature, impact))

		# Calculate AUFC
		aufc = 0.0
		prev_impact = 0.0
		for i, (_, impact) in enumerate(feature_impacts):
			if i > 0:
				aufc += (impact + prev_impact) / 2
			prev_impact = impact

		# Calculate correlation between feature importance and impact
		importance_values = np.array([abs(imp) for _, imp in sorted_features])
		impact_values = np.array([imp for _, imp in feature_impacts])

		# Handle zero standard deviation and NaN values
		if np.all(importance_values == 0) or np.all(impact_values == 0):
			correlation = 0.0
		else:
			# Remove any NaN values
			mask = ~(np.isnan(importance_values) | np.isnan(impact_values))
			if np.sum(mask) > 1:  # Need at least 2 points for correlation
				correlation = np.corrcoef(importance_values[mask], impact_values[mask])[0, 1]
			else:
				correlation = 0.0

		return aufc, correlation

	def analyze_fidelity(
		self, instances: pd.DataFrame, explanations: dict, method: str
	) -> dict[str, Any]:
		"""Analyze the fidelity of explanations.

		Args:
			instances: DataFrame containing instances to analyze
			explanations: Dictionary containing explanation results
			method: Explanation method used ('shap' or 'lime')

		Returns:
			Dictionary containing fidelity analysis results
		"""
		results = {"aufc_scores": [], "correlations": []}

		# Get features excluding 'id' and 'Class'
		features = [col for col in instances.columns if col not in ["id", "Class"]]
		instances = instances[features]

		for idx, instance in instances.iterrows():
			# Get original prediction
			dmatrix = xgb.DMatrix(instance.to_frame().T)
			original_pred = self.model.predict(dmatrix)[0]

			# Get feature importance based on explanation method
			if method == "shap":
				feature_importance = dict(
					zip(explanations["feature_names"], explanations["shap_values"][idx])
				)
			else:  # LIME
				feature_importance = dict(explanations["explanations"][idx]["explanation"])

			# Calculate fidelity metrics
			aufc, correlation = self.calculate_fidelity_metrics(
				instance, feature_importance, original_pred
			)

			results["aufc_scores"].append(aufc)
			results["correlations"].append(correlation)

			# Calculate fidelity curve (no plot generation)
			n_features = len(feature_importance)
			fidelity_losses = []
			k_values = range(1, n_features + 1)

			for k in k_values:
				perturbed = self.perturb_features(instance, feature_importance, k)
				dmatrix = xgb.DMatrix(perturbed.to_frame().T)
				perturbed_pred = self.model.predict(dmatrix)[0]
				loss = (original_pred - perturbed_pred) ** 2
				fidelity_losses.append(loss)

		# AUFC Score Distribution Plot
		plt.figure(figsize=(6, 5))
		plt.hist(results["aufc_scores"], bins=20, color='skyblue', edgecolor='black')
		plt.title("AUFC Score Distribution")
		plt.xlabel("AUFC Score")
		plt.ylabel("Frequency")
		plt.tight_layout()
		log_dirs = sorted(Path("logs").glob("*"))
		if not log_dirs:
			raise RuntimeError("No log directory found")
		log_dir = log_dirs[-1]
		plots_dir = log_dir / "plots"
		plots_dir.mkdir(parents=True, exist_ok=True)
		overview_plot_aufc_path = plots_dir / f"fidelity_overview_aufc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
		plt.savefig(overview_plot_aufc_path)
		plt.close()

		# Correlation Coefficient Distribution Plot
		plt.figure(figsize=(6, 5))
		plt.hist(results["correlations"], bins=20, color='salmon', edgecolor='black')
		plt.title("Correlation Coefficient Distribution")
		plt.xlabel("Correlation Coefficient")
		plt.ylabel("Frequency")
		plt.tight_layout()
		overview_plot_corr_path = plots_dir / f"fidelity_overview_corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
		plt.savefig(overview_plot_corr_path)
		plt.close()

		results["overview_plot_aufc"] = str(overview_plot_aufc_path)
		results["overview_plot_corr"] = str(overview_plot_corr_path)

		return results
