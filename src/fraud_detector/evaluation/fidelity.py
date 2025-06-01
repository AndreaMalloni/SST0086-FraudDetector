from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from fraud_detector.core.logger import LoggingManager

class FidelityAnalyzer:
    def __init__(self, model_path: Path, training_data: pd.DataFrame):
        """Initialize the fidelity analyzer.
        
        Args:
            model_path: Path to the trained XGBoost model
            training_data: Training dataset used to calculate feature statistics
        """
        self.logger = LoggingManager().get_logger("FidelityAnalysis")
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        
        # Calculate feature statistics from training data
        self.feature_means = training_data.drop(columns=['Class', 'id']).mean()
        self.feature_medians = training_data.drop(columns=['Class', 'id']).median()
        
    def perturb_features(self, 
                        instance: pd.Series, 
                        feature_importance: Dict[str, float], 
                        k: int) -> pd.Series:
        """Create a perturbed copy of the instance by replacing top k features.
        
        Args:
            instance: Original instance to perturb
            feature_importance: Dictionary mapping feature names to their importance scores
            k: Number of top features to perturb
            
        Returns:
            Perturbed instance
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Create a copy of the instance
        perturbed = instance.copy()
        
        # Replace top k features with their median values
        for feature, _ in sorted_features[:k]:
            perturbed[feature] = self.feature_medians[feature]
            
        return perturbed
    
    def calculate_fidelity_metrics(self,
                                 instance: pd.Series,
                                 feature_importance: Dict[str, float],
                                 original_pred: float) -> Tuple[float, float]:
        """Calculate fidelity metrics for a single instance.
        
        Args:
            instance: Instance to analyze
            feature_importance: Dictionary mapping feature names to their importance scores
            original_pred: Original prediction probability
            
        Returns:
            Tuple of (AUFC, correlation)
        """
        n_features = len(feature_importance)
        fidelity_losses = []
        feature_impacts = []
        
        # Calculate impact of perturbing each feature individually
        for feature in instance.index:
            if feature in ['Class', 'id']:
                continue
                
            perturbed = instance.copy()
            perturbed[feature] = self.feature_medians[feature]
            
            # Get prediction for perturbed instance
            dmatrix = xgb.DMatrix(perturbed.to_frame().T)
            perturbed_pred = self.model.predict(dmatrix)[0]
            
            # Calculate impact
            impact = abs(original_pred - perturbed_pred)
            feature_impacts.append(impact)
        
        # Calculate correlation between feature importance and impact
        importance_values = [abs(v) for v in feature_importance.values()]
        correlation = np.corrcoef(importance_values, feature_impacts)[0, 1]
        
        # Calculate AUFC
        for k in range(1, n_features + 1):
            perturbed = self.perturb_features(instance, feature_importance, k)
            dmatrix = xgb.DMatrix(perturbed.to_frame().T)
            perturbed_pred = self.model.predict(dmatrix)[0]
            
            loss = (original_pred - perturbed_pred) ** 2
            fidelity_losses.append(loss)
        
        # Calculate AUFC using trapezoidal rule
        aufc = np.trapz(fidelity_losses, dx=1)
        
        return aufc, correlation
    
    def analyze_fidelity(self,
                        instances: pd.DataFrame,
                        explanations: Dict[str, Any],
                        method: str) -> Dict[str, Any]:
        """Analyze fidelity for multiple instances.
        
        Args:
            instances: DataFrame containing instances to analyze
            explanations: Dictionary containing explanation results
            method: Explanation method used ('shap' or 'lime')
            
        Returns:
            Dictionary containing fidelity analysis results
        """
        results = {
            'aufc_scores': [],
            'correlations': [],
            'plots': []
        }
        
        for idx, instance in instances.iterrows():
            # Get original prediction
            dmatrix = xgb.DMatrix(instance.drop(['Class', 'id']).to_frame().T)
            original_pred = self.model.predict(dmatrix)[0]
            
            # Get feature importance based on explanation method
            if method == 'shap':
                feature_importance = dict(zip(
                    explanations['feature_names'],
                    explanations['shap_values'][idx]
                ))
            else:  # LIME
                feature_importance = dict(explanations['explanations'][idx]['explanation'])
            
            # Calculate fidelity metrics
            aufc, correlation = self.calculate_fidelity_metrics(
                instance,
                feature_importance,
                original_pred
            )
            
            results['aufc_scores'].append(aufc)
            results['correlations'].append(correlation)
            
            # Create fidelity curve plot
            self._plot_fidelity_curve(instance, feature_importance, idx)
            
        return results
    
    def _plot_fidelity_curve(self,
                           instance: pd.Series,
                           feature_importance: Dict[str, float],
                           idx: int):
        """Create and save fidelity curve plot.
        
        Args:
            instance: Instance being analyzed
            feature_importance: Dictionary mapping feature names to their importance scores
            idx: Instance index
        """
        n_features = len(feature_importance)
        fidelity_losses = []
        k_values = range(1, n_features + 1)
        
        # Calculate original prediction
        dmatrix = xgb.DMatrix(instance.drop(['Class', 'id']).to_frame().T)
        original_pred = self.model.predict(dmatrix)[0]
        
        # Calculate losses for each k
        for k in k_values:
            perturbed = self.perturb_features(instance, feature_importance, k)
            dmatrix = xgb.DMatrix(perturbed.to_frame().T)
            perturbed_pred = self.model.predict(dmatrix)[0]
            loss = (original_pred - perturbed_pred) ** 2
            fidelity_losses.append(loss)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, fidelity_losses, 'b-', label='Fidelity Loss')
        plt.fill_between(k_values, fidelity_losses, alpha=0.2)
        plt.xlabel('Number of Top Features Perturbed (k)')
        plt.ylabel('Fidelity Loss')
        plt.title(f'Fidelity Curve for Instance {idx}')
        plt.legend()
        
        # Save plot
        log_dirs = sorted(Path("logs").glob("[*]"))
        if not log_dirs:
            raise RuntimeError("No log directory found")
        log_dir = log_dirs[-1]
        plots_dir = log_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = plots_dir / f"fidelity_curve_{idx}.png"
        plt.savefig(plot_path)
        plt.close()