from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    recall_score,
    precision_score,
    accuracy_score
)
import matplotlib.pyplot as plt
from fraud_detector.core.logger import LoggingManager

class DiscriminativePowerAnalyzer:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize the discriminative power analyzer.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.logger = LoggingManager().get_logger("DiscriminativePowerAnalysis")
        self.test_size = test_size
        self.random_state = random_state
        
    def prepare_explanation_features(self,
                                   explanations: Dict[str, Any],
                                   method: str) -> pd.DataFrame:
        """Convert explanations into feature matrix.
        
        Args:
            explanations: Dictionary containing explanation results
            method: Explanation method used ('shap' or 'lime')
            
        Returns:
            DataFrame containing explanation features
        """
        if method == 'shap':
            # For SHAP, we already have the values in the correct format
            return pd.DataFrame(
                explanations['shap_values'],
                columns=explanations['feature_names']
            )
        else:  # LIME
            # For LIME, we need to convert the list of (feature, weight) pairs
            # into a feature matrix
            feature_matrix = []
            for exp in explanations['explanations']:
                # Create a dictionary of feature weights
                feature_weights = dict(exp['explanation'])
                # Ensure all features are present (some might be missing in individual explanations)
                row = {feature: 0.0 for feature in explanations['feature_names']}
                row.update(feature_weights)
                feature_matrix.append(row)
            return pd.DataFrame(feature_matrix)
    
    def train_and_evaluate(self,
                          X_explanations: pd.DataFrame,
                          y_true: pd.Series,
                          model_type: str = 'logistic') -> Dict[str, float]:
        """Train and evaluate a simple model on explanation features.
        
        Args:
            X_explanations: DataFrame containing explanation features
            y_true: Series containing true labels
            model_type: Type of model to use ('logistic' or 'tree')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_explanations,
            y_true,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_true
        )
        
        # Initialize and train the model
        if model_type == 'logistic':
            model = LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state
            )
        else:  # tree
            model = DecisionTreeClassifier(
                max_depth=3,  # Keep it simple and interpretable
                class_weight='balanced',
                random_state=self.random_state
            )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'auc_pr': average_precision_score(y_test, y_pred_proba),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
        }
        
        # If using logistic regression, add feature importance
        if model_type == 'logistic':
            metrics['feature_importance'] = dict(zip(
                X_explanations.columns,
                np.abs(model.coef_[0])
            ))
        
        return metrics
    
    def analyze_discriminative_power(self,
                                   explanations: Dict[str, Any],
                                   y_true: pd.Series,
                                   method: str) -> Dict[str, Any]:
        """Analyze discriminative power of explanations.
        
        Args:
            explanations: Dictionary containing explanation results
            y_true: Series containing true labels
            method: Explanation method used ('shap' or 'lime')
            
        Returns:
            Dictionary containing analysis results
        """
        # Prepare explanation features
        X_explanations = self.prepare_explanation_features(explanations, method)
        
        # Train and evaluate both models
        logistic_metrics = self.train_and_evaluate(
            X_explanations,
            y_true,
            model_type='logistic'
        )
        tree_metrics = self.train_and_evaluate(
            X_explanations,
            y_true,
            model_type='tree'
        )
        
        # Create visualization
        self._plot_metrics_comparison(logistic_metrics, tree_metrics, method)
        
        return {
            'logistic_regression': {
                'metrics': {k: v for k, v in logistic_metrics.items() 
                          if k != 'feature_importance'},
                'feature_importance': logistic_metrics.get('feature_importance', {})
            },
            'decision_tree': {
                'metrics': tree_metrics
            }
        }
    
    def _plot_metrics_comparison(self,
                               logistic_metrics: Dict[str, float],
                               tree_metrics: Dict[str, float],
                               method: str):
        """Create and save metrics comparison plot.
        
        Args:
            logistic_metrics: Metrics from logistic regression
            tree_metrics: Metrics from decision tree
            method: Explanation method used
        """
        # Prepare data for plotting
        metrics = ['f1_score', 'auc_pr', 'recall', 'precision', 'accuracy']
        logistic_values = [logistic_metrics[m] for m in metrics]
        tree_values = [tree_metrics[m] for m in metrics]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, logistic_values, width, label='Logistic Regression')
        plt.bar(x + width/2, tree_values, width, label='Decision Tree')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'Discriminative Power Metrics - {method.upper()}')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        log_dirs = sorted(Path("logs").glob("[*]"))
        if not log_dirs:
            raise RuntimeError("No log directory found")
        log_dir = log_dirs[-1]
        plots_dir = log_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = plots_dir / f"discriminative_power_{method}.png"
        plt.savefig(plot_path)
        plt.close()