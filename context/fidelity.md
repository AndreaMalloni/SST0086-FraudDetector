# Fidelity Analysis for AI Explainability

This document details the process for conducting a **Fidelity Analysis** of explanation methods (specifically SHAP and LIME) in the context of our banking fraud detection project. This analysis addresses the "Quantitative Evaluation" gap identified in our research.

## 1. What is Fidelity?

Fidelity (also known as Faithfulness) measures how accurately an explanation reflects the behavior of the original "black-box" model[cite: 1]. In our case, it assesses whether the SHAP (or LIME) explanations truly represent how our XGBoost fraud detection model makes its predictions. If an explanation method has high fidelity, it means that features deemed important by the explanation are indeed the ones driving the original model's output.

## 2. Why is Fidelity Important for Anonymized Features?

Even with anonymized features (e.g., `V1`, `V2`), a high fidelity score is crucial. It confirms that despite not knowing the semantic meaning of the features, the explanation correctly identifies the technical levers of the black-box model. This validates the explanation method itself, ensuring we are interpreting the actual model behavior, not a misleading representation.

## 3. Perturbation-Based Approach for Fidelity

We will use a perturbation-based approach to assess fidelity. This involves systematically altering the input features of a given instance and observing how the original model's prediction changes. We then compare these changes to the feature importances provided by the explanation method.

### 3.1. Instances for Analysis

We will select a representative subset of instances for fidelity analysis. This could include:
* A random sample of `N` transactions from the test set.
* Specific "key examples" where the model made correct or incorrect predictions, as highlighted in our main paper.

### 3.2. Perturbation Strategy

For each selected instance, we will perform iterative perturbations based on the feature importance ranks derived from the SHAP (or LIME) explanation for that instance.

**Steps for each instance:**

1.  **Generate Explanation:** Obtain the SHAP values (or LIME weights) for the instance. This provides an importance ranking for each feature (e.g., `V17` is most important, `V9` is second, etc.).
2.  **Sort Features:** Sort all features in descending order based on the absolute value of their SHAP value (or LIME weight).
3.  **Iterative Perturbation:** For each `k` from 1 to the total number of features:
    * **Create a Perturbed Copy:** Make a deep copy of the original instance's feature vector.
    * **Apply Perturbation:** For the top `k` features (according to the sorted importance list), replace their values in the perturbed copy.
        * **Recommended Method (for anonymized features): Mean/Median Replacement.** Replace the feature's value with its corresponding mean or median calculated from the *entire training dataset*. This approach ensures that the perturbed values are within the "normal" statistical range observed by the model during training, even without knowing the feature's original meaning.
        * *Alternative methods (consider if suitable):*
            * Replacing with `0.0` (assuming numerical features): Simple, but might place the feature far outside its typical distribution.
            * Adding Gaussian Noise: Adds minor fluctuations, but defining appropriate noise parameters for anonymized features can be challenging.
            * Random Sampling from Distribution: Replaces with a random value from the feature's training distribution.
4.  **Re-run Prediction:** Pass the `k`-th perturbed instance copy to the original trained XGBoost model to get its new prediction probability (or class output).
5.  **Calculate Fidelity Loss:** Compare the new prediction with the original prediction (from the unperturbed instance). A common metric for loss is the squared difference between the original prediction probability ($P_{original}$) and the perturbed prediction probability ($P_{perturbed}$):
    $Loss_k = (P_{original} - P_{perturbed,k})^2$

### 3.3. Fidelity Metrics

After generating the fidelity loss for each `k` (number of perturbed features), we will use the following metrics:

1.  **Area Under the Fidelity Curve (AUFC):**
    * **Calculation:** Plot the `Fidelity Loss` on the y-axis against the `Number of Top Features Perturbed (k)` on the x-axis. The AUFC is then calculated as the area under this curve using numerical integration (e.g., trapezoidal rule).
    * **Interpretation:** A smaller AUFC indicates higher fidelity. This means that perturbing only a few of the top-ranked features (as identified by SHAP) leads to a rapid and significant change in the original model's prediction, implying that SHAP accurately captured the true drivers of the prediction.

2.  **Correlation:**
    * **Calculation:** For each feature, calculate the absolute SHAP value and the absolute change in the model's prediction when *only that specific feature* is perturbed (e.g., replaced with its mean). Then, compute the Pearson correlation coefficient between these two sets of values across all features for a given instance (or aggregated over multiple instances).
    * **Interpretation:** A high positive correlation (closer to +1) suggests that features with larger SHAP values consistently cause larger changes in the original model's output when perturbed, indicating strong fidelity.

## 4. Implementation Details

The implementation will involve:

* **Data Preparation:** Loading the anonymized dataset and performing initial feature engineering/scaling if necessary. Pre-calculating means/medians of features from the training set.
* **Model Training:** Training the XGBoost classifier on the prepared data.
* **Explanation Generation:** Using `shap.TreeExplainer` (for SHAP) or `lime.lime_tabular.LimeTabularExplainer` (for LIME) to generate explanations for selected instances.
* **Perturbation Logic:** Functions to create perturbed copies of instances based on the sorted feature importance and chosen perturbation strategy (mean/median replacement).
* **Prediction with Original Model:** Ensuring perturbed instances are correctly formatted (e.g., as a NumPy array or `xgb.DMatrix`) and passed to the `original_model.predict()` method.
* **Metric Calculation:** Implementing the calculation of AUFC and correlation as described above.
* **Visualization:** Generating plots of the fidelity curve to visually represent the findings.
