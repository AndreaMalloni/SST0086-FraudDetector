Here's the markdown README for the Stability (Robustness) analysis, formatted for Cursor AI context:

# Stability (Robustness) Analysis for AI Explainability

This document details the process for conducting a **Stability (Robustness) Analysis** of explanation methods (specifically SHAP and LIME) in the context of our banking fraud detection project. This analysis addresses the "Quantitative Evaluation" gap identified in our research[cite: 54].

## 1. What is Stability (Robustness)?

Stability, or robustness, evaluates how consistent and reliable an explanation is to small changes in the input data. An explanation is considered stable if minor alterations to the input instance lead to proportionally minor changes in the generated explanation[cite: 59]. This is crucial because an unstable explanation, which fluctuates wildly with slight input variations, would be unreliable for real-world decision-making and auditing.

## 2. Why is Stability Important for Anonymized Features?

Even with anonymized features (e.g., `V1`, `V2`), assessing stability is vital. It ensures that the model's reliance on these features, as revealed by the explanation, is consistent. If the top influencing `V` features change dramatically with negligible input noise, it signals an unstable explanation that could lead to inconsistent interpretations or debugging efforts.

## 3. Small Perturbation Approach for Stability

We will use a "small perturbation" approach to assess stability. This involves:
1.  Generating an explanation for an original input instance.
2.  Creating a slightly perturbed version of that same instance.
3.  Generating an explanation for the perturbed instance.
4.  Comparing the two explanations using specific metrics.

### 3.1. Instances for Analysis

We will select a representative subset of instances from the test set for this analysis. This could include:
* A random sample of `N` transactions from the test set.
* Specific "key examples" where the model made correct or incorrect predictions, to observe how their explanations behave under minor variations[cite: 24].

### 3.2. Defining "Small Perturbations"

For our numerical, anonymized features, "small changes" will be implemented by adding a small amount of random Gaussian noise to each feature's value in the input instance.

* **Gaussian Noise Parameters:**
    * **Mean ($\mu$):** `0` (ensures the perturbation is centered around the original value).
    * **Standard Deviation ($\sigma$):** This is critical. We will calculate the standard deviation (`std`) for each feature across our *training dataset*. The noise added to a feature will then be drawn from `N(0, noise_multiplier * feature_std)`, where `noise_multiplier` is a small factor (e.g., 0.01 or 0.05). This ensures the perturbation is proportional to the inherent variability of each feature.

**Example:** If an original instance has `V1` with value `X`, the perturbed instance will have `V1_perturbed = X + noise`, where `noise` is drawn from `N(0, noise_multiplier * std_dev_of_V1_from_training_data)`.

### 3.3. Practical Steps for Stability Evaluation

For each instance selected for analysis:

1.  **Generate Original Explanation:**
    * Take the `original_instance` (e.g., a row from your test set).
    * Use your `shap.TreeExplainer` (or LIME explainer) to generate the `shap_values_original` for this instance.

2.  **Generate Perturbed Instance and its Explanation:**
    * Create a `perturbed_instance` by applying the defined "small perturbation" (adding Gaussian noise) to every feature of the `original_instance`.
    * Generate `shap_values_perturbed` for this `perturbed_instance` using the *same explainer*.

3.  **Compare Explanations:**
    * You now have two vectors of SHAP values (`shap_values_original` and `shap_values_perturbed`) for almost identical inputs. These will be compared using the following metrics.

### 3.4. Metrics for Comparison

#### 3.4.1. Jaccard Index (for comparing sets of important features)

The Jaccard Index measures the overlap between the set of top-K important features identified by the original explanation and the set of top-K important features identified by the perturbed explanation[cite: 59].

* **Process:**
    1.  For `shap_values_original`, identify the `top_K_features_original`. This set can be defined by selecting features with the highest absolute SHAP values (e.g., top 10 features) or by setting a threshold (e.g., features whose absolute SHAP value is above a certain percentage of the maximum).
    2.  Similarly, for `shap_values_perturbed`, identify the `top_K_features_perturbed` using the same criterion.
    3.  Calculate the Jaccard Index using the formula:
        $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$
        Where `A` is `top_K_features_original` and `B` is `top_K_features_perturbed`.
* **Interpretation:** A Jaccard Index closer to `1` signifies high stability in the *selection* of the most important features. A low score indicates that the set of top features changes significantly with minor input variations, pointing to instability. We will evaluate this for several `K` values (e.g., top 5, top 10, top 20) to provide a comprehensive view.

#### 3.4.2. Cosine Similarity (for comparing vectors of SHAP values)

Cosine Similarity quantifies the similarity in *direction* and *magnitude* between the two vectors of feature contributions (SHAP values)[cite: 59].

* **Process:**
    1.  Ensure `shap_values_original` and `shap_values_perturbed` are treated as vectors with the same feature order.
    2.  Calculate the cosine similarity:
        $CosineSimilarity(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||}$
        Where $\mathbf{A}$ is `shap_values_original` and $\mathbf{B}$ is `shap_values_perturbed`.
* **Interpretation:** A cosine similarity score close to `1` indicates high stability, meaning the feature contributions (both their relative strengths and their signs) are very similar between the original and perturbed explanations. A value close to `0` or negative indicates high instability.

## 4. Implementation Details

The implementation will build upon our existing data processing and model training pipeline:

* **Data Preparation:** Ensure feature standard deviations from the training set are readily available for the perturbation step.
* **Model Training:** The `original_model` (XGBoost) will be pre-trained.
* **Explanation Generation:** Utilize the appropriate `shap.TreeExplainer` or LIME explainer to produce explanations for both original and perturbed instances.
* **Perturbation Logic:** Implement a function (`apply_small_perturbation`) that takes an instance and adds Gaussian noise to its features based on calculated standard deviations and a `noise_multiplier`.
* **Metric Calculation:** Implement functions to compute the Jaccard Index and Cosine Similarity for pairs of explanation vectors.
* **Aggregation and Reporting:** Perform the analysis for a sample of instances from the test set, aggregate the Jaccard and Cosine similarity scores (e.g., by averaging), and report the overall stability measures. Visualization (e.g., histograms of scores) may be used to illustrate the distribution of stability.
