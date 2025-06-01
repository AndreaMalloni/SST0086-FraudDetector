# Opening the Black Box: AI Explainability in Banking Fraud Detection with SHAP

## 1. Project Overview

**Problem Statement:**
In recent years, machine learning (ML) has become increasingly important in financial fraud detection, particularly with the rise of online payments and digital banking. However, the most accurate ML models, such as ensemble methods or deep learning, often operate as "black boxes," making predictions without clearly explaining their decisions. This lack of transparency hinders trust and adoption in heavily regulated sectors like banking, where regulations such as GDPR require explainable automated decisions, especially when they directly affect customers.

**Research Gap:**
While Explainable Artificial Intelligence (XAI) aims to make ML models more transparent without sacrificing accuracy, there's limited research on the helpfulness of XAI techniques like SHAP when input data is anonymized, a common practice for privacy reasons. Furthermore, current XAI use in fraud detection research often lacks thorough quantitative evaluation.

**Research Questions:**
This paper aims to address these gaps by focusing on the utility of SHAP for explaining machine learning model decisions in credit card fraud detection. We investigate two main questions:
1.  Can SHAP provide explanations that are understandable, even when the features are anonymized?
2.  Could these explanations help in real settings, such as for internal audits or regulatory checks?

**Contributions:**
This study offers three main contributions:
1.  A practical case study using SHAP to explain a fraud detection model, based on real, anonymized data and a widely used ML algorithm.
2.  An evaluation of how understandable and meaningful SHAP explanations are, both in general and for individual predictions.
3.  A comparison between SHAP and LIME, focusing on their performance, ease of use, and clarity of output.

## 2. Methodology

The project utilizes the Kaggle Credit Card Fraud Detection dataset, which contains anonymized transaction data from European cardholders. An XGBoost model, known for its performance with imbalanced datasets, is trained on this data. SHAP is then applied to explain the model's behavior, examining both global patterns (global feature importance) and specific predictions (local explanations). A brief comparison with LIME is also included.

### 2.1. Technical Stack

* **Machine Learning Library:** XGBoost
* **Explainability Libraries:** SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanations)
* **Data Processing:** Apache Spark
* **Configuration Management:** YAML

### 2.2. Training Pipeline Configuration

The [config.yaml](config/config.yaml) file defines the global parameters, Spark settings, training parameters, explanation configuration, and analysis parameters.
## 3. Addressing Research Gaps

### 3.1. Utility of Explanations (RQ1 & RQ2)

To address the utility and understandability of explanations, especially with anonymized data, the project will go beyond purely theoretical discussion through:

* **Practical Demonstrations:**
    * **Local Explanations:** Present SHAP force/waterfall plots for specific transactions (e.g., one correct prediction, one incorrect prediction). Even with anonymized features (e.g., `V1`, `V2`), the theoretical interpretation of feature contributions will be discussed. For example, how a fraud analyst *would* interpret these contributions if the feature meanings were known, and how they would investigate `V` features acting as strong positive or negative contributors.
    * **Global Feature Importance:** Display global SHAP feature importance plots to show generally influential anonymized features across the dataset. Discussion will cover whether these top features align with general fraud characteristics, offering insights into the model's overall behavior on this real-world anonymized dataset.
* **Qualitative Argument for Utility:**
    * **Auditability & Regulatory Compliance:** Argue that SHAP provides a traceable path from inputs to outputs, which inherently contributes to auditability and compliance (e.g., satisfying GDPR requirements for explaining automated decisions), even if the semantic meaning of `V` features is hidden.
    * **Model Debugging & Improvement:** Discuss how SHAP explanations, even with anonymized data, can help data scientists identify unexpected correlations, pinpoint error sources, and guide future feature engineering efforts.
    * **SHAP vs. LIME Comparison:** Qualitatively assess the "clarity of output" between SHAP and LIME when dealing with anonymized features, considering which method presents contributions more intuitively or consistently.
* **Acknowledged Limitations:** Explicitly state that direct user interpretation of `V` features and direct "user perspective" evaluation by banking professionals are beyond the scope due to data anonymization and practical constraints.

### 3.2. Quantitative Evaluation of Explanations (RQ1 & RQ2)

To rigorously evaluate the explanations quantitatively, the following approaches and metrics will be employed:

1.  **Fidelity (Faithfulness):**
    * **Objective:** Measure how well the explanation reflects the behavior of the original black-box model.
    * **Approach:** Perturbation-based. Systematically perturb features based on their importance from the explanation and observe the change in the original model's prediction.
    * **Metrics:**
        * **Area Under the Fidelity Curve (AUFC):** Measures the drop in prediction probability/accuracy as features are removed/perturbed based on importance.
        * **Correlation:** Between SHAP scores and actual changes in model output due to feature perturbation.
    * **Consideration:** Define specific perturbation strategies (e.g., replacing with zeros, mean, or random values from distribution).

2.  **Stability (Robustness):**
    * **Objective:** Evaluate the consistency and robustness of an explanation to small input changes.
    * **Approach:** Small perturbation. Apply minor changes to input instances and compare the resulting explanations.
    * **Metrics:**
        * **Jaccard Index:** To compare the overlap of top-K important features between explanations of original and perturbed instances.
        * **Cosine Similarity:** To compare the vectors of SHAP values/LIME weights for original and perturbed instances, assessing stability in magnitude and direction of contributions.
    * **Consideration:** Clearly define "small changes" (e.g., adding Gaussian noise, slight feature modifications).

3.  **Explanation Complexity / Sparsity:**
    * **Objective:** Measure how concise and focused an explanation is. Simpler explanations are generally more interpretable.
    * **Metrics:**
        * **Number of Relevant Features:** Count features whose absolute SHAP value exceeds a defined threshold (e.g., absolute value or percentage of max SHAP value).
        * **Contribution Percentage:** Calculate the percentage of total absolute SHAP values accounted for by the top N features.
    * **Consideration:** Define a suitable threshold for "relevant features."

4.  **Discriminative Power of Explanations:**
    * **Objective:** Assess whether the explanations themselves capture sufficient information to distinguish between fraud and non-fraud.
    * **Approach:** Use SHAP values (or LIME weights) as features to train a new, simple model (e.g., Logistic Regression or a shallow Decision Tree) to classify transactions.
    * **Metrics:** Evaluate the performance of this simple model using accuracy, F1-score, and AUC (especially AUC-PR for imbalanced data).
    * **Consideration:** Compare the performance of this "explanation-trained" model to a simple model trained directly on the original anonymized features to gauge information preservation. This metric is particularly strong as it assesses intrinsic utility without requiring semantic understanding of anonymized features.

These quantitative evaluation methods, applied to both SHAP and LIME, will provide a robust empirical foundation for the research findings and contribute significantly to addressing the identified gaps.


## References

1. I. P. Ojo and A. Tomy, "Explainable AI for credit card fraud detection: Bridging the gap between accuracy and interpretability," *World Journal of Advanced Research and Reviews*, vol. 25, no. 2, pp. 1246–1256, 2023.

2. N. Faruk, A. Tariq, S. Oladele, and M. Gok, "Explainable AI (XAI) for fraud detection: Building trust and transparency in AI-driven financial security systems," *Preprint*, ResearchGate, 2023. Available: [https://www.researchgate.net/publication/390235753](https://www.researchgate.net/publication/390235753)

3. S. K. Aljunaid, S. J. Almheiri, H. Dawood, and M. A. Khan, "Secure and transparent banking: Explainable AI-driven federated learning model for financial fraud detection," *Journal of Risk and Financial Management*, vol. 18, no. 4, p. 179, 2023.

4. B. van Veen, "A user-centered explainable artificial intelligence approach for financial fraud detection," *Finance Research Letters*, vol. 58, Part A, 104309, 2024.

5. M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you?: Explaining the predictions of any classifier," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 1135–1144, 2016.

6. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 4765–4774, 2017.

7. T. Miller, "Explanation in artificial intelligence: Insights from the social sciences," *Artificial Intelligence*, vol. 267, pp. 1–38, 2019.

8. R. Nelgiriyewithana, "Credit Card Fraud Detection Dataset (2023)," *Kaggle*, 2023. Available: [https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
