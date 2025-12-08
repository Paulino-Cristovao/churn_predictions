# Next Steps and Recommendations for Model Improvement

This document outlines specific, actionable steps to improve the performance and deployment readiness of the churn prediction models.

## 1. General Data & Feature Engineering Improvements

- **Expand Feature Set**: Currently, we only use aggregated usage data (sum, mean, max, std). Consider adding:
    - **Temporal Trends**: Slope of usage over the 15 days (is usage increasing or decreasing?).
    - **Ratios**: `nb_transfers_sent` / `nb_connections`.
    - **Categorical Data**: Industry, company size, country (if available in raw subscriptions).
- **Handle Class Imbalance**: The dataset is imbalanced.
    - **SMOTE/ADASYN**: Experiment with oversampling the minority class (churners) during training.
    - **Class Weights**: Tune `scale_pos_weight` (XGBoost/LightGBM) or `pos_weight` (BCEWithLogitsLoss) more aggressively.

## 2. Model-Specific Recommendations

### XGBoost (Current Best Model)
- **Tuning**: Run a more extensive GridSearch or Optuna optimization specifically for `max_depth` (3-10) and `min_child_weight`.
- **Regularization**: Increase `reg_alpha` (L1) and `reg_lambda` (L2) to prevent potential overfitting on the small feature set.
- **Explainability**: Use SHAP values (SHapley Additive exPlanations) instead of standard feature importance for more granular insight into *why* specific users are flagged.

### LightGBM
- **Hyperparameters**: The current Optuna search (50 trials) is a good start. Increase to 100-200 trials.
- **Categorical Features**: If we add static user data, LightGBM handles categoricals natively (faster and often better than OneHot).

### Deep Learning (GRU / Transformer)
- **Sequence Length**: We currently reshape the aggregated 76 features into a pseudo-sequence of 15 days? **Correction**: The current Deep Learning models use the *aggregated* features (vector of 76) treated as a sequence or projected.
    - **True Sequential Data**: For the NEXT iteration, correct the Data Processing pipeline to **preserve the 15-day daily sequence** (shape: `[Batch, 15, 19]`) instead of aggregating it first. GRU and Transformer are designed for this raw sequential input and will likely perform MUCH better than on the aggregated stats.
- **Architecture**:
    - **Bidirectional GRU**: Allow the model to see future context (within the 15 days) relative to early days.
    - **Attention Heads**: Increase Transformer heads if using the full sequence.

### Logistic Regression (Baseline)
- **Polynomial Features**: Interactions between usage metrics (e.g., `invoices_created` * `transfers_sent`) could capture non-linearities that linear models miss.

## 3. Deployment & MLOps

- **Model Registry**: Move from local `.pkl` files to a proper registry (MLflow, AWS SageMaker) to version control models.
- **Monitoring**: Implement drift detection. If the distribution of `nb_transfers_sent` changes (e.g., due to a product update), the model needs retraining.
- **Feedback Loop**: Capture actual outcomes for the predicted users and feed them back into the training set monthly.

## 4. Business Integration

- **Threshold Tuning**: Don't just use `p > 0.5`.
    - If the cost of a retention campaign is low (e.g., email), lower the threshold to 0.3 to catch more potential churners (High Recall).
    - If the cost is high (e.g., human call), raise it to 0.7 to prioritize sure cases (High Precision).
- **Segmentation**: Build separate models for "New" vs "Established" customers if the behaviors differ significantly.
