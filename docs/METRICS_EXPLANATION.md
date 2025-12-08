#  Model Evaluation Metrics Explanation

This document explains the key performance metrics used in the Kolecto Churn Prediction project. Understanding these metrics is crucial for interpreting model performance, especially in the context of **imbalanced classification** (where one class, typically "Churn" or "Conversion", is rarer than the other).

## 1. Accuracy
**Definition**: The ratio of correctly predicted observations to the total observations.
$$ \text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}} $$

*   **Advantages**:
    *   Intuitive and easy to explain to stakeholders.
    *   Useful when classes are balanced (e.g., 50% Churn / 50% No Churn).
*   **Drawbacks**:
    *   **Highly misleading for imbalanced data.** If 95% of customers don't churn, a "dumb" model that predicts "No Churn" for everyone will have 95% accuracy but is useless.
    *   Does not distinguish between the types of errors (False Positives vs. False Negatives).

---

## 2. Precision
**Definition**: The ratio of correctly predicted positive observations to the total predicted positives.
$$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$

*   ** Advantages**:
    *   Critical when the cost of a **False Positive** is high.
    *   *Example*: If you are sending a very expensive gift to potential converters, you want high precision to ensure you aren't wasting money on people who won't actually convert.
*   ** Drawbacks**:
    *   Ignores False Negatives. A model could find only 1 valid conversion out of 100 actual conversions; it would have 100% precision but miss 99% of opportunities.

---

## 3. Recall (Sensitivity / True Positive Rate)
**Definition**: The ratio of correctly predicted positive observations to the all observations in actual class.
$$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$

*   ** Advantages**:
    *   Critical when the cost of a **False Negative** is high.
    *   *Example*: In churn prediction, you often want high recall to identify *all* at-risk customers, even if it means flagging some safe ones (False Positives) by mistake. It's better to verify a safe customer than to lose a valuable one.
*   ** Drawbacks**:
    *   Improving recall often reduces precision. A model that predicts "Churn" for everyone has 100% recall but terrible precision.

---

## 4. F1 Score
**Definition**: The weighted average of Precision and Recall.
$$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

*   ** Advantages**:
    *   Good balance when you have an uneven class distribution (large number of Actual Negatives).
    *   Useful when False Positives and False Negatives are equally important.
*   ** Drawbacks**:
    *   Less interpretable than accuracy.
    *   Assumes equal importance of precision and recall (which might not be true for your business case).

---

## 5. ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
**Definition**: Measures the ability of a classifier to distinguish between classes. It plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings.

*   ** Advantages**:
    *   **Threshold Independent**: Evaluates the model's ranking ability across *all* possible decision thresholds, not just 0.5.
    *   Excellent for comparing different models generally. An AUC of 0.5 represents random guessing; 1.0 is perfect.
*   ** Drawbacks**:
    *   Can be overly optimistic when classes are highly imbalanced. The False Positive Rate can stay low simply because the number of Negatives is huge.

---

## 6. PR-AUC (Area Under the Precision-Recall Curve)
**Definition**: Similar to ROC-AUC but plots Precision against Recall.

*   ** Advantages**:
    *   **The Gold Standard for Imbalanced Datasets.**
    *   Focuses exclusively on the minority class (Positive). It does not care about True Negatives (which dominate in imbalance).
    *   If PR-AUC is high, the model is truly good at finding the "needle in the haystack."
*   ** Drawbacks**:
    *   Harder to interpret for non-technical stakeholders compared to ROC-AUC.

---

## 7. Brier Score
**Definition**: The mean squared difference between the predicted probability and the actual outcome (0 or 1).
$$ \text{Brier Score} = \frac{1}{N} \sum_{t=1}^{N} (f_t - o_t)^2 $$

*   ** Advantages**:
    *   Measures **Calibration**. It checks not just if the classification is right, but if the *probability* is accurate.
    *   *Example*: If a model predicts 70% chance of conversion, then 70% of those customers should actually convert.
    *   Lower is better (0 is perfect).
*   ** Drawbacks**:
    *   Not a classification metric (doesn't give a Yes/No answer).
    *   Can be low even for a model with poor discrimination if the base rate is very low (e.g., predicting 1% for everyone when the true rate is 1% gives a great Brier score but identifies no one).

---

## Summary Recommendation
For the **Kolecto** project:
1.  **Primary Metric: ROC-AUC / PR-AUC** (to select the best model architecture).
2.  **Operational Metric: Recall** (tune the threshold to catch at least 80-90% of potential conversions/churners).
3.  **Calibration: Brier Score** (ensure the probabilities shown in the App are realistic).
