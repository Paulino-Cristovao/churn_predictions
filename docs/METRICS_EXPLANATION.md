# Model Evaluation Metrics Explanation

This document explains the key performance metrics used in the Kolecto Churn Prediction project, structured to help you choose the right tool for the job.

## 1. Accuracy
*   **Definition**: The percentage of predictions that are correct (both True Positives and True Negatives).
*   **Positive Aspects**: Very intuitive and easy to explain to non-technical stakeholders. Ideally, we want 100%.
*   **Negative Aspects**: Highly misleading for imbalanced datasets. If 90% of users don't churn, a model that predicts "No Churn" for everyone has 90% accuracy but is useless.
*   **When to Use**: Only when the classes are balanced (e.g., 50% Churn / 50% No Churn) and errors are equally costly.
*   **Why we used it**: As a baseline metric to check if our models are performing better than random chance or the "majority class" baseline.

## 2. Precision
*   **Definition**: The accuracy of positive predictions. Out of all the times the model said "Churn", how many actually churned?
*   **Positive Aspects**: Minimizes **False Positives**. It ensures that when we flag a customer, we are confident they are at risk.
*   **Negative Aspects**: Can miss many actual churners (False Negatives) if the model is too conservative.
*   **When to Use**: When the cost of a False Positive is high (e.g., sending an expensive gift or alerting a busy account manager).
*   **Why we used it**: To verify that our "High Risk" lists are high quality and won't waste the Customer Success team's time.

## 3. Recall (Sensitivity)
*   **Definition**: The ability to find all positive instances. Out of all actual churners, how many did the model find?
*   **Positive Aspects**: Minimizes **False Negatives**. Ensures we don't miss at-risk customers.
*   **Negative Aspects**: Predicting "Churn" for everyone gives 100% recall but implies many False Positives (spamming safe customers).
*   **When to Use**: When the cost of missing a positive case is high (e.g., losing a high-value customer).
*   **Why we used it**: Critical for Kolecto because acquiring a new customer is harder than saving an existing one; we want to catch as many potential churners as possible.

## 4. F1 Score
*   **Definition**: The harmonic mean of Precision and Recall. It provides a single score that balances the two.
*   **Positive Aspects**: Good for imbalanced datasets as it requires both good precision and good recall to be high.
*   **Negative Aspects**: Harder to interpret than Accuracy. Assumes Precision and Recall are equally important (which isn't always true business-wise).
*   **When to Use**: When you need a single metric to compare models on imbalanced data and don't have a specific preference for Precision or Recall.
*   **Why we used it**: To quickly rank models during the initial screening phase.

## 5. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
*   **Definition**: Measures the model's ability to distinguish between classes across *all* possible thresholds. 0.5 is random guessing, 1.0 is perfect discrimination.
*   **Positive Aspects**: **Threshold Independent**. It tells you how good the model is fundamentally, regardless of where you set the decision cutoff.
*   **Negative Aspects**: Can be slightly optimistic on highly imbalanced data (though less so than Accuracy).
*   **When to Use**: The standard metric for comparing the general predictive power of different classification models.
*   **Why we used it**: **This was our Primary Metric.** It allowed us to robustly compare LightGBM vs. Deep Learning models without worrying about threshold tuning yet.

## 6. PR-AUC (Precision-Recall - Area Under Curve)
*   **Definition**: Similar to ROC-AUC but focuses on the trade-off between Precision and Recall.
*   **Positive Aspects**: **The Gold Standard for Imbalanced Data.** It ignores True Negatives and focuses purely on how well the model finds the minority class of interest (Churners).
*   **Negative Aspects**: Harder to explain to non-technical audiences.
*   **When to Use**: When the "Positive" class is rare (e.g., Fraud, Churn) and is the only thing you care about.
*   **Why we used it**: To confirm that our best ROC-AUC model (LightGBM) was genuinely good at finding Churners and not just good at classifying the easy non-churners.

## 7. Brier Score (Calibration)
*   **Definition**: The mean squared difference between the predicted probability and the actual outcome.
*   **Positive Aspects**: Measures **Calibration/Reliability**. A low Brier score means the probabilities are trustworthy (e.g., "70% risk" really means 7/10 times they churn).
*   **Negative Aspects**: Not a classification metric (doesn't give a Yes/No answer).
*   **When to Use**: When the inaccurate probability estimate is costly (e.g., algo-trading or automated bidding).
*   **Why we used it**: To ensure that the risk scores shown to the CS team are realistic probabilities they can trust, not just arbitrary numbers.
