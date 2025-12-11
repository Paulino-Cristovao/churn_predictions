# Next Steps and Roadmap

This document outlines the strategic roadmap for the **SaaS Churn Prediction System**. We have successfully completed **Phase 1 (MVP)** and **Phase 2 (Expanded Categorical Features)**.

---

## Completed Milestones
*   **Feature Expansion**: Integrated distinct company firmographics (Industry, Legal Structure, Revenue, etc.) via `ColumnTransformer`.
*   **Model Variety**: Evaluated Linear, Tree-based (XGB/LGBM), and Deep Learning (GRU/Transformer) architectures.
*   **Optimization**: Tuned LightGBM to achieve top-tier performance (AUC 0.82).
*   **Interactive App**: Built a fully dynamic `app.py` for real-time inference.

---

## Phase 3: Advanced Data Engineering

### 1. True Sequential Input for Deep Learning
*   **Current State**: The Deep Learning models currently use *aggregated* usage stats (sum/mean over 15 days) broadcasted across time steps, or static sequences.
*   **Improvement**: Modify the data pipeline to feed **raw daily vectors** (shape `[Batch, 15, 19]`) into the GRU/Transformer.
*   **Hypothesis**: Capturing the *order* of actions (e.g., "Invoices created *after* bank connection" vs "before") will significantly boost DL performance.

### 2. Temporal Trends
*   **Feature Engineering**: Explicitly calculate trend features for tree models:
    *   `messages_sent_trend`: Slope of messages over the 15-day window.
    *   `acceleration`: Is usage accelerating or decelerating in the last 3 days?

---

## 🛠 Phase 4: Production Engineering (MLOps)

### 1. Model Registry & Versioning
*   Move from local `.pkl` files to **MLflow** or **Weights & Biases**.
*   Track experiments, metrics, and distinct model versions (e.g., `v2_lightgbm_157feats`).

### 2. API Deployment
*   Wrap the prediction logic in a **FastAPI** microservice.
*   **Dockerize** the application for consistent deployment on Kubernetes or Cloud Run.

### 3. Drift Monitoring
*   Implement monitoring to detect if user behavior changes over time (Data Drift).
*   *Example*: If `nb_mobile_connections` drops suddenly across all users (app bug?), the model should alert.

---

##  Phase 5: Explainability & Insights

### 1. SHAP (SHapley Additive exPlanations)
*   Integrate SHAP to explain *individual* predictions in the App.
*   *User Value*: Instead of just "85% Churn Risk", tell the Customer Success Manager: *"Risk is high because 'Last Login' was 14 days ago AND 'NPS Score' is missing."*

### 2. Segment Analysis
*   Analyze error rates by segment. Does the model underperform on "Large Enterprise" vs "Freelancers"?
