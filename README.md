# Kolecto Churn Prediction System "Pro"

![Status](https://img.shields.io/badge/Status-Production_Ready-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![Gradio](https://img.shields.io/badge/Gradio-App-pink)

> **Predicting B2B Customer Churn with Deep Learning and 157-Dimensional Feature Engineering.**

---

## Executive Summary
This project provides a robust, end-to-end machine learning solution to predict customer churn for **Kolecto**. By analyzing usage patterns (15 days history) and rich company firmographics (`subscriptions.csv`), the system identifies at-risk clients with high precision.

**Key Highlights:**
*   **State-of-the-Art Models**: Compare **LightGBM**, **XGBoost**, **Logistic Regression**, **GRU (RNN)**, and **Transformer**.
*   **Deep Feature Engineering**: Automates the processing of **157 features**, using standard scaling for metrics and One-Hot/Ordinal encoding for categorical data like *NAF Codes, Revenue Ranges, and Legal Structures*.
*   **Interactive Application**: A user-friendly Gradio web interface (`app.py`) for real-time scoring.
*   **Top Performance**: **LightGBM** achieves **AUC 0.82**, significantly outperforming baselines.

---

## Model Performance Leaderboard

| Model | Accuracy | ROC-AUC | PR-AUC | Calibration (Brier) |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **75.9%** | **0.822** | **0.869** | **0.188** |
| **Transformer** | 72.3% | 0.736 | 0.733 | 0.208 |
| **LSTM/GRU** | 68.7% | 0.722 | 0.774 | 0.213 |
| **Logistic Regression** | 69.9% | 0.689 | 0.760 | 0.226 |
| **XGBoost** | 63.9% | 0.672 | 0.777 | 0.246 |

> *See [docs/METRICS_EXPLANATION.md](docs/METRICS_EXPLANATION.md) for details on these metrics.*

---

## Architecture

The system is modular, ensuring reproducibility and easy maintenance.

```mermaid
graph LR
    A[Raw Data] --> B(01_Data_Processing)
    B --> C{Feature Engineering}
    C -->|OneHot + Ordinal| D[X_train (157 feats)]
    D --> E[Model Requests]
    E --> F[LightGBM]
    E --> G[GRU / Transformer]
    E --> H[XGBoost]
    F & G & H --> I[Verification & Selection]
    I --> J[Gradio App]
```

---

## Installation & Usage

### 1. Setup Environment
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive App
Launch the web interface to test predictions on new data.
```bash
python app.py
```
*The app will open in your browser at `http://0.0.0.0:7860`.*

### 3. Reproduce Training
To re-run the entire pipeline from scratch:
```bash
# Process Data
jupyter nbconvert --to notebook --execute notebooks/01_data_processing.ipynb --inplace

# Train Models (e.g., LightGBM)
jupyter nbconvert --to notebook --execute notebooks/04_lightgbm.ipynb --inplace
```

---

## Documentation Index

All detailed documentation is located in the `docs/` directory:

*   **[Model Documentation](docs/MODEL_DOCUMENTATION.md)**: Deep dive into architectures, hyperparameters, and training strategies.
*   **[Metrics Explanation](docs/METRICS_EXPLANATION.md)**: Guide to understanding AUC, Precision, Recall, etc.
*   **[Next Steps](docs/NEXT_STEPS.md)**: Roadmap for future improvements (Deployment, MLOps).
*   **[Testing Results](docs/TESTING_RESULTS.md)**: Logs of validation runs.

---

## Repository Structure

```
├── app.py                  # Main Gradio Application
├── notebooks/              # Jupyter Notebooks (Data, Training, Evaluation)
│   ├── 01_data_processing.ipynb
│   ├── ...
│   └── 07_model_comparison.ipynb
├── models/                 # PyTorch Model Definitions (GRU/Transformer)
├── config/                 # Configuration Files
├── docs/                   # Documentation
├── results/                # Saved Artifacts (Models, Figures, Metrics)
│   ├── models/             # PKL and PT files
│   ├── preprocessor.pkl    # Fitted ColumnTransformer
│   └── app_config.json     # App Dropdown Options
└── requirements.txt        # Dependencies
```

---

*Built for Kolecto by "Antigravity"*
