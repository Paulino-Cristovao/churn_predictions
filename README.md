# Client Churn Prediction System "Pro"
# SaaS Churn Prediction System "Pro"

![Status](https://img.shields.io/badge/Status-Production_Ready-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![Gradio](https://img.shields.io/badge/Gradio-App-pink)

> [!NOTE]
> **Project Status**: Completed (v1.0)
> **Goal**: Predict **Trial-to-Paid Conversion** for B2B Client (SaaS).

This project verifies and implements a robust, end-to-end machine learning solution to predict customer churn for **the Client**, based on a case study of SaaS trial conversions. By analyzing **~11,600 daily usage records** and **~500 subscriptions** (filtered to **416 exact 15-day trials** for consistency), the system identifies significantly predictive signals of conversion (**~60% conversion rate**) versus cancellation.

**Key Highlights:**
*   **State-of-the-Art Models**: Compare **LightGBM**, **XGBoost**, **Logistic Regression**, **GRU (RNN)**, and **Transformer**.
| Model | ROC-AUC | PR-AUC | Notes |
| :--- | :--- | :--- | :--- |
| **LightGBM + Optuna** | 0.7976 | 0.8414 | **Best Single Model** (Fast & Accurate) |

### 🏆 Pourquoi LightGBM & l'Ensemble gagnent ?
LightGBM gère mieux les données tabulaires denses que les réseaux de neurones sur ce petit volume de données. Il capture efficacement les interactions non-linéaires sans nécessiter des milliers d'exemples comme le Deep Learning.
*   **Deep Feature Engineering**: Automates the processing of **157 features**, aggregating **19 daily usage metrics** (e.g., `nb_transfers_sent`, `nb_mobile_connections`) via sum/mean/max/std, combined with categorical firmographics like *NAF Codes* and *Revenue Ranges*.
*   **Interactive Application**: A user-friendly Gradio web interface (`app.py`) for real-time scoring.
*   **Top Performance**: **LightGBM** achieves **AUC ~0.80**, capturing non-linear interactions better than baselines.
*   **Causal Uplift**: New **Uplift Model** aligns with predictive performance (**75.9% Accuracy**, **0.80 AUC**) while identifying "Persuadable" users for a **20.2% efficiency gain**.

---

## Data Sources & Structure
The analysis is based on two primary datasets provided in the case study:

*   **`subscriptions.csv`**: Static information on trials.
    *   **Size**: ~503 raw rows, filtered to 416 for analysis.
    *   **Key Fields**: `subscription_id`, `trial_starts_at` (2023-2024 range), `v2_segment` (e.g., "TPE/PME"), `naf_section`, `revenue_range`.
    *   **Target**: Derived from `first_paid_invoice_paid_at` (1 = Converted, 0 = Churn).
*   **`daily_usage.csv`**: Time-series activity log.
    *   **Size**: ~11,685 rows.
    *   **Key Fields**: `day_date`, and 19+ activity counters (e.g., `nb_client_invoices_created`, `nb_banking_accounts_connected`).
    *   **Processing**: Aggregated to single-row features per subscription to capture total engagement and variability.

---

## Model Performance Leaderboard

| Model | Accuracy | ROC-AUC | PR-AUC | Calibration (Brier) |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **72.3%** | **0.790** | **0.835** | **0.193** |
| **GRU (RNN)** | 65.1% | 0.713 | 0.743 | 0.217 |
| **Transformer** | 66.3% | 0.711 | 0.748 | 0.211 |
| **Logistic Regression** | 65.1% | 0.684 | 0.769 | 0.229 |
| **XGBoost** | 63.9% | 0.671 | 0.772 | 0.242 |

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
Using pip:
```bash
pip install -r requirements.txt
```

Using poetry:
```bash
poetry install
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
│   ├── 07_uplift_causal_model.ipynb
│   └── 08_model_comparison.ipynb
├── models/                 # PyTorch Model Definitions (GRU/Transformer)
├── config/                 # Configuration Files
├── docs/                   # Documentation
├── results/                # Saved Artifacts (Models, Figures, Metrics)
│   ├── models/             # PKL and PT files
│   ├── preprocessor.pkl    # Fitted ColumnTransformer
│   ├── app_config.json     # App Dropdown Options
│   └── feature_columns.json # Feature Mapping
├── requirements.txt        # Dependencies
└── poetry.lock             # Poetry Lockfile
```

---

*Built for Client by "Antigravity"*
