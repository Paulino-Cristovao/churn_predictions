# Kolecto Churn Prediction

Predict trial-to-paid conversion for Kolecto's 15-day trials using machine learning.

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the modular notebooks in order:
jupyter notebook notebooks/01_data_processing.ipynb
# ... then 02, 03, 04, 05, 06, 07
```

**Refactored Project**: The analysis is split into **7 modular notebooks** for better organization and reproducibility.

---

## 📊 Models Implemented

1. **Logistic Regression** - Baseline interpretable model
2. **XGBoost** - Gradient boosting (best tree-based)
3. **LightGBM** - Fast alternative (now with Optuna optimization)
4. **LSTM/GRU** - Sequential model for temporal patterns
5. **Transformer** - Attention-based sequential model

---

## 📁 Project Structure

```
churn_predictions/
├── notebooks/
│   ├── 01_data_processing.ipynb    # Data cleaning & splitting
│   ├── 02_logistic_regression.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_lightgbm.ipynb           # With Optuna tuning
│   ├── 05_gru_model.ipynb          # Deep Learning (GRU)
│   ├── 06_transformer_model.ipynb  # Deep Learning (Transformer)
│   └── 07_model_comparison.ipynb   # Final results aggregation
│
├── models/                         # Model class definitions
│   ├── gru_model.py
│   └── transformer_model.py
│
├── docs/
│   ├── MODEL_DOCUMENTATION.md      # Detailed theoretical guide
│   └── NEXT_STEPS.md               # Recommendations for improvement
│
├── data/
│   ├── raw/                        # Original CSVs
│   └── processed/                  # Generated pickles (churn_data.pkl)
│
└── results/                        # Generated Artifacts
    ├── models/                     # Saved models (e.g., xgboost/xgboost.pkl)
    ├── figures/                    # Plots (e.g., comparison/model_comparison.png)
    └── metrics/                    # JSON metrics
```

---

## 🚀 Usage

### Run Analysis
Execute the notebooks in numerical order (`01` to `07`).
- **01**: Generates `data/processed/churn_data.pkl` (Required by all others).
- **02-06**: Train individual models and save artifacts to `results/`.
- **07**: Loads all metrics and generates comparison charts.

### Interactive App
Launch the Gradio demo to test predictions:

```bash
python app.py
```
Open `http://localhost:7860`.

---

## 📈 Key Results

| Model | PR-AUC (Key Metric) | ROC-AUC | Accuracy |
|-------|---------------------|---------|----------|
| **XGBoost** | **0.7602** | 0.6200 | 59.04% |
| **LightGBM** | 0.7592 | 0.5818 | 54.22% |
| **LSTM/GRU** | 0.7241 | 0.6606 | 60.24% |
| **Transformer** | 0.7162 | 0.6485 | **61.45%** |
| **Logistic Regression** | 0.7145 | 0.6333 | 59.04% |

**Winner**: **XGBoost** slightly outperforms others in Precision-Recall AUC, making it the best candidate for identifying churners in this imbalanced dataset.

---

## 📚 Documentation
- **[MODEL_DOCUMENTATION.md](docs/MODEL_DOCUMENTATION.md)**: Deep dive into model architectures and validation.
- **[NEXT_STEPS.md](docs/NEXT_STEPS.md)**: Roadmap for future improvements.

## 📦 Requirements
- Pandas, NumPy, Scikit-learn
- XGBoost, LightGBM, Optuna
- PyTorch (torch)
- Matplotlib, Seaborn
- Gradio (for app)

See `requirements.txt`.

---

## 📝 License
MIT License
