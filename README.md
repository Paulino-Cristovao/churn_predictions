# Kolecto Churn Prediction

Predict trial-to-paid conversion for Kolecto's 15-day trials using machine learning.

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis (trains all 5 models)
jupyter notebook notebooks/churn_analysis.ipynb
```

**One notebook** contains everything - data loading, preprocessing, and training for all 5 models.

---

## 📊 Models Implemented

1. **Logistic Regression** - Baseline interpretable model
2. **XGBoost** - Gradient boosting (best tree-based)
3. **LightGBM** - Fast alternative
4. **LSTM/GRU** - Sequential model for temporal patterns
5. **Transformer** - Attention-based sequential model

All models train in ~5-7 minutes total.

---

## 📁 Project Structure

```
churn_predictions/
├── notebooks/
│   └── churn_analysis.ipynb    # Complete analysis (all 5 models)
│
├── models/                      # Model class definitions
│   ├── gru_model.py            # LSTM/GRU architecture
│   └── transformer_model.py    # Transformer architecture
│
├── config/
│   └── model_config.py         # Hyperparameters for deep learning
│
├── data/raw/                    # Original data
│   ├── subscriptions.csv
│   ├── daily_usage.csv
│   └── Case Study Data Scientist.pdf
│
└── results/                     # Generated when notebook runs
    ├── models/                  # Trained model weights
    ├── figures/                 # Visualizations
    └── metrics/                 # Performance JSONs
```

---

## 🚀 Usage

### Run Complete Analysis

```bash
jupyter notebook notebooks/churn_analysis.ipynb
# Then: Cell → Run All
```

**What it does:**
1. Loads & preprocesses data (15-day trials)
2. Trains Logistic Regression, XGBoost, LightGBM
3. Trains LSTM/GRU and Transformer models
4. Generates comparison plots (ROC, PR curves)
5. Saves all results to `results/`

### Modify Hyperparameters

Edit `config/model_config.py` and re-run notebook.

---

## 📈 Expected Results

**Tree-Based Models:**
- Logistic Regression: ~0.64 ROC-AUC
- XGBoost: ~0.74 ROC-AUC
- LightGBM: ~0.73 ROC-AUC

**Deep Learning:**
- LSTM/GRU: ~0.72 ROC-AUC, ~0.80 PR-AUC ⭐
- Transformer: ~0.71 ROC-AUC

All models evaluated on same test set with comprehensive metrics.

---

## 🔑 Key Features

- **Single notebook** - Complete analysis in one file
- **Reproducible** - Fixed random seeds
- **Organized code** - Model classes in separate files
- **Comprehensive evaluation** - Multiple metrics + plots
- **Easy to extend** - Add new models easily

---

## 📦 Requirements

- Python 3.10+
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- torch (PyTorch for deep learning)
- matplotlib, seaborn
- jupyter

See `requirements.txt` for complete list.

---

## 🎓 Case Study

This project addresses the Kolecto data scientist case study:
- **Goal**: Predict 15-day trial conversion (~60% baseline)
- **Data**: Subscriptions + daily usage features
- **Deliverable**: ML models + insights for Customer Experience team

---

## 📝 License

MIT License

---

**Ready to run!** Just open the notebook and execute all cells.
