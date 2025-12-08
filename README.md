# Churn Prediction for Kolecto Trial Conversions

Predict whether users will convert from trial to paid subscriptions using machine learning. This project implements **5 production-ready models** achieving up to **74% ROC-AUC** (XGBoost) and **78% PR-AUC** (GRU).

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
# Or use poetry
poetry install

# Explore the data
jupyter notebook notebooks/01_data_exploration.ipynb

# Run feature engineering
jupyter notebook notebooks/02_feature_engineering.ipynb

# Train tree-based models
jupyter notebook notebooks/03_tree_models.ipynb

# Train sequential models
python scripts/train_lstm_model.py
python scripts/train_transformer_model.py

# View sequential model analysis
jupyter notebook notebooks/04_sequential_models.ipynb

# Compare all models
python scripts/compare_models.py
```

---

## 📊 Model Performance Summary

| Model | Accuracy | ROC-AUC | PR-AUC | Brier | Best For |
|-------|----------|---------|--------|-------|----------|
| **XGBoost** ⭐ | **67.5%** | **0.737** | 0.720 | **0.180** | Overall best, production deployment |
| **LightGBM** | **67.5%** | 0.725 | 0.710 | 0.190 | Fast alternative |
| **GRU (Adamax)** | 60.2% | 0.718 | **0.781** ⭐ | 0.225 | Best precision-recall, temporal patterns |
| **Transformer** | 62.0% | 0.705 | 0.750 | 0.215 | Advanced sequential modeling |
| Logistic Regression | 60.2% | 0.636 | 0.620 | 0.220 | Baseline, interpretable |

**Improvement from baseline**: +14% accuracy, +12% ROC-AUC

---

## 🔑 Key Insights

### Top 5 Conversion Drivers (SHAP Analysis)
1. **Late trial activity** (days 12-14) → 3x higher conversion
2. **Feature diversity** (5+ features) → 2.5x boost
3. **Early engagement** (days 0-2) → 2x higher rate
4. **Invoice creation** by day 7 → 85% conversion
5. **Banking connections** → Strong commitment signal

### Business Recommendations
- **CS Team**: Proactive outreach for users with <3 features by day 7
- **Product**: Drive 5+ feature adoption in first 3 days
- **Marketing**: Focus on TPE/PME segments (65% vs 52% conversion)

**Expected Impact**: +4-6% conversion improvement (60.7% → 66%)

---

## 📁 Project Structure

```
churn_predictions/
├── data/                          # Data files
│   └── raw/                       # Original datasets
│       ├── daily_usage.csv
│       ├── subscriptions.csv
│       ├── Case Study Data Scientist.pdf
│       └── README.md             # Data dictionary
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Initial data analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation
│   ├── 03_tree_models.ipynb      # Tree-based models (LR, XGBoost, LightGBM)
│   ├── 04_sequential_models.ipynb # Sequential model analysis
│   └── churn_analysis_original.ipynb  # Original unified notebook (backup)
│
├── src/                           # Source code
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data loading and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py        # Feature engineering functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm/                 # GRU/LSTM implementation
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── train.py
│   │   └── transformer/          # Transformer implementation
│   │       ├── __init__.py
│   │       └── model.py
│   └── utils/
│       ├── __init__.py
│       └── validation.py         # Pydantic data models
│
├── scripts/                       # Training and evaluation scripts
│   ├── train_lstm_model.py       # Train GRU model
│   ├── train_transformer_model.py # Train Transformer model
│   └── compare_models.py         # Compare all models
│
├── results/                       # Outputs and artifacts
│   ├── models/                   # Saved model weights
│   │   ├── lstm_best_model.pt
│   │   └── transformer_best_model.pt
│   ├── figures/                  # Visualizations
│   │   ├── feature_importance.png
│   │   ├── roc_curves.png
│   │   ├── model_comparison.png
│   │   ├── lstm_training_curves.png
│   │   └── transformer_training_curves.png
│   └── metrics/                  # Model performance metrics
│       ├── lstm_results.json
│       └── transformer_results.json
│
├── docs/                          # Documentation
│   ├── MODEL_ANALYSIS_REPORT.md  # ⭐ Comprehensive analysis
│   ├── SUMMARY.md                # Complete summary
│   ├── executive_presentation.md # Business stakeholder deck
│   ├── next_steps.md            # Deployment roadmap
│   └── TRAINING_TIME_ANALYSIS.md # Training performance
│
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Poetry configuration
└── README.md                    # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (for sequential models)
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone https://github.com/Paulino-Cristovao/churn_predictions.git
cd churn_predictions

# Install dependencies
pip install -r requirements.txt

# Or use poetry
poetry install
```

### Running Notebooks

The project is organized into 4 sequential notebooks:

1. **Data Exploration** (`01_data_exploration.ipynb`)
   - Load and validate data
   - Explore conversion patterns
   - Identify data quality issues

2. **Feature Engineering** (`02_feature_engineering.ipynb`)
   - Create temporal features
   - Build milestone and diversity metrics
   - Analyze feature correlations

3. **Tree Models** (`03_tree_models.ipynb`)
   - Train Logistic Regression, XGBoost, LightGBM
   - SHAP analysis and interpretability
   - Model comparison

4. **Sequential Models** (`04_sequential_models.ipynb`)
   - Analyze LSTM/GRU and Transformer results
   - Compare with tree-based models
   - Business recommendations

### Training Models

```bash
# Train GRU model (100 epochs, ~10-15 minutes)
python scripts/train_lstm_model.py

# Train Transformer model (100 epochs, ~15-20 minutes)
python scripts/train_transformer_model.py

# Compare all models
python scripts/compare_models.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [MODEL_ANALYSIS_REPORT.md](docs/MODEL_ANALYSIS_REPORT.md) | **⭐ Comprehensive model comparison and recommendations** |
| [SUMMARY.md](docs/SUMMARY.md) | Complete project summary with all results |
| [executive_presentation.md](docs/executive_presentation.md) | Business stakeholder presentation |
| [next_steps.md](docs/next_steps.md) | Deployment roadmap and future work |
| [data/raw/README.md](data/raw/README.md) | Data dictionary and descriptions |

---

## 🔬 Technical Highlights

### Data Processing
- **Preprocessing**: Date handling, filtering, target definition (see `src/data/preprocessing.py`)
- **Feature Engineering**: 90+ features including temporal, milestone, diversity, and velocity metrics (see `src/features/engineering.py`)
- **Validation**: Pydantic models for data integrity (see `src/utils/validation.py`)

### Models
- **Tree-Based**: Logistic Regression, XGBoost, LightGBM with hyperparameter tuning
- **Sequential**: GRU with Adamax optimizer, Transformer with attention
- **Evaluation**: ROC-AUC, PR-AUC, Brier Score, calibration analysis

### Interpretability
- SHAP values for feature importance
- Temporal pattern analysis
- Conversion driver identification

---

## 🎯 Next Steps

1. ✅ **Complete** - All 5 models trained and documented
2. ✅ **Complete** - Project reorganized with clean structure
3. ⏭️ Deploy XGBoost to production API
4. ⏭️ CRM integration for CS team scoring
5. ⏭️ A/B test interventions
6. ⏭️ Monthly model retraining pipeline

---

## 📈 Results

The complete model comparison is available in `results/figures/model_comparison.png`.

Key findings:
- **XGBoost** achieves the best overall performance with 73.7% ROC-AUC
- **GRU** excels at precision-recall with 78.1% PR-AUC
- Sequential models better capture temporal patterns
- Tree-based models offer faster inference and easier interpretation

---

**Project Status**: ✅ Production Ready  
**Best Model**: XGBoost (73.7% ROC-AUC)  
**Best PR-AUC**: GRU (78.1%)  
**Last Updated**: December 8, 2025

**GitHub**: https://github.com/Paulino-Cristovao/churn_predictions

---

## 📝 License

This project is part of a data science case study for Kolecto.

## 👤 Author

**Paulino Cristovao**
