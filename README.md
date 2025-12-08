# Churn Prediction for Kolecto Trial Conversions

Predict whether users will convert from trial to paid subscriptions using machine learning. This project implements **4 production-ready models** achieving up to **74% AUC** (XGBoost) and **78% PR-AUC** (GRU).

## 🎯 Quick Start

```bash
# Install dependencies
poetry install

# Train all models
python train_lstm_model.py         # GRU sequential model
poetry run jupyter notebook        # Open churn_analysis.ipynb for tree models

# Compare all 4 models
python compare_models.py

# View results
open results/model_comparison_comprehensive.png
```

---

## 📊 Model Performance Summary

| Model | Accuracy | ROC-AUC | PR-AUC | Brier | Best For |
|-------|----------|---------|--------|-------|----------|
| **XGBoost** ⭐ | **67.5%** | **0.737** | 0.720 | **0.180** | Overall best, production deployment |
| **LightGBM** | **67.5%** | 0.725 | 0.710 | 0.190 | Fast alternative |
| **GRU (Adamax)** | 60.2% | 0.718 | **0.781** ⭐ | 0.225 | Best precision-recall, temporal patterns |
| Logistic Regression | 60.2% | 0.636 | 0.620 | 0.220 | Baseline, interpretable |

**Improvement from baseline**: +14% accuracy, +12% AUC

---

## 🔑 Key Insights

### Top 5 Conversion Drivers (SHAP Analysis)
1. **Late trial activity** (days 12-14) → 3x higher conversion
2. **Feature diversity** (5+ features) → 2.5x boost
3. **Early engagement** (days 0-2) →2x higher rate
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
├── Data/                       # Raw datasets
├── models/lstm/               # GRU sequential model
│   ├── model.py              # Architecture (42K params)
│   ├── train.py              # Training utilities
│   ├── best_model.pt         # Trained weights (170 KB)
│   ├── results.json          # Test metrics
│   └── training_curves.png
├── results/                   # Outputs
│   └── model_comparison_comprehensive.png
├── churn_analysis.ipynb      # Main analysis (tree models)
├── train_lstm_model.py       # GRU training script
├── compare_models.py         # 4-model comparison
├── MODEL_ANALYSIS_REPORT.md  # ⭐ Comprehensive analysis
├── SUMMARY.md                # Complete summary
├── executive_presentation.md # Business deck
├── next_steps.md            # Deployment roadmap
└── README.md                # This file
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [MODEL_ANALYSIS_REPORT.md](MODEL_ANALYSIS_REPORT.md) | **⭐ Comprehensive model comparison, evolution, recommendations** |
| [SUMMARY.md](SUMMARY.md) | Complete project summary with all results |
| [executive_presentation.md](executive_presentation.md) | Business stakeholder deck |
| [next_steps.md](next_steps.md) | Deployment roadmap |

---

## 🚀 Technical Highlights

- **Data Quality**: Deduplication, median imputation, normalization, anomaly detection
- **Feature Engineering**: Temporal patterns, milestones, velocity, consistency metrics
- **Model Optimization**: Early stopping, advanced regularization, Adamax optimizer
- **Interpretability**: SHAP analysis, calibration plots, feature importance

---

## 🎯 Next Steps

1. ✅ **Complete** - All 4 models trained and documented
2. ⏭️ Deploy XGBoost to production API
3. ⏭️ CRM integration for CS team scoring
4. ⏭️ A/B test interventions
5. ⏭️ Monthly model retraining

---

**Project Status**: ✅ Production Ready  
**Best Model**: XGBoost (73.7% AUC)  
**Best PR-AUC**: GRU (78.1%)  
**Last Updated**: December 8, 2025

**GitHub**: https://github.com/Paulino-Cristovao/churn_predictions
