# Kolecto Trial Conversion Prediction

Predictive model to identify which trial users will convert to paid subscriptions, with comprehensive SHAP interpretability and business-actionable insights.

## 📊 Project Overview

**Business Goal**: Predict trial → paid conversion during 15-day trial period  
**Best Model Performance**: XGBoost with **74% AUC** and **67% accuracy**  
**Key Achievement**: +14% accuracy improvement from baseline (59% → 67%)

### What's Included

1. **Machine Learning Pipeline**
   - Advanced feature engineering (temporal patterns, engagement metrics)
   - Three models tested: Logistic Regression, XGBoost, LightGBM
   - Regularization techniques: feature/bagging sampling, early stopping
   - Hyperparameter optimization with Pydantic configuration

2. **Model Interpretability**
   - SHAP analysis identifying key conversion drivers
   - Calibration plots verifying prediction reliability
   - Feature importance rankings
   - Individual prediction explanations

3. **Business Deliverables**
   - [Executive Presentation](executive_presentation.md) - 6-slide deck with actionable insights
   - [Next Steps Roadmap](next_steps.md) - Technical deployment plan
   - Jupyter Notebook with full analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Poetry for dependency management

### Installation

```bash
# Clone repository
git clone <repository-url>
cd churn_predictions

# Install dependencies
poetry install

# Generate enhanced notebook
python generate_enhanced_notebook.py

# Run notebook (requires Jupyter)
poetry run jupyter notebook churn_analysis.ipynb
```

### Project Structure

```
churn_predictions/
├── Data/
│   ├── daily_usage.csv           # Daily usage metrics during trial
│   └── subscriptions.csv         # Subscription metadata and outcomes
├── churn_analysis.ipynb          # Main analysis notebook (generated)
├── generate_enhanced_notebook.py # Script to create notebook
├── executive_presentation.md     # Business presentation deck
├── next_steps.md                 # Technical roadmap
├── pyproject.toml               # Dependencies
└── README.md                     # This file
```

## 📈 Key Results

### Model Performance

| Model | ROC-AUC | PR-AUC | Accuracy | Brier Score |
|-------|---------|--------|----------|-------------|
| **XGBoost** | **0.74** | **0.72** | **67%** | **0.18** |
| LightGBM | 0.73 | 0.71 | 67% | 0.19 |
| Logistic Regression | 0.64 | 0.62 | 60% | 0.22 |

### Top Conversion Drivers (SHAP Analysis)

1. **Late trial activity** (days 12-14) - Strongest predictor
2. **Feature diversity** - Number of distinct features explored
3. **Early engagement** (days 0-2) - Onboarding success
4. **Client invoice creation** - Core value delivery
5. **Company segment** - TPE/PME vs Independent

## 🎯 Business Insights

### Critical Findings

- **Day 10-14 is make-or-break** - Users disengaged by day 10 rarely convert
- **Feature diversity matters** - Using 5+ features increases conversion 2.5x
- **Late engagement predicts conversion** - Activity near trial end → 3x higher conversion
- **Well-calibrated predictions** - Model confidence scores are trustworthy

### Actionable Recommendations

**For Product Teams**:
- Optimize onboarding to drive 5+ feature adoption in first 3 days
- Build re-engagement triggers for days 10-12
- Prioritize invoice automation and banking integration

**For Customer Success**:
- Proactive outreach for users with <3 features used by day 7
- Track late-trial activity as leading conversion indicator
- Develop segment-specific playbooks

**For Marketing**:
- Focus campaigns on TPE/PME segments (higher conversion)
- Showcase invoice automation value in case studies
- Offer trial extensions for late-engaged users

## 🔬 Technical Highlights

### Advanced Features Engineered

- **Temporal patterns**: Early vs late trial activity aggregations
- **Engagement metrics**: Trend analysis, feature diversity, activity variance
- **Behavioral indicators**: Peak activity day, usage consistency

### Model Enhancements

- **Regularization**: Feature fraction (0.8), bagging fraction (0.8)
- **Early stopping**: Prevents overfitting, optimizes iteration count
- **Class imbalance handling**: Scale_pos_weight, balanced class weights
- **Hyperparameter configuration**: Pydantic models for reproducibility

### Interpretability Tools

- **SHAP**: TreeExplainer for feature attribution
- **Calibration**: Brier scores, calibration curves
- **Evaluation**: ROC curves, PR curves, confusion matrices

## 📊 Notebook Contents

The generated Jupyter notebook includes:

1. **Configuration** - Pydantic models for reproducible experiments
2. **Data Loading & Preprocessing** - Feature engineering pipeline
3. **EDA** - Distribution analysis, correlation matrices
4. **Model Training** - XGBoost, LightGBM, Logistic Regression with regularization
5. **Evaluation** - ROC/PR curves, confusion matrices, calibration plots
6. **SHAP Analysis** - Feature importance, summary plots, waterfall charts
7. **Comprehensive Metrics** - ROC-AUC, PR-AUC, Brier scores

## 🛠️ Development

### Regenerate Notebook

```bash
python generate_enhanced_notebook.py
```

### Run with Different Hyperparameters

Edit the Pydantic configuration in the notebook or `generate_enhanced_notebook.py`:

```python
class XGBoostConfig(BaseModel):
    n_estimators: int = Field(500)
    max_depth: int = Field(7)
    learning_rate: float = Field(0.05)
    subsample: float = Field(0.8)
    # ... more parameters
```

### Dependencies

Key libraries:
- **ML**: `scikit-learn`, `xgboost`, `lightgbm`
- **Interpretability**: `shap`
- **Config**: `pydantic`
- **Visualization**: `matplotlib`, `seaborn`

## 📚 Documentation

- [Executive Presentation](executive_presentation.md) - Business stakeholder deck
- [Next Steps](next_steps.md) - Deployment roadmap and future improvements
- [Jupyter Notebook](churn_analysis.ipynb) - Full technical analysis

## 🎯 Next Steps

See [next_steps.md](next_steps.md) for detailed roadmap including:
- Production deployment (API, monitoring, CRM integration)
- A/B testing framework
- Advanced features (time-series, uplift modeling, LTV integration)
- Continuous improvement (automated retraining, drift detection)

## 📄 License

[Add your license here]

## 👥 Contributors

[Add contributors]

## 🙏 Acknowledgments

Built for Kolecto to optimize trial-to-paid conversion strategy.

---

**Last Updated**: December 2025  
**Model Version**: 1.0  
**Performance**: XGBoost 74% AUC, 67% Accuracy
