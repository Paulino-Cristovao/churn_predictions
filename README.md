# Kolecto Churn Prediction

This project analyzes user behavior during the 15-day trial period to predict conversion to paid subscriptions for Kolecto.

## Project Overview

The goal is to identify early signals of cancellation or successful conversion to help the Customer Experience (CX) team take targeted actions.
We use machine learning models (Logistic Regression, XGBoost, LightGBM) to predict the probability of conversion based on usage data and company characteristics.

## Data

The analysis uses two datasets located in the `Data/` directory:
- `daily_usage.csv`: Daily user activity metrics.
- `subscriptions.csv`: Subscription details and status.

## Installation

This project requires Python 3.9+.

1. Clone the repository:
   ```bash
   git clone https://github.com/Paulino-Cristovao/churn_predictions.git
   cd churn_predictions
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # OR if using pyproject.toml directly with a tool like poetry or flit
   pip install .
   ```
   *Note: You may need to install `libomp` for XGBoost/LightGBM on macOS (`brew install libomp`).*

## Usage

The main analysis is contained in the Jupyter Notebook `churn_analysis.ipynb`.

To run the analysis:
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `churn_analysis.ipynb` and run all cells.

## Results

The analysis generates:
- **ROC Curves**: Visualizing model performance (`roc_curves.png`).
- **Feature Importance**: Highlighting key factors driving conversion (`feature_importance.png`).

Key findings and recommendations are summarized in the notebook and the presentation slides.
