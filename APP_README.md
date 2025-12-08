# Kolecto Churn Prediction - Interactive Demo 🎯

Run the Gradio app to test model predictions interactively!

## Quick Start

```bash
# Install Gradio
pip install gradio

# Run the app
python app.py
```

The app will launch at `http://localhost:7860`

## Features

- **Select any of 5 models**: Logistic Regression, XGBoost, LightGBM, LSTM/GRU, Transformer
- **Input trial usage features**: Invoices, products, bank connections, etc.
- **See instant prediction**: Will convert or not
- **View probability**: Confidence percentage
- **Visual confidence bar**: Easy interpretation

## Usage

1. Choose a model from the dropdown
2. Adjust sliders for trial usage metrics
3. Click "Predict Conversion"
4. See results: prediction, probability, confidence

## Example Scenarios

- **High Engagement**: Many invoices, products, bank connections → Likely converts ✅
- **Low Engagement**: Few activities, no banking setup → Likely cancels ❌
- **Medium**: Some usage → Borderline (target for intervention)

Run it now: `python app.py`
