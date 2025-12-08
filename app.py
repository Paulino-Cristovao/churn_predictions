"""
Gradio App for Kolecto Churn Prediction
Interactive demo to test all 5 models with custom inputs
"""
import gradio as gr
import pandas as pd
import numpy as np
import torch
import pickle
import json
from pathlib import Path

# Import models
import sys
sys.path.append('..')
from models.gru_model import GRUChurnModel
from models.transformer_model import TransformerChurnModel

# Load trained models
def load_models():
    """Load all trained models"""
    models = {}
    
    # Load tree-based models (assuming they're pickled)
    try:
        with open('../results/models/logistic_regression.pkl', 'rb') as f:
            models['Logistic Regression'] = pickle.load(f)
    except:
        models['Logistic Regression'] = None
    
    try:
        with open('../results/models/xgboost.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
    except:
        models['XGBoost'] = None
    
    try:
        with open('../results/models/lightgbm.pkl', 'rb') as f:
            models['LightGBM'] = pickle.load(f)
    except:
        models['LightGBM'] = None
    
    # Load deep learning models
    try:
        device = torch.device('cpu')
        gru = GRUChurnModel(20, 64, 2, 0.3).to(device)  # Adjust input_size as needed
        gru.load_state_dict(torch.load('../results/models/lstm_best_model.pt', map_location=device))
        gru.eval()
        models['LSTM/GRU'] = gru
    except Exception as e:
        print(f"Error loading GRU: {e}")
        models['LSTM/GRU'] = None
    
    try:
        trans = TransformerChurnModel(20, 64, 4, 2, 0.3).to(device)
        trans.load_state_dict(torch.load('../results/models/transformer_best_model.pt', map_location=device))
        trans.eval()
        models['Transformer'] = trans
    except Exception as e:
        print(f"Error loading Transformer: {e}")
        models['Transformer'] = None
    
    return models

# Global models
MODELS = load_models()

def predict_conversion(
    model_name,
    nb_transfers_sent,
    nb_transfers_received,
    nb_iban_verifications,
    nb_mobile_connections,
    nb_banking_accounts,
    nb_products_created,
    nb_invoices_created,
    nb_customers_created,
    nb_invoices_sent,
    nb_suppliers_created,
    nb_transactions_reconciled,
    company_age_years
):
    """
    Predict conversion probability for a trial user
    
    Returns:
        prediction: "Will Convert ✅" or "Won't Convert ❌"
        probability: Float probability of conversion
        confidence: Visual confidence bar
    """
    # Check if model is loaded
    model = MODELS.get(model_name)
    if model is None:
        return "Model not loaded ❌", 0.0, "Run notebook first to train models"
    
    # Prepare features (simplified - you'd aggregate properly for real usage)
    features = np.array([[
        nb_transfers_sent,
        nb_transfers_received,
        nb_iban_verifications,
        nb_mobile_connections,
        nb_banking_accounts,
        nb_products_created,
        nb_invoices_created,
        nb_customers_created,
        nb_invoices_sent,
        nb_suppliers_created,
        nb_transactions_reconciled,
        company_age_years,
        # Add more features as needed (padding for now)
        0, 0, 0, 0, 0, 0, 0, 0  # Placeholder for other features
    ]])
    
    # Make prediction based on model type
    try:
        if model_name in ['Logistic Regression', 'XGBoost', 'LightGBM']:
            # Tree-based models
            probability = model.predict_proba(features)[0][1]
        else:
            # Deep learning models
            with torch.no_grad():
                if model_name == 'LSTM/GRU':
                    # Reshape for sequence (15 days, features)
                    seq_features = features.reshape(1, 1, -1).repeat(15, axis=1)
                    probability = model(torch.FloatTensor(seq_features)).item()
                else:  # Transformer
                    seq_features = features.reshape(1, 1, -1).repeat(15, axis=1)
                    probability = model(torch.FloatTensor(seq_features)).item()
        
        # Determine prediction
        prediction = "✅ Will Convert" if probability >= 0.5 else "❌ Won't Convert"
        confidence_text = f"{'🟢' * int(probability * 10)}{'⚪' * (10 - int(probability * 10))}"
        
        return prediction, f"{probability:.1%}", confidence_text
        
    except Exception as e:
        return f"Error: {str(e)}", "0%", "❌"

# Create Gradio interface
def create_app():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Kolecto Churn Prediction", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 Kolecto Trial Conversion Predictor
        
        Test all 5 machine learning models to predict if a trial user will convert to paid subscription.
        
        **Input typical usage metrics from a 15-day trial and see the prediction!**
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 Model Selection")
                model_choice = gr.Radio(
                    choices=[
                        'Logistic Regression',
                        'XGBoost',
                        'LightGBM',
                        'LSTM/GRU',
                        'Transformer'
                    ],
                    value='XGBoost',
                    label="Choose Model"
                )
                
                gr.Markdown("### 📊 Trial Usage Features")
                gr.Markdown("*Enter total counts over 15-day trial*")
                
                nb_transfers_sent = gr.Slider(0, 100, value=5, label="Transfers Sent", step=1)
                nb_transfers_received = gr.Slider(0, 100, value=3, label="Transfers Received", step=1)
                nb_iban_verifications = gr.Slider(0, 20, value=1, label="IBAN Verifications", step=1)
                nb_mobile_connections = gr.Slider(0, 50, value=10, label="Mobile Connections", step=1)
                nb_banking_accounts = gr.Slider(0, 10, value=2, label="Banking Accounts Connected", step=1)
                nb_products_created = gr.Slider(0, 50, value=5, label="Products Created", step=1)
                nb_invoices_created = gr.Slider(0, 100, value=8, label="Invoices Created", step=1)
                nb_customers_created = gr.Slider(0, 50, value=4, label="Customers Created", step=1)
                nb_invoices_sent = gr.Slider(0, 50, value=3, label="Invoices Sent", step=1)
                nb_suppliers_created = gr.Slider(0, 30, value=2, label="Suppliers Created", step=1)
                nb_transactions_reconciled = gr.Slider(0, 100, value=10, label="Transactions Reconciled", step=1)
                
                gr.Markdown("### 🏢 Company Info")
                company_age_years = gr.Slider(0, 50, value=3, label="Company Age (years)", step=1)
                
                predict_btn = gr.Button("🔮 Predict Conversion", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### 📈 Prediction Results")
                
                prediction_output = gr.Textbox(label="Prediction", scale=2)
                probability_output = gr.Textbox(label="Conversion Probability", scale=2)
                confidence_output = gr.Textbox(label="Confidence Bar", scale=2)
                
                gr.Markdown("""
                ### 💡 Interpretation Guide
                
                - **✅ Will Convert**: Probability ≥ 50% → User likely to subscribe
                - **❌ Won't Convert**: Probability < 50% → User likely to cancel
                
                **Key Drivers:**
                - Creating invoices early in trial
                - Connecting banking accounts
                - High engagement (many logins)
                - Using diverse features
                
                **Recommendation:** Target users with <40% probability for intervention!
                """)
        
        # Connect prediction function
        predict_btn.click(
            fn=predict_conversion,
            inputs=[
                model_choice,
                nb_transfers_sent,
                nb_transfers_received,
                nb_iban_verifications,
                nb_mobile_connections,
                nb_banking_accounts,
                nb_products_created,
                nb_invoices_created,
                nb_customers_created,
                nb_invoices_sent,
                nb_suppliers_created,
                nb_transactions_reconciled,
                company_age_years
            ],
            outputs=[prediction_output, probability_output, confidence_output]
        )
        
        # Add examples
        gr.Examples(
            examples=[
                ['XGBoost', 10, 5, 2, 20, 3, 8, 15, 6, 10, 4, 20, 5],  # High engagement
                ['LSTM/GRU', 2, 1, 0, 5, 1, 2, 3, 1, 1, 1, 3, 1],     # Low engagement
                ['Transformer', 5, 3, 1, 12, 2, 5, 8, 3, 5, 2, 12, 3], # Medium
            ],
            inputs=[
                model_choice,
                nb_transfers_sent,
                nb_transfers_received,
                nb_iban_verifications,
                nb_mobile_connections,
                nb_banking_accounts,
                nb_products_created,
                nb_invoices_created,
                nb_customers_created,
                nb_invoices_sent,
                nb_suppliers_created,
                nb_transactions_reconciled,
                company_age_years
            ],
            label="Try Example Scenarios"
        )
    
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
