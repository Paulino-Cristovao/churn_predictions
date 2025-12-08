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
    
    # Load tree-based models (if they exist)
    try:
        with open('results/models/logistic_regression/logistic_regression.pkl', 'rb') as f:
            models['Logistic Regression'] = pickle.load(f)
            print("✅ Loaded Logistic Regression")
    except Exception as e:
        print(f"⚠️  Logistic Regression not available: {e}")
        models['Logistic Regression'] = None
    
    try:
        with open('results/models/xgboost/xgboost.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
            print("✅ Loaded XGBoost")
    except Exception as e:
        print(f"⚠️  XGBoost not available: {e}")
        models['XGBoost'] = None
    
    try:
        with open('results/models/lightgbm/lightgbm.pkl', 'rb') as f:
            models['LightGBM'] = pickle.load(f)
            print("✅ Loaded LightGBM")
    except Exception as e:
        print(f"⚠️  LightGBM not available: {e}")
        models['LightGBM'] = None
    
    # Load deep learning models
    try:
        device = torch.device('cpu')
        # GRU was trained with fc1 output size 16 (not 32 as in current gru_model.py)
        # Load model with strict=False to handle minor architecture differences
        gru = GRUChurnModel(19, 64, 2, 0.3).to(device)
        state_dict = torch.load('results/models/gru/gru_best_model.pt', map_location=device)
        # Modify fc1 layer to match saved model (16 instead of 32)
        gru.fc1 = torch.nn.Linear(64, 16).to(device)
        gru.fc2 = torch.nn.Linear(16, 1).to(device)
        gru.load_state_dict(state_dict)
        gru.eval()
        models['LSTM/GRU'] = gru
        print("✅ Loaded LSTM/GRU")
    except Exception as e:
        print(f"⚠️  LSTM/GRU not available: {e}")
        models['LSTM/GRU'] = None
    
    try:
        device = torch.device('cpu')
        # Transformer was trained with input_projection (not input_proj) and dim_feedforward=128
        # Import and create custom Transformer loader
        import torch.nn as nn
        from models.transformer_model import PositionalEncoding
        
        class TransformerChurnModel(nn.Module):
            """Match the architecture of the saved transformer model (76 features)"""
            def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3):
                super(TransformerChurnModel, self).__init__()
                
                self.input_projection = nn.Linear(input_size, d_model)
                # Checkpoint has [1, 5000, 64] (default from notebook)
                # models/transformer_model.py default is 15. We must force 5000.
                self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=d_model * 2,
                    dropout=dropout, batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                self.fc1 = nn.Linear(d_model, 16)  # Trained with 16, not 32
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(16, 1)        # Trained with 16->1
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                # Project input features to d_model dimension
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                
                # Transformer encoding
                # Permute for transformer: (Seq, Batch, Feature) -> (Batch, Seq, Feature) is batch_first=True??
                # Wait, trained model in 06_transformer used default batch_first=False for encoder??
                # Notebook 06: `encoder_layers = nn.TransformerEncoderLayer(..., batch_first=False)` (Default)
                # Let's check 06 notebook code I wrote:
                # `encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout)` -> Default batch_first=False
                # `x = x.permute(1, 0, 2)` -> (Seq, Batch, Feature) 
                # `transformer_out = self.transformer_encoder(x)`
                
                # So I need to match EXACTLY what was in 06_transformer_model.ipynb
                
                # Replicating 06_transformer_model struct:
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = x.permute(1, 0, 2) # (Seq, Batch, Feature)
                transformer_out = self.transformer_encoder(x)
                pooled = transformer_out.mean(dim=0) # Mean across Seq dim (dim 0 now)
                
                out = self.fc1(pooled)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                out = self.sigmoid(out)
                return out.squeeze()
        
        # Load model with input_size=76 (aggregated features)
        trans = TransformerChurnModel(76, 64, 4, 2, 0.3).to(device)
        trans.load_state_dict(torch.load('results/models/transformer/transformer_best_model.pt', map_location=device))
        trans.eval()
        models['Transformer'] = trans
        print("✅ Loaded Transformer")
    except Exception as e:
        print(f"⚠️  Transformer not available: {e}")
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
        return "❌ Model not loaded", "0.0%", "⚠️  Run notebook first to train models"
    
    # Prepare features based on model type
    # For tree-based models: Use 76 aggregated features (sum, mean, max, std of 19 metrics)
    # For deep learning: Use 19 features per timestep over 15 days
    
    # Create basic 19 daily features (simplified - assumes equal distribution)
    basic_features = [
        nb_transfers_sent / 15,
        nb_transfers_received / 15,
        nb_iban_verifications / 15,
        nb_mobile_connections / 15,
        nb_banking_accounts / 15,
        nb_products_created / 15,
        nb_invoices_created / 15,
        nb_customers_created / 15,
        nb_invoices_sent / 15,
        nb_suppliers_created / 15,
        nb_transactions_reconciled / 15,
        company_age_years,
        0, 0, 0, 0, 0, 0, 0  # Padding for other 7 features (nb_clicks, nb_searches, etc.)
    ]
    
    # Make prediction based on model type
    try:
        if model_name in ['Logistic Regression', 'XGBoost', 'LightGBM']:
            # Tree-based models expect 76 features (19 metrics × 4 aggregations)
            # Create aggregated features: sum, mean, max, std
            total_values = [
                nb_transfers_sent, nb_transfers_received, nb_iban_verifications,
                nb_mobile_connections, nb_banking_accounts, nb_products_created,
                nb_invoices_created, nb_customers_created, nb_invoices_sent,
                nb_suppliers_created, nb_transactions_reconciled, company_age_years,
                0, 0, 0, 0, 0, 0, 0  # Padding
            ]
            # Aggregate: sum, mean, max, std (simplified - use same for all)
            features = np.array([total_values + basic_features + total_values + basic_features])
            probability = model.predict_proba(features)[0][1]
        else:
            # Deep learning models expect (batch, 15 timesteps, 19 features)
            with torch.no_grad():
                # Create sequence: repeat daily average over 15 days
                seq_features = np.array([basic_features] * 15).reshape(1, 15, 19)
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
