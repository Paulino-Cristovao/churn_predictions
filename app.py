"""
Gradio App for Client Churn Prediction
Interactive demo with Expanded Categorical Features
"""
import gradio as gr
import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
import sys

# Add path for model imports
sys.path.append('..')
# We import the classes, but we might need to redefine if checkpoints assume specific structure
from models.gru_model import GRUChurnModel
# Transformer defined locally to allow flexibility

# 1. Load Configuration & Preprocessor
print("Loading Feature Engineering Artifacts...")
try:
    with open('results/app_config.json', 'r') as f:
        APP_CONFIG = json.load(f)
    print("Loaded app_config.json")
    
    with open('results/preprocessor.pkl', 'rb') as f:
        PREPROCESSOR = pickle.load(f)
    print("Loaded preprocessor.pkl")
    
    # Calculate Input Size from one transform
    # We can infer input size from feature names or by testing
    try:
        INPUT_SIZE = len(PREPROCESSOR.get_feature_names_out())
    except:
        # Fallback if scikit version old
        INPUT_SIZE = 120 # Estimate
    print(f"Feature Dimension: {INPUT_SIZE}")

except Exception as e:
    print(f"Artifacts missing: {e}")
    APP_CONFIG = {}
    PREPROCESSOR = None
    INPUT_SIZE = 120

# Load trained models
def load_models():
    """Load all trained models"""
    models = {}
    
    # Load tree-based models
    for name, path in [
        ('Logistic Regression', 'results/models/logistic_regression/logistic_regression.pkl'),
        ('XGBoost', 'results/models/xgboost/xgboost.pkl'),
        ('LightGBM', 'results/models/lightgbm/lightgbm.pkl')
    ]:
        try:
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
                print(f"Loaded {name}")
        except Exception as e:
            print(f"{name} not available: {e}")
            models[name] = None
    
    # Load deep learning models
    device = torch.device('cpu')
    
    # GRU
    try:
        # Load model with correct dimensions
        gru = GRUChurnModel(INPUT_SIZE, 64, 2, 0.3).to(device)
        state_dict = torch.load('results/models/gru/gru_best_model.pt', map_location=device)
        gru.load_state_dict(state_dict)
        gru.eval()
        models['LSTM/GRU'] = gru
        print("Loaded LSTM/GRU")
    except Exception as e:
        print(f"LSTM/GRU not available: {e}")
        models['LSTM/GRU'] = None
    
    # Transformer
    try:
        import torch.nn as nn
        import math
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super(PositionalEncoding, self).__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe.unsqueeze(0))
            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        class TransformerChurnModel(nn.Module):
            def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3):
                super(TransformerChurnModel, self).__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
                    dropout=dropout, batch_first=False 
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc1 = nn.Linear(d_model, 16)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(16, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = x.permute(1, 0, 2) # (Seq, Batch, Feat)
                out = self.transformer_encoder(x)
                out = out.mean(dim=0)
                out = self.fc1(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                return self.sigmoid(out).squeeze()

        trans = TransformerChurnModel(INPUT_SIZE, 64, 4, 2, 0.3).to(device)
        trans.load_state_dict(torch.load('results/models/transformer/transformer_best_model.pt', map_location=device))
        trans.eval()
        models['Transformer'] = trans
        print("Loaded Transformer")
    except Exception as e:
        print(f"Transformer not available: {e}")
        models['Transformer'] = None
    
    return models

# Global models
MODELS = load_models()

def predict_conversion(
    model_name,
    vendor, v2_segment, naf_section, revenue_range, employee_count, 
    regional_pole, market, legal_structure, company_age_group, naf_code,
    nb_transfers_sent, nb_transfers_received, nb_iban_verifications,
    nb_mobile_connections, nb_banking_accounts, nb_products_created,
    nb_invoices_created, nb_customers_created, nb_invoices_sent,
    nb_suppliers_created, nb_transactions_reconciled, company_age_years
):
    """Predict conversion with ColumnTransformer Pipeline"""
    model = MODELS.get(model_name)
    if model is None:
        return "Model not loaded", "0.0%", "Error"
    
    if PREPROCESSOR is None:
        return "Preprocessor not loaded", "0.0%", "Error"

    # 1. Create Input DataFrame (Raw Strings & Nums)
    # We must match the columns expected by the transformer
    # Which are: num_cols + ohe_cols + ord_cols
    # num_cols are nb_... + company_age_years? 
    # Wait, in the rewrite script, I selected num_cols = df.select_dtypes(include=[np.number])
    # And df came from aggregated usage.
    # The usage agg has cols like: nb_transfers_sent_sum, nb_transfers_sent_mean, etc.
    # The inputs here are just "nb_transfers_sent" (single value).
    # SIMPLIFICATION: We must generate the 4 aggregates (sum, mean, max, std) for each input metric
    # To match the ~76 numeric columns.
    
    raw_input = {}
    
    # 1a. Categoricals
    raw_input['vendor'] = [vendor]
    raw_input['v2_segment'] = [v2_segment]
    raw_input['naf_section'] = [naf_section]
    raw_input['revenue_range'] = [revenue_range]
    raw_input['employee_count'] = [employee_count]
    raw_input['regional_pole'] = [regional_pole]
    raw_input['market'] = [market]
    raw_input['legal_structure'] = [legal_structure]
    raw_input['company_age_group'] = [company_age_group]
    raw_input['naf_code'] = [naf_code]
    _ = [0] # Dummy for subscription_id if needed? No, transform doesn't need it.

    # 1b. Numerics (Metrics)
    # We need to provide ALL columns expected by the Scaler.
    # Retrieve expected columns from the preprocessor's 'num' transformer
    # Structure: transformers_ list of (name, transformer, columns)
    try:
        # Find 'num' transformer
        num_cols_expected = []
        for name, trans, cols in PREPROCESSOR.transformers_:
            if name == 'num':
                num_cols_expected = cols
                break
        
        # Initialize all expected numerics to 0
        for col in num_cols_expected:
            raw_input[col] = [0]
            
        # Set trial_duration if present
        if 'trial_duration' in num_cols_expected:
            raw_input['trial_duration'] = [15]

        # Map UI inputs to specific columns
        # We loop through our known UI metrics and populate their aggregates
        ui_metrics = {
            'nb_transfers_sent': nb_transfers_sent,
            'nb_transfers_received': nb_transfers_received,
            'nb_iban_verification_requests_created': nb_iban_verifications,
            'nb_mobile_connections': nb_mobile_connections,
            'nb_banking_accounts_connected': nb_banking_accounts,
            'nb_products_created': nb_products_created,
            'nb_client_invoices_created': nb_invoices_created,
            'nb_customers_created': nb_customers_created,
            'nb_client_invoices_sent': nb_invoices_sent,
            'nb_suppliers_created': nb_suppliers_created,
            'nb_transactions_reconciled': nb_transactions_reconciled
        }
        
        # Populate aggregates (sum, mean, max, std)
        for metric, val in ui_metrics.items():
            for agg in ['sum', 'mean', 'max', 'std']:
                col_name = f"{metric}_{agg}"
                if col_name in num_cols_expected:
                    # Simple heuristic for aggregates based on single total
                    if agg == 'sum':
                        raw_input[col_name] = [val]
                    elif agg == 'mean':
                        raw_input[col_name] = [val / 15.0]
                    elif agg == 'max':
                        raw_input[col_name] = [val / 15.0 * 2] # Dummy rough max
                    elif agg == 'std':
                        raw_input[col_name] = [val / 15.0 * 0.5] # Dummy rough std
                        
        # Company Age
        if 'company_age_in_years' in num_cols_expected:
            raw_input['company_age_in_years'] = [company_age_years]
            
    except Exception as e:
        return f"Feature Mapping Error: {e}", "0%", "Error"
    
    # Construct DF
    input_df = pd.DataFrame(raw_input)
    
    # 2. Transform
    # ColumnTransformer will select columns by name and transform them
    # It handles order automatically
    try:
        final_input_ohe = PREPROCESSOR.transform(input_df) # Returns numpy array
    except Exception as e:
        return f"Preprocessing Error: {e}", "0%", "Error"
        
    final_input = final_input_ohe.astype(float)
    
    # 3. Predict
    try:
        if model_name in ['Logistic Regression', 'XGBoost', 'LightGBM']:
            probability = model.predict_proba(final_input)[0][1]
        else:
            # DL models expect (Batch, 1, Features)
            tensor_in = torch.FloatTensor(final_input).unsqueeze(1) # (1, 1, F)
            with torch.no_grad():
                probability = model(tensor_in).item()
        
        prediction = "Will Convert" if probability >= 0.5 else "Won't Convert"
        confidence_text = f"{'|' * int(probability * 10)}{'.' * (10 - int(probability * 10))}"
     # Gradio App for SaaS Churn Prediction
import gradio as gr
import pandas as pd
import joblib
import torch
import numpy as np

# Load generic resources
print("Loading resources...")

def predict_churn(company_age, nb_transfers_sent, nb_mobile_connections, total_invoices):
    # Dummy logic for demonstration if models not found
    return "0.45", "Medium Risk"

with gr.Blocks(title="SaaS Churn Prediction Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SaaS Churn Prediction (Categorical Enhanced)")
        gr.Markdown("Input rich customer profile data to predict conversion probability.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Strategy")
                model_choice = gr.Radio(
                    ['Logistic Regression', 'XGBoost', 'LightGBM', 'LSTM/GRU', 'Transformer'],
                    value='XGBoost', label="Model"
                )
                
                gr.Markdown("### Company Profile")
                # Dynamic Dropdowns
                # Helper to get options or default
                def get_opts(key, default=['Unknown']):
                    return APP_CONFIG.get(key, default)
                
                vendor = gr.Dropdown(get_opts('vendor'), label="Vendor", value='CA')
                segment = gr.Dropdown(get_opts('v2_segment'), label="Segment", value='TPE')
                legal = gr.Dropdown(get_opts('legal_structure'), label="Legal Structure", value=get_opts('legal_structure')[0])
                naf_sec = gr.Dropdown(get_opts('naf_section'), label="Industry (NAF Section)", value='M')
                rev = gr.Dropdown(get_opts('revenue_range'), label="Revenue Range", value='Unknown')
                emp = gr.Dropdown(get_opts('employee_count'), label="Employee Count", value='Unknown')
                pole = gr.Dropdown(get_opts('regional_pole'), label="Regional Pole", value='Unknown')
                market = gr.Dropdown(get_opts('market'), label="Market", value='PRO')
                age_grp = gr.Dropdown(get_opts('company_age_group'), label="Age Group", value='Non renseigné')
                naf_code = gr.Dropdown(get_opts('naf_code'), label="NAF Code", value=get_opts('naf_code')[0])
                
            with gr.Column(scale=1):
                gr.Markdown("### Usage Activity (15 Days)")
                nb_invoices_created = gr.Slider(0, 100, 8, label="Invoices Created")
                nb_transfers_sent = gr.Slider(0, 100, 5, label="Transfers Sent")
                nb_banking_accounts = gr.Slider(0, 10, 2, label="Banking Accounts")
                nb_mobile_connections = gr.Slider(0, 50, 10, label="Mobile Conns")
                nb_products_created = gr.Slider(0, 50, 5, label="Products")
                nb_customers_created = gr.Slider(0, 50, 4, label="Customers")
                nb_transfers_received = gr.Slider(0, 100, 1, label="Transfers Recv")
                nb_iban_verifications = gr.Slider(0, 20, 0, label="IBAN Verif")
                nb_invoices_sent = gr.Slider(0, 50, 3, label="Invoices Sent")
                nb_suppliers_created = gr.Slider(0, 30, 2, label="Suppliers")
                nb_transactions_reconciled = gr.Slider(0, 100, 10, label="Trans. Reconciled")
                company_age_years = gr.Slider(0, 50, 3, label="Company Age (Years)")
                
                predict_btn = gr.Button("Predict Score", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Result")
                pred_out = gr.Textbox(label="Decision", show_label=True)
                prob_out = gr.Textbox(label="Probability", show_label=True)
                conf_out = gr.Textbox(label="Confidence", show_label=True)
                
        predict_btn.click(
            predict_conversion,
            inputs=[
                model_choice,
                vendor, segment, naf_sec, rev, emp, pole, market, legal, age_grp, naf_code,
                nb_transfers_sent, nb_transfers_received, nb_iban_verifications,
                nb_mobile_connections, nb_banking_accounts, nb_products_created,
                nb_invoices_created, nb_customers_created, nb_invoices_sent,
                nb_suppliers_created, nb_transactions_reconciled, company_age_years
            ],
            outputs=[pred_out, prob_out, conf_out]
        )

    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
