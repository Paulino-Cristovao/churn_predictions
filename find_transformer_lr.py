"""
Learning Rate Search for Transformer Model
Tests multiple learning rates and selects the best one
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, brier_score_loss
import json
import os
import sys
sys.path.append('.')

from models.transformer.model import TransformerChurnModel
from models.lstm.train import SequenceDataset, prepare_sequence_data

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Using device: {device}')

# Load data  
print("📂 Loading data...")
daily_usage_df = pd.read_csv('Data/daily_usage.csv')
subscriptions_df = pd.read_csv('Data/subscriptions.csv')

# Preprocess
subscriptions_df['trial_starts_at'] = pd.to_datetime(subscriptions_df['trial_starts_at'], errors='coerce')
subscriptions_df['trial_ends_at'] = pd.to_datetime(subscriptions_df['trial_ends_at'], errors='coerce')
daily_usage_df['day_date'] = pd.to_datetime(daily_usage_df['day_date'], errors='coerce')

subscriptions_df = subscriptions_df.dropna(subset=['trial_starts_at', 'trial_ends_at'])
subscriptions_df['trial_duration'] = (subscriptions_df['trial_ends_at'] - subscriptions_df['trial_starts_at']).dt.days
subscriptions_df = subscriptions_df[subscriptions_df['trial_duration'] == 15].copy()
subscriptions_df['converted'] = subscriptions_df['first_paid_invoice_paid_at'].notnull().astype(int)

usage_cols = [col for col in daily_usage_df.columns if col.startswith('nb_')]

# Prepare sequences
X_seq, y_seq = prepare_sequence_data(daily_usage_df, subscriptions_df, usage_cols)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

learning_rates = [0.01, 0.001, 0.05, 0.005]
results = {}

print("\n" + "="*70)
print("TRANSFORMER LEARNING RATE SEARCH")
print("="*70)

for lr in learning_rates:
    print(f"\n🔬 Testing LR = {lr}")
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = TransformerChurnModel(input_dim=len(usage_cols), d_model=64, num_heads=4, ff_dim=128, num_layers=2, dropout=0.2).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Train
    best_val_auc = 0
    counter = 0
    for epoch in range(50):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x.to(device)).cpu().numpy()
                val_probs.extend(outputs)
                val_labels.extend(batch_y.numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'models/transformer/lr_{lr}_best.pt')
            counter = 0
        else:
            counter += 1
            if counter >= 10:
                break
    
    # Evaluate
    model.load_state_dict(torch.load(f'models/transformer/lr_{lr}_best.pt'))
    test_dataset = SequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.to(device)).cpu().numpy()
            test_probs.extend(outputs)
            test_labels.extend(batch_y.numpy())
    
    test_auc = roc_auc_score(test_labels, test_probs)
    precision, recall, _ = precision_recall_curve(test_labels, test_probs)
    test_pr_auc = auc(recall, precision)
    
    results[lr] = {
        'test_auc': test_auc,
        'test_pr_auc': test_pr_auc,
        'best_val_auc': best_val_auc,
        'epochs': epoch + 1
    }
    
    print(f"✅ LR={lr}: Val AUC={best_val_auc:.4f}, Test AUC={test_auc:.4f}, Epochs={epoch+1}")

# Best LR
best_lr = max(results.items(), key=lambda x: x[1]['best_val_auc'])[0]
print("\n" + "="*70)
print(f"🏆 BEST LEARNING RATE: {best_lr}")
print(f"Validation AUC: {results[best_lr]['best_val_auc']:.4f}")
print(f"Test AUC: {results[best_lr]['test_auc']:.4f}")
print("="*70)

# Save comparison
import pandas as pd
pd.DataFrame(results).T.to_csv('models/transformer/lr_comparison.csv')
print(f"\n✅ Results saved to models/transformer/lr_comparison.csv")
print(f"✅ Best LR: {best_lr}")
