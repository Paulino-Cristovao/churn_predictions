"""
Learning Rate Search for GRU Model
Tests multiple learning rates and selects the best one
"""
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')

from models.lstm.train import (
    prepare_sequence_data, 
    train_lstm_model, 
    evaluate_model,
    plot_training_curves
)

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
print(f'📊 Using {len(usage_cols)} usage features')

# Prepare sequences
print("🔄 Preparing sequential data...")
X_seq, y_seq = prepare_sequence_data(daily_usage_df, subscriptions_df, usage_cols)
print(f'Sequence shape: {X_seq.shape}')
print(f'Conversion rate: {y_seq.mean():.2%}')

# Split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f'Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}')

# Learning rates to test
learning_rates = [0.01, 0.001, 0.05, 0.005]
results = {}

print("\n" + "="*70)
print("LEARNING RATE SEARCH")
print("="*70)

for lr in learning_rates:
    print(f"\n🔬 Testing LR = {lr}")
    print("-" * 70)
    
    # Train model
    model, history = train_lstm_model(
        X_train, y_train, X_val, y_val,
        input_size=len(usage_cols),
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        batch_size=32,
        learning_rate=lr,
        num_epochs=50,
        patience=10,
        device=device,
        save_dir=f'models/lstm/lr_{lr}'
    )
    
    # Evaluate
    model.load_state_dict(torch.load(f'models/lstm/lr_{lr}/best_model.pt'))
    test_results = evaluate_model(model, X_test, y_test, device=device, save_dir=f'models/lstm/lr_{lr}')
    
    # Store results
    results[lr] = {
        'test_auc': test_results['test_auc'],
        'test_pr_auc': test_results['test_pr_auc'],
        'test_accuracy': test_results['test_accuracy'],
        'test_brier': test_results['test_brier'],
        'best_val_auc': history['best_val_auc'],
        'epochs_trained': history['epochs_trained']
    }
    
    print(f"\n✅ LR={lr} Results:")
    print(f"   Test AUC: {test_results['test_auc']:.4f}")
    print(f"   Test PR-AUC: {test_results['test_pr_auc']:.4f}")
    print(f"   Best Val AUC: {history['best_val_auc']:.4f}")
    print(f"   Epochs: {history['epochs_trained']}")

# Find best learning rate
print("\n" + "="*70)
print("LEARNING RATE COMPARISON")
print("="*70)

import pandas as pd
results_df = pd.DataFrame(results).T
results_df.index.name = 'Learning Rate'
print("\n", results_df.round(4))

# Best by validation AUC
best_lr = results_df['best_val_auc'].idxmax()
best_test_auc = results_df.loc[best_lr, 'test_auc']
best_val_auc = results_df.loc[best_lr, 'best_val_auc']
best_epochs = int(results_df.loc[best_lr, 'epochs_trained'])

print("\n" + "="*70)
print(f"🏆 BEST LEARNING RATE: {best_lr}")
print("="*70)
print(f"Validation AUC: {best_val_auc:.4f}")
print(f"Test AUC: {best_test_auc:.4f}")
print(f"Epochs trained: {best_epochs}")
print(f"Convergence: {'Fast' if best_epochs < 20 else 'Moderate' if best_epochs < 35 else 'Slow'}")

# Copy best model to main directory
import shutil
import os
os.makedirs('models/lstm', exist_ok=True)
shutil.copy(f'models/lstm/lr_{best_lr}/best_model.pt', 'models/lstm/best_model.pt')
shutil.copy(f'models/lstm/lr_{best_lr}/results.json', 'models/lstm/results.json')
shutil.copy(f'models/lstm/lr_{best_lr}/training_curves.png', 'models/lstm/training_curves.png')

print(f"\n✅ Copied best model (LR={best_lr}) to models/lstm/")
print(f"✅ All LR experiments saved in models/lstm/lr_*/")

# Save comparison
results_df.to_csv('models/lstm/lr_comparison.csv')
print(f"✅ Results saved to models/lstm/lr_comparison.csv")
